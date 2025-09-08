# emission, 1.0.0 license
"""
Train a emission predict model on a regression dataset.

Usage - None-GPU training:
    $ python train.py --model best.pt --data warmup --epochs 100

Datasets:           --data, 'path/to/data'
emission models:  --model best.pt
"""
import argparse
import os
import subprocess
import math
import numpy  as np
import sys
import time
# from copy import deepcopy
from datetime import datetime
from pathlib import Path


import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import *

from utils.loggers import GenericLogger,SummaryWriter

from utils.general import (
    DATASETS_DIR,
    LOGGER,
    TQDM_BAR_FORMAT,
    colorstr,
    increment_path,
    init_seeds,
    yaml_save,
    print_args,
    try_gpu,
)

from utils.createDataSets import TimeSeriesDataset
from torch.utils.data import DataLoader
from model import RNN

from utils.torch_utils import (
    grad_clipping,
    smart_resume,
    smart_optimizer,
    EarlyStopping,
)

import val as validate

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # flame root directory#
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

def train(opt, device):
    """Trains an emission model, managing datasets, model optimization, logging, and saving checkpoints."""
    # init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir, weights, bs, ns, epochs, resume, nw, pretrained, is_train, nh, nl,dims = (
        opt.save_dir,
        opt.weights,
        opt.batch_size,
        opt.num_steps,
        opt.epochs,
        opt.resume,
        min(os.cpu_count() - 1, opt.workers),
        str(opt.pretrained).lower() == "true",
        opt.is_train,
        opt.num_hiddens,
        opt.num_layers,
        opt.dims,
    )
    cuda = device.type != "cpu"

   # Directories
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / "last.pt", wdir / "best.pt"

    # Save run settings
    yaml_save(save_dir / "opt.yaml", vars(opt))

    # Logger
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None

    # tensorboard
    writer = SummaryWriter(log_dir=str(save_dir))

    # Dataloaders
    train_dir = DATASETS_DIR / "train"
    # train_dir = opt.train_folder
    train_dataset = TimeSeriesDataset(train_dir,
                                      num_steps=ns,
                                      scaler_X=None,
                                      scaler_y=None,
                                      fit_scaler=True,)
    scaler_X, scaler_y = train_dataset.scaler_X, train_dataset.scaler_y
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=bs,
                              drop_last=True,)
    print("y scaler min:", scaler_y.data_min_, "max:", scaler_y.data_max_)
    for X, y in train_loader:
        print("X shape:", X.shape, "y shape:", y.shape)
        break

    import pickle
    scalers = {"X": scaler_X, "y": scaler_y}
    scaler_bytes = pickle.dumps(scalers)

    valid_dir = DATASETS_DIR / "valid"
    if RANK in {-1, 0}:
        valid_dataset = TimeSeriesDataset(valid_dir,
                                          num_steps=ns,
                                          scaler_X=scaler_X,
                                          scaler_y=scaler_y,
                                          fit_scaler=False)
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=bs*2,
                                  drop_last=True,)

    # model
    # create model
    get_rnn_layer = RNN.get_rnn_layer(input_size=dims,
                                      num_hiddens=nh,
                                      num_layers=nl,
                                      dropout=opt.dropout,)
    rnn_layer = get_rnn_layer.construct_rnn(selected_model=opt.model_name)

    net = RNN.RNNModel(rnn_layer=rnn_layer)  # out_feature = 1

    print(net)

    net.to(device)
    net.train()
    weights = str(weights[0]) if isinstance(weights, (list, tuple)) else str(weights) # 需要改

    pretrained = str(weights).endswith(".pt") and pretrained
    print(weights)
    print(pretrained)

    if pretrained:
        # load model directly
        weights = str(weights[0]) if isinstance(weights, (list, tuple)) else str(weights)
        ckpt = torch.load(weights, map_location="cpu",weights_only=False)
        net = ckpt["model"]
        # net.load_state_dict(ckpt["model_state_dict"])
        LOGGER.info(f"Loaded full model from {weights}")  # report
    else:  # create
        net = net.to(device)
    # Info
    if RANK in {-1, 0}:
        if opt.verbose:
            LOGGER.info(net)

    # Optimizer
    optimizer = smart_optimizer(model=net,
                                 name=opt.optimizer,
                                 lr=opt.lr0,
                                 momentum=0.9,
                                 decay=opt.decay)
    # Scheduler
    lrf = opt.lrf  # final lr (fraction of lr0)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine

    def lf(x):
        """Linear learning rate scheduler function, scaling learning rate from initial value to `lrf` over `epochs`."""
        return (1 - x / epochs) * (1 - lrf) + lrf  # linear

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf) # cosine method
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_period, opt.lr_decay)

    # Resume
    best_fitness, start_epoch = -np.inf, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, None, weights, epochs, resume)
        del ckpt

    # Train
    t0 = time.time()
    scheduler.last_epoch = start_epoch - 1
    criterion = nn.MSELoss(reduction="none")
    # a ,b = torch.ones(5),  torch.tensor([1,2,3,4,5])
    # tensor(6.0) default = "mean"
    # tensor(30.0) reduction = "sum"
    # tensor([ 0.0, 1.0, 4.0, 9.0, 16.0]) reduction = "none"
    best_fitness = 0.0
    stopper, stop = EarlyStopping(patience=opt.patience, min_delta=opt.min_delta), False
    val = valid_dir.stem  # 'valid' or 'test'
    LOGGER.info(
        f'data iter {len(train_loader)} train, {len(valid_loader)} valid\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f'Starting training emission model for {epochs} epochs...\n\n'
        f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'{val}_loss':>14}{f'{val}_mae':>13}{f'{val}_rmse':>13}{f'{val}_R2':>13}"
    )

    state = None
    # Train loop
    for epoch in range(start_epoch, epochs): # loop over the datasets multiple times
        tloss, vloss, fitness = 0.0, 0.0, 0.0 # train loss, val loss, fitness

        pbar = enumerate(train_loader)
        # state = None

        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
        for i, (X,y) in pbar: # progress bar
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)

            if state is None:
                state = net.begin_state(batch_size=X.shape[0], device=device)
            else:
                if isinstance(state, tuple):
                    state = tuple(s.detach_() for s in state)
                else:
                    state = state.detach_()
            # Forward
            y_hat, state = net(X, state)
            y_hat = y_hat[:,-1]
            # print(y_hat.shape, y.shape)
            l = criterion(y_hat, y).mean()
            # Backward
            l.backward()
            # Optimize
            grad_clipping(net=net, theta=1) # clip gradients
            optimizer.step()
            if RANK in {-1, 0}:
                # Print
                tloss = (tloss*i + l.item()) / (i+1) # update mean loss
                mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)  # (GB)
                pbar.set_description(
                    f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + " " * 60
                )

                # Validate metrics
                mae, rmse, R2 = 0.0, 0.0, 0.0

                if i == len(pbar) - 1: # last batch
                    mae, rmse, R2, vloss = validate.run(
                        model=net, dataloader=valid_loader,criterion=criterion, pbar=pbar,scalers=scalers,
                    )
                    fitness = R2

        # Scheduler
        scheduler.step()
        stop = stopper(epoch=epochs, fitness=fitness) # Early stopping

        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness

            # Logger
            metrics = {
                "train/loss": tloss,
                f"{val}/loss": vloss,
                "metrics/mae": mae,
                "metrics/rmse": rmse,
                "metrics/R2": R2,
                "lr/0": optimizer.param_groups[0]["lr"],
            }  # learning rate
            logger.log_metrics(metrics, epoch + 1)
            writer.add_scalars(main_tag="training over epoch",
                               tag_scalar_dict={f"train/loss": tloss,
                                                f"{val}/loss": vloss,
                                                f"metrics/mae": mae,
                                                f"metrics/rmse": rmse,
                                                f"metrics/R2": R2,},
                               global_step=epoch, )
            for k, v in metrics.items():
                writer.add_scalar(tag=k,
                                  scalar_value=v,
                                  global_step=epoch)

            # Save model
            final_epoch = epoch + 1 == epochs
            if (not opt.nosave) or final_epoch:
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model_state_dict": net.state_dict(),  # 保存模型权重
                    "model": net,  # 保存整个模型对象（含权重）
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "date": datetime.now().isoformat(),
                    "scaler": scaler_bytes}
                # Save last, best and delete
                torch.save(ckpt, last)
                # TODO：add ckpt (save: state_dict & TorchScript tw0 version)
                # model_scripted = torch.jit.script(net)
                # model_scripted.save(wdir)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, wdir / f"epoch{epoch}.pt")
                del ckpt

        #  EarlyStopping
        if stop:
            break

    # Train complete
    if RANK in {-1, 0}:
        LOGGER.info(
            f"\n{epochs - start_epoch} epochs completed in {(time.time() - t0) / 3600:.3f} hours."
            f"\nResults saved to {colorstr('bold', save_dir)}"
            f'\nPredict:         python3 predict.py --weights {best} --source example.csv'
            f'\nValidate:        python3 val.py --weights {best}'
        )


def parse_opt(known=False):
    """Parses command line arguments for EmissionV1 training including model path, dataset, epochs, and more, returning
    parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train-reg/exp20/weights/best.pt",
                        help="model.pt path(s)")
    # ==============================================================about data===================================================================#
    parser.add_argument("--train-folder", nargs="+",type=str, default=ROOT / "datasets/min-level/train", help="train dataset path")
    parser.add_argument("--valid-folder",nargs="+", type=str, default=ROOT / "datasets/min-level/valid", help="valid dataset path")
    parser.add_argument("--use-random", type=bool, default=False, help="use random concat")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--num-steps", type=int, default=10, help="number of steps")

    # ==============================================================about model===================================================================#

    parser.add_argument("--num-hiddens", type=int, default=32, help="number of hidden")
    parser.add_argument("--num-layers", type=int, default=2, help="number of layers")
    parser.add_argument("--dims", type=int, default=14, help="input size")
    parser.add_argument("--model-name", type=str, default="rnn", help="select rnn model, i.e. RNN, GRU, LSTM et.al")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")

    # ==============================================================about train===================================================================#
    parser.add_argument("--epochs", type=int, default=1000, help="total training epochs")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help='--cache images in "ram" (default) or "disk"')
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train-reg", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--pretrained", nargs="?", const=True, default=False, help="start from i.e. --pretrained False")
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW", "RMSProp"], default="Adam", help="optimizer")
    parser.add_argument("--lr0", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--lrf", type=float, default=1e-1, help="terminal learning rate")
    parser.add_argument("--lr-period", type=int, default=50, help="learning rate period")
    parser.add_argument("--lr-decay", type=float, default=0.9, help="learning rate * decay over period per")
    parser.add_argument("--decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local-rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")
    parser.add_argument("--is-train", default=False, help="")
    parser.add_argument("--patience", type=int, default=20, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--min-delta", type=float, default=0.01,help="EarlyStopping Minimum Delta (epochs without improvement)")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[1], help="Freeze layers: backbone=10, first3=0 1 2")
    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    """Executes YOLOv5 training with given options, handling device setup and DDP mode; includes pre-training checks."""
    if RANK in {-1, 0}:
        print_args(vars(opt))
        # check_requirements(ROOT / "requirements.txt")
    device = try_gpu()
    # Parameters
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # Train
    train(opt, device)

def run(**kwargs):
    """
    Executes YOLOv5 model training or inference with specified parameters, returning updated options.

    Example: from yolov5 import classify; classify.train.run(data=mnist, imgsz=320, model='yolov5m')
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)