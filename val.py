#val.py -

# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Validate a trained YOLOv5 classification model on a classification dataset.

Usage:
    $ bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
    $ python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate ImageNet

Usage - formats:
    $ python3 val/val.py --weights best.pt                 # PyTorch

"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.metrics import (mean_absolute_error,
                             root_mean_squared_error,
                             r2_score,)

from utils.createDataSets import TimeSeriesDataset
from torch.utils.data import DataLoader
RANK = int(os.getenv("RANK", -1))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # EMISSION root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    colorstr,
    intersect_dicts,
    increment_path,
    print_args,
)

from utils.general import try_gpu

# @smart_inference_mode()
def run(
        data="",  # dataset dir
        weights="",  # model.pt path(s)
        project=ROOT / "runs/val-reg",   # save to project/name
        name="exp",  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        criterion=None,
        pbar=None,
        scalers=None,
        ns = 10, # num_steps = 10,
        bs = 32, # batch_size = 32,
        mask_threhold = 0.0,
):
    """Validates a rnn model on a dataset, computing metrics like mae, rmse, r2 et. al."""
    # Initialize/load model and set device

    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()

    else:  # called directly
        device = try_gpu()

        # # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir
        result_file = save_dir/"results.csv"

        # Load model
        weights = str(weights[0]) if isinstance(weights, (list, tuple)) else str(weights)
        ckpt = torch.load(weights, map_location="cpu",weights_only=False)  # 直接加载model
        model = ckpt["model"]
        LOGGER.info(f"Loaded full model from {weights}") # report
        # model.half() if half else model.float()
        import pickle
        scalers = pickle.loads(ckpt["scaler"])

        scaler_X, scaler_y = scalers["X"], scalers["y"]

        # Dataloader
        valid_dir = Path(data) / "valid" # or "test"
        # valid_dir = Path(data) / "test"

        valid_dataset = TimeSeriesDataset(valid_dir,
                                          num_steps=ns,
                                          scaler_X=scaler_X,
                                          scaler_y=scaler_y,
                                          fit_scaler=False)
        dataloader = DataLoader(dataset=valid_dataset,
                                batch_size=bs * 2,
                                drop_last=True,) # i.e valid_loader
    model.eval()
    preds, targets, loss, dt = [], [], 0, (Profile(device=device), Profile(device=device), Profile(device=device))
    n = len(dataloader)  # number of batches
    action = "validating"
    desc = f"{pbar.desc[:-36]}{action:>36}" if pbar else f"{action}"

    bar = tqdm(dataloader, desc, total=n, bar_format=TQDM_BAR_FORMAT, position=0,leave=False if training else True, disable=False)
    with torch.amp.autocast(device_type=device.type, enabled=(device.type != "cpu")):
        for X, y in bar:
            with dt[0]:
                X, y = X.to(device, non_blocking=True), y.to(device)

            # Inference
            with dt[1]:
                y_hat, _ = model(X.to(torch.float32), )
                y_hat = y_hat[:,-1].squeeze()

            with dt[2]:
                preds.append(y_hat.reshape(-1).cpu().detach())
                targets.append(y.reshape(-1).cpu().detach())
                if criterion:
                    loss += criterion(y_hat.reshape((-1)), y.reshape((-1))).mean()

    loss /= n
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()

    preds = scalers["y"].inverse_transform(preds.reshape(-1, 1)).reshape(-1)
    targets = scalers["y"].inverse_transform(targets.reshape(-1, 1)).reshape(-1)

    mask = targets > mask_threhold
    preds_mask = preds[mask]
    targets_mask = targets[mask]
    def get_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        R2 = r2_score(y_true, y_pred)
        # mae = np.mean(np.abs(y_true - y_pred))
        # rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        # ss_res = np.sum((y_true - y_pred) ** 2)
        # ss_tot = np.sum((y_true - np.mean(y_pred)) ** 2)
        # R2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        return mae, rmse, R2

    mae, rmse, R2 = get_metrics(targets_mask, preds_mask)

    if not training:
        df_compare = pd.DataFrame({
            'true_label': targets_mask,
            'prediction': preds_mask,
        })
        df_compare.to_csv(result_file, index=False)
        # ============================baseline mean============================#
        mae_mean, rmse_mean, R2_mean = get_metrics(targets_mask,
                                                   np.full_like(targets_mask, np.mean(targets_mask)))
        mae_naive, rmse_naive, R2_naive = get_metrics(targets_mask[1:], targets_mask[:-1])
        # Validate complete
        if RANK in {-1, 0}:
            LOGGER.info(
                f"\nValidate completed."
                f"\nResults saved to {colorstr('bold', result_file)}"
                f"\nmae: {mae:.2f}\t rmse: {rmse:.2f}\tR2:{R2:.2f}"
            )
            # ---- 保存 metrics ----
            metrics = [
                {"model": "our-model", "mae": round(mae, 2), "rmse": round(rmse, 2), "R2": round(R2, 2)},
                {"model": "Baseline-mean", "mae": round(mae_mean, 2), "rmse": round(rmse_mean,2), "R2": round(R2_mean,2)},
                {"model": "Baseline-naive-t-1", "mae": round(mae_naive, 2), "rmse": round(rmse_naive,2), "R2": round(R2_naive,2)},
            ]

            metrics_file = save_dir / "metrics.csv"
            df = pd.DataFrame(metrics)
            if metrics_file.exists():
                df.to_csv(metrics_file, mode="a", header=False, index=False)
            else:
                df.to_csv(metrics_file, index=False)
            LOGGER.info(f"Validate metrics saved to {colorstr('bold',metrics_file)}")
    else:
        pass

    if pbar:
        pbar.set_description (f"{pbar.desc[:-60]}{loss:>12.3g}{mae:>12.3g}{rmse:>12.3g}{R2:>12.3g}")
    return mae, rmse, R2, loss

def parse_opt():
    """Parses and returns command line arguments for YOLOv5 model evaluation and inference settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "datasets/min-level/processed",help="dataset path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train-reg/exp16/weights/best.pt",help="model.pt path(s)")
    parser.add_argument("--project", default=ROOT / "runs/val-reg", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--mask-threhold", default=20.0, help="validate threhold for mask")
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes the EMISSION model prediction workflow, handling argument parsing and requirement checks."""
    # check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

