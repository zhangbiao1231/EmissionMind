# flame-mind 🔥, 1.0.0 license
"""PyTorch utils."""

import math
import os
import platform
import subprocess
import time
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.general import LOGGER, colorstr

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")
warnings.filterwarnings("ignore", category=UserWarning)


def smartCrossEntropyLoss():
    """Returns a CrossEntropyLoss with optional label smoothing for torch>=1.10.0; warns if smoothing on lower
    versions.
    """
    return nn.CrossEntropyLoss()

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Context manager ensuring ordered operations in distributed training by making all processes wait for the leading
    process.
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])

def grad_clipping(net, theta):
    """梯度裁剪"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params # 自己搭建的网络模型
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def smart_optimizer(model, name="SGD", lr=0.001, momentum=0.9, decay=1e-5):
    """
    Initializes YOLOv5 optimizer with 3 parameter groups for different decay configurations.

    Groups are 0) weights with decay, 1) weights no decay, 2) biases no decay.
    """
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)
    if name == "Adam":
        optimizer = torch.optim.Adam((param for param in model.parameters()
                                      if param.requires_grad),
                                     lr=lr,
                                     betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW((param for param in model.parameters()
                                      if param.requires_grad),
                                      lr=lr,
                                      betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop((param for param in model.parameters()
                                      if param.requires_grad),
                                        lr=lr,
                                        momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD((param for param in model.parameters()
                                      if param.requires_grad),
                                    lr=lr,
                                    momentum=momentum,
                                    nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    # optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    # optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
        f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias'
    )
    return optimizer
# def smart_inference_mode(torch_1_9=check_version(torch.__version__, "1.9.0")):
#     """Applies torch.inference_mode() if torch>=1.9.0, else torch.no_grad() as a decorator for functions."""
#
#     def decorate(fn):
#         """Applies torch.inference_mode() if torch>=1.9.0, else torch.no_grad() to the decorated function."""
#         return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)
#
#     return decorate
def smart_resume(ckpt, optimizer, ema=None, weights="yolov5s.pt", epochs=300, resume=True):
    """Resumes training from a checkpoint, updating optimizer, ema, and epochs, with optional resume verification."""
    best_fitness = 0.0
    start_epoch = ckpt["epoch"] + 1
    if ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
        best_fitness = ckpt["best_fitness"]
    if ema and ckpt.get("ema"):
        ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
        ema.updates = ckpt["updates"]
    if resume:
        assert start_epoch > 0, (
            f"{weights} training to {epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without --resume, i.e. 'python train.py --weights {weights}'"
        )
        LOGGER.info(f"Resuming training from {weights} from epoch {start_epoch} to {epochs} total epochs")
    if epochs < start_epoch:
        LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
        epochs += ckpt["epoch"]  # finetune additional epochs
    return best_fitness, start_epoch, epochs
class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Initializes EMA with model parameters, decay rate, tau for decay adjustment, and update count; sets model to
        evaluation mode.
        """
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """Updates the Exponential Moving Average (EMA) parameters based on the current model's parameters."""
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """Updates EMA attributes by copying specified attributes from model to EMA, excluding certain attributes by
        default.
        """
        copy_attr(self.ema, model, include, exclude)

def copy_attr(a, b, include=(), exclude=()):
    """Copies attributes from object b to a, optionally filtering with include and exclude lists."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)

def smart_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5):
    """
    Initializes YOLOv5 smart optimizer with 3 parameter groups for different decay configurations.

    Groups are 0) weights with decay, 1) weights no decay, 2) biases no decay.
    """
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == "Adam":
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
        f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias'
    )
    return optimizer

def smart_resume(ckpt, optimizer, ema=None, weights="", epochs=300, resume=True):
    """Resumes training from a checkpoint, updating optimizer, ema, and epochs, with optional resume verification."""
    best_fitness = 0.0
    start_epoch = ckpt["epoch"] + 1
    if ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
        best_fitness = ckpt["best_fitness"]
    if ema and ckpt.get("ema"):
        ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
        ema.updates = ckpt["updates"]
    if resume:
        assert start_epoch > 0, (
            f"{weights} training to {epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without --resume, i.e. 'python train.py --weights {weights}'"
        )
        LOGGER.info(f"Resuming training from {weights} from epoch {start_epoch} to {epochs} total epochs")
    if epochs <= start_epoch:
        LOGGER.info(f"{weights} has been trained for {start_epoch} epochs. Fine-tuning for {epochs} more epochs.")
        epochs += start_epoch  # finetune additional epochs
    return best_fitness, start_epoch, epochs
def is_parallel(model):
    """Checks if the model is using Data Parallelism (DP) or Distributed Data Parallelism (DDP)."""
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    """Returns a single-GPU model by removing Data Parallelism (DP) or Distributed Data Parallelism (DDP) if applied."""
    return model.module if is_parallel(model) else model

class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=20, min_delta=1e-3):
        """Initializes simple early stopping mechanism for YOLOv5, with adjustable patience for non-improving epochs."""
        self.best_fitness = -np.inf  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch
        self.min_delta = min_delta

    def __call__(self, epoch, fitness):
        """Evaluates if training should stop based on fitness improvement and patience, returning a boolean."""
        if fitness > self.best_fitness +self.min_delta:  # >= 0+0.01 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        # self.best_fitness <= fitness-self.min_delta
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience # stop training if patience exceeded
        if stop:
            LOGGER.info(
                f"Stopping training early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping."
            )
        return stop