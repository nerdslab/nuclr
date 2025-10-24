import os
import sys
from pathlib import Path
import copy
from typing import Any, Union
import numpy as np
from omegaconf import OmegaConf, DictConfig
import torch
import torch.distributed as dist

from temporaldata import IrregularTimeSeries
import logging


class Precision:
    def __init__(self, precision_str: str):
        self.precision_str = precision_str

        if precision_str == "fp32":
            self.dtype = torch.float32
        elif precision_str == "fp16":
            self.dtype = torch.float16
        elif precision_str == "bf16":
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"Invalid precision string: {precision_str}")

    def __repr__(self):
        return self.precision_str

    def __str__(self):
        return self.precision_str


def bin_spikes(
    spikes: IrregularTimeSeries,
    ctx_size: float,
    num_units: int,
    bin_size: float,
    right: bool = True,
) -> np.ndarray:
    r"""Adapted from torch_brain/utils/binning.py"""

    discard = ctx_size - np.floor(ctx_size / bin_size) * bin_size
    if discard != 0:
        if right:
            start += discard
        else:
            end -= discard
        # reslice
        spikes = spikes.slice(start, end)

    num_bins = round(ctx_size / bin_size)

    rate = 1 / bin_size  # avoid precision issues
    binned_spikes = np.zeros((num_units, num_bins))
    bin_index = np.floor((spikes.timestamps) * rate).astype(int)
    np.add.at(binned_spikes, (spikes.unit_index, bin_index), 1)

    return binned_spikes


def move_to_device(data, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    else:
        raise TypeError(f"Unknown data type {type(data)}")


def setup_ddp(rank, world_size, ddp_cfg):
    import os

    if (world_size > 1) or ddp_cfg.force:
        os.environ["MASTER_ADDR"] = str(ddp_cfg.master_addr)
        os.environ["MASTER_PORT"] = str(ddp_cfg.master_port)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        get_cli_logger().info(f"Skipping DDP setup")


def get_open_port() -> int:
    """Find an available port on the system.

    This function creates a temporary socket, binds it to port 0 to let the OS assign
    an available port, and returns that port number. The socket is automatically closed
    after getting the port.

    Returns:
        int: An available port number that can be used for network communication.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed):
    import random, torch, numpy

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank_zero_only(func):
    def wrapper(self, *args, **kwargs):
        if self.rank != 0:
            return
        return func(self, *args, **kwargs)

    return wrapper


def expand_path(path: Union[str, Path]) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path)))


def get_num_params(module: torch.nn.Module):
    return sum(p.numel() for p in module.parameters())


def get_num_trainable_params(module: torch.nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def validate_weights(module: torch.nn.Module):
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module but got {type(module)}")

    nan_weights = []
    for k, v in module.named_parameters():
        if not v.isfinite().all():
            nan_weights.append(k)
    if len(nan_weights) != 0:
        raise RuntimeError(f"Found NaN weights in layers: {nan_weights}")


def get_cli_logger(all_ranks=False):

    # Custom filter to convert absolute paths to relative paths
    class RelativePathFilter(logging.Filter):
        def filter(self, record):
            if hasattr(record, "pathname"):
                record.relpath = os.path.relpath(record.pathname)
            return True

    LOGGING_FORMAT = (
        "\033[1;33m%(levelname)s\033[0m | "
        "\033[1;36m%(asctime)s\033[0m | "
        "\033[1;32m%(relpath)s:%(lineno)d\033[0m | "
        "%(message)s"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=LOGGING_FORMAT,
        force=True,
        stream=sys.stdout,
    )
    logger = logging.getLogger()
    logger.addFilter(RelativePathFilter())
    if not all_ranks:
        if dist.is_initialized() and not dist.get_rank() == 0:
            logger.setLevel(logging.ERROR)
    return logger


def get_class_from_path(path: str):
    import importlib

    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def instantiate(dictconfig: DictConfig | dict, *args, **kwargs) -> Any:
    """
    Instantiate a class from a configuration dictionary.

    Args:
        cfg: Configuration with _target_ field specifying the class path
        **kwargs: Additional arguments to pass to the class constructor

    Returns:
        An instance of the specified class
    """
    if isinstance(dictconfig, DictConfig):
        cfg_dict = OmegaConf.to_container(dictconfig)
    else:
        cfg_dict = copy.deepcopy(dictconfig)

    target = cfg_dict.pop("_target_", None)
    assert target is not None
    cls = get_class_from_path(target)
    return cls(*args, **cfg_dict, **kwargs)


def is_divisible(x: float, y: float, tol: float = 1e-6) -> bool:
    """Returns true if x is divisible by y"""
    return abs(x / y - round(x / y)) < tol


def apply_unit_mask_(data, mask):
    num_units_before = len(data.units)
    data.units = data.units.select_by_mask(mask)
    num_units_after = len(data.units)

    old_unit_idx = np.arange(num_units_before)[mask]
    unit_idx_remap = np.zeros(num_units_before, dtype=int)
    unit_idx_remap[old_unit_idx] = np.arange(num_units_after)

    spike_mask = np.isin(data.spikes.unit_index, old_unit_idx)
    data.spikes = data.spikes.select_by_mask(spike_mask)
    data.spikes.unit_index = unit_idx_remap[data.spikes.unit_index]

    return data
