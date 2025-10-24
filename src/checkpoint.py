from typing import Any, Dict
import torch
import os
from pathlib import Path


CheckpointDict = Dict[str, Any]


def save_ckpt(filepath: Path, **kwargs) -> Path:
    checkpoint = {}
    if os.path.exists(".git"):
        checkpoint["git_hash"] = os.popen("git rev-parse HEAD").read().strip()

    for k, v in kwargs.items():
        if isinstance(v, torch.nn.parallel.DistributedDataParallel):
            checkpoint[f"{k}_state_dict"] = v.module.state_dict()
        elif hasattr(v, "state_dict"):
            checkpoint[f"{k}_state_dict"] = v.state_dict()
        else:
            checkpoint[k] = v

    filepath.parent.mkdir(exist_ok=True, parents=True)
    torch.save(checkpoint, filepath)
    return filepath
