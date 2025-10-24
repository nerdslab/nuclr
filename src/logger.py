import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb
from wandb.sdk.wandb_run import Run
from .utils import get_cli_logger

EPOCH_PBAR_FMT = "{l_bar}{bar}| Epoch {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
STEP_PBAR_FMT = "{l_bar}{bar}| Step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


def skip_if_dummy(func):
    def wrapper(self, *args, **kwargs):
        if self.dummy:
            return
        return func(self, *args, **kwargs)

    return wrapper


class Logger:
    r"""A wrapper for TQDM based progress bar and WandB"""

    _pbar = None
    log_to_pbar = {}
    log_to_wandb = {}
    dummy: bool = False
    wandb_run: Optional[Run] = None
    run_id: str = None
    pbar_prefix: str = ""

    def __init__(self, rank: int):
        self.rank = rank
        self.dummy = rank != 0

        self.cli_log = get_cli_logger(all_ranks=True)
        self.info(
            f"Rank {rank} online",
            all_ranks=True,
        )

    @skip_if_dummy
    def init_wandb(self, cfg: DictConfig):
        Path(cfg.dir).mkdir(exist_ok=True, parents=True)
        wandb.init(**cfg)  # type: ignore
        assert wandb.run is not None
        self.wandb_run = wandb.run
        self.wandb_run.log_code(root=".", include_fn=wandb_code_include_fn)
        self.run_id = self.wandb_run.id

    @skip_if_dummy
    def save_config(self, cfg: DictConfig):
        wandb.config.update(OmegaConf.to_container(cfg), allow_val_change=True)

    def info(self, msg, all_ranks=False):
        if all_ranks or not self.dummy:
            self.cli_log.info(msg, stacklevel=2)

    def warn(self, msg, all_ranks=False):
        if all_ranks or not self.dummy:
            self.cli_log.warn(msg, stacklevel=2)

    def error(self, msg):
        self.cli_log.error(msg, stacklevel=2)

    @skip_if_dummy
    def push(self):
        self._push_pbar()
        self._push_wandb()

    def _push_wandb(self):
        if self.wandb_run is None:
            return

        self.wandb_run.log(self.log_to_wandb)
        self.log_to_wandb = {}

    def _push_pbar(self):
        if self._pbar is None:
            return

        formatted_dict = {k: _pbar_value_format(v) for k, v in self.log_to_pbar.items()}
        desc = " | ".join([f"{k}: {v}" for k, v in formatted_dict.items()])
        if len(self.pbar_prefix) > 0:
            desc = f"[{self.pbar_prefix}] {desc}"
        desc += " "
        self._pbar.set_description(desc)
        self.log_to_pbar = {}

    @skip_if_dummy
    def log(self, name: str, value: Any, pbar: bool | str = False, wandb: bool = True):
        if pbar:
            if isinstance(pbar, str):
                self.log_to_pbar[pbar] = value
            else:
                self.log_to_pbar[name] = value

        if wandb:
            self.log_to_wandb[name] = value

    @skip_if_dummy
    def log_dict(self, data: Dict[str, Any]):
        self.log_to_wandb.update(data)

    def get_pbar(self, iterable: Iterable, prefix: str = "", leave: bool = False):
        self._pbar = tqdm(
            iterable,
            desc=f"[{prefix}]",
            bar_format=STEP_PBAR_FMT,
            total=len(iterable),
            disable=self.dummy,
            dynamic_ncols=True,
            leave=leave,
        )
        self.pbar_prefix = prefix

        for d in self._pbar:
            yield d

        self.pbar_prefix = ""


def _pbar_value_format(value):
    if isinstance(value, str):
        return value

    if isinstance(value, (int, float)):
        if abs(value) < 0.001 or abs(value) > 1000:
            return f"{value:.2e}"
        return f"{value:.3f}"

    return str(value)


def wandb_code_include_fn(path: str):
    _path = Path(path)
    if sys.prefix in str(_path):
        # Ignore any paths containing the Python path (virtual env)
        return False
    if "logs/" in str(_path):
        return False
    if "wandb/" in str(_path):
        return False
    if "torch_brain/" in str(_path):
        return False
    if _path.suffix == ".py":
        return True
    if _path.name == "requirements.txt":
        return True
    if _path.name == "venv_setup.sh":
        return True


def log_on_rank_zero(msg: str):
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    import logging

    logging.info(msg)
