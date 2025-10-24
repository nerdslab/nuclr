import os
from pathlib import Path
from abc import ABC, abstractmethod
from functools import cached_property
import subprocess
from typing import Optional
from omegaconf import DictConfig
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils import (
    setup_ddp,
    seed_everything,
    rank_zero_only,
    expand_path,
)
from src.logger import Logger, EPOCH_PBAR_FMT
from src.checkpoint import save_ckpt, CheckpointDict


class Trainer(ABC):
    def __init__(self, cfg: DictConfig, rank: int, world_size: int):
        seed_everything(cfg.seed)
        setup_ddp(rank, world_size, cfg.ddp)

        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = dist.is_initialized()
        self.device = torch.device(rank)
        self.train_step = 0
        self.epoch = 0
        self.to_ckpt = {}

        if self.cfg.debug is not None:
            self.cfg.wandb.mode = "disabled"
            self.cfg.ckpt.enable = False

        if self.cfg.debug == "tokenizer":
            self.cfg.num_workers = 0

        self._setup_logger()

        if "CKPT_DIR" in os.environ:
            self.cfg.ckpt.dir = os.environ["CKPT_DIR"]
            self.logger.info(f"Overriding ckpt.dir to {self.cfg.ckpt.dir}")

        ckpt = self._load_ckpt()
        self.setup(ckpt)
        self._setup_ckpt()
        self.logger.save_config(self.cfg)

    def train(self):
        if self.cfg.val.first:
            self.val_epoch()

        for epoch in tqdm(
            range(self.epoch, self.cfg.num_epochs),
            initial=self.epoch,
            total=self.cfg.num_epochs,
            bar_format=EPOCH_PBAR_FMT,
            dynamic_ncols=True,
            disable=(self.rank != 0),
        ):
            self.epoch = epoch
            self.barrier()
            self.train_epoch()
            self._save_checkpoint()
            if (self.epoch % self.cfg.val.every_n_epochs == 0) or (  # if user asked
                self.epoch == self.cfg.num_epochs - 1  # if last epoch
            ):
                self.val_epoch()

    def train_epoch(self): ...

    def val_epoch(self): ...

    def _setup_logger(self):
        self.logger = Logger(self.rank)
        self.logger.init_wandb(self.cfg.wandb)
        self.logger.info(f"Run ID: {self.logger.run_id}")
        if self.cfg.wandb.mode != "online":
            self.logger.warn(f"W&B mode: {self.cfg.wandb.mode}")

    @abstractmethod
    def setup(self, ckpt: Optional[CheckpointDict]):
        """Set up the model, loss function, and optimizer.

        This abstract method must be implemented by subclasses to initialize:
        - self.model: Your model (an nn.Module)
        - self.loss: Your loss function (could be an nn.Module)
        - self.optimizer: The optimizer for training
        - self.train_loader: a torch.utils.data.DataLoader
        """
        ...

    @rank_zero_only
    def add_checkpoint_items(self, **kwargs):
        self.to_ckpt.update(kwargs)

    @rank_zero_only
    def _setup_ckpt(self):
        if not self.cfg.ckpt.enable:
            self.logger.warn("Checkpointing disabled")
            self.ckpt_dir = None
            return

        # Create checkpoint directory
        self.ckpt_dir = expand_path(self.cfg.ckpt.dir) / self.logger.run_id
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"Checkpoint directory: {self.ckpt_dir}")

        # Save git diff patch
        if os.path.exists(".git"):
            diff = subprocess.check_output(["git", "diff", "HEAD"])
            diff_path = self.ckpt_dir / "diff.patch"
            diff_path.write_bytes(diff)
            self.logger.wandb_run.log_artifact(diff_path)

            # Save git commit ID
            commit_id = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("utf-8")
                .strip()
            )
            commit_path = self.ckpt_dir / "commit_id.txt"
            commit_path.write_text(commit_id)
            self.logger.info(f"Saved diff.patch and commit_id.txt")
        else:
            self.logger.warn(
                f"Not a git repo. Not saving diff.patch and commid_id.txt."
            )

        # Tell user what will be checkpointed
        if len(self.to_ckpt) > 0:
            self.logger.info(f"Checkpoints will track: {list(self.to_ckpt.keys())}")
        else:
            self.logger.warn(f"No items specified for checkpointing")

    @rank_zero_only
    def _save_checkpoint(self):
        if self.ckpt_dir is None:
            return

        every_n_epochs = self.cfg.ckpt.every_n_epochs
        if (every_n_epochs is None) or (self.epoch % every_n_epochs != 0):
            return

        self.last_ckpt_path = save_ckpt(
            filepath=self.ckpt_dir / f"epoch_{self.epoch}.pt",
            train_step=self.train_step,
            epoch=self.epoch,
            run_id=self.logger.run_id,
            cfg=self.cfg,
            **self.to_ckpt,
        )

    def _load_ckpt(self) -> CheckpointDict | None:
        if self.cfg.ckpt.load_from is None:
            return

        ckpt_path = expand_path(self.cfg.ckpt.load_from)
        if not ckpt_path.exists():
            # try this other path
            ckpt_path = expand_path(self.cfg.ckpt.dir) / self.cfg.ckpt.load_from
            if not ckpt_path.exists():
                raise ValueError(
                    f"Checkpoint not found at {self.cfg.ckpt.load_from} or {ckpt_path}"
                )
        self.cfg.ckpt.load_from = ckpt_path

        self.logger.info(f"Loading checkpoint {ckpt_path}")
        ckpt = torch.load(
            ckpt_path,
            map_location=self.device,
            weights_only=False,
        )

        if self.cfg.ckpt.resume:
            self.epoch = ckpt["epoch"] + 1
            self.train_step = ckpt["train_step"]
            self.logger.info(
                f"Resuming from: epoch = {self.epoch}, train_step = {self.train_step}"
            )

        return ckpt

    def make_ddp(self, module: torch.nn.Module):
        if self.is_distributed:
            ret = DDP(module, device_ids=[self.rank])
            ret = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ret)
            return ret
        else:
            return module

    def barrier(self):
        if self.is_distributed:
            dist.barrier()

    def log_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            name = param_group["name"] if "name" in param_group else f"group_{i}"
            self.logger.log(f"lr/{name}", param_group["lr"])

    def log_weight_stats(self):
        if not self.cfg.log_weights:
            return

        for name, module in self.__dict__.items():
            if isinstance(module, torch.nn.Module):
                self.logger.log_dict(
                    {
                        f"{name}_weight_norm/{k}": v.norm()
                        for k, v in module.named_parameters()
                    }
                )
                self.logger.log_dict(
                    {
                        f"{name}_weight_max/{k}": v.max()
                        for k, v in module.named_parameters()
                    }
                )
                self.logger.log_dict(
                    {
                        f"{name}_weight_min/{k}": v.min()
                        for k, v in module.named_parameters()
                    }
                )

    def log_grad_stats(self):
        if not self.cfg.log_grads:
            return

        for name, module in self.__dict__.items():
            if isinstance(module, torch.nn.Module):
                # Individual norms
                param_grad_norms = {
                    f"{name}_grad_norm/{k}": v.grad.norm()
                    for k, v in module.named_parameters()
                    if v.grad is not None
                }
                self.logger.log_dict(param_grad_norms)
                # Total norms
                total_grad_norm = torch.norm(
                    torch.stack(list(param_grad_norms.values()))
                )
                self.logger.log(f"{name}_grad_norm/total", total_grad_norm)
                # Max
                self.logger.log_dict(
                    {
                        f"{name}_grad_max/{k}": v.grad.max()
                        for k, v in module.named_parameters()
                        if v.grad is not None
                    }
                )
                # Min
                self.logger.log_dict(
                    {
                        f"{name}_grad_min/{k}": v.grad.min()
                        for k, v in module.named_parameters()
                        if v.grad is not None
                    }
                )

    @cached_property
    def train_unit_ids(self):
        return self.train_loader.dataset.get_unit_ids()

    @cached_property
    def val_unit_ids(self):
        return self.val_loader.dataset.get_unit_ids()

    @cached_property
    def train_session_ids(self):
        return self.train_loader.dataset.get_session_ids()

    @cached_property
    def val_session_ids(self):
        return self.val_loader.dataset.get_session_ids()

    def push_logs(self):
        self.logger.log("train/epoch", self.epoch)
        self.logger.log("train/step", self.train_step)
        self.logger.push()
