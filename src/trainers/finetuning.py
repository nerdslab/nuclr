from collections import defaultdict
from typing import Optional
from hydra.utils import instantiate

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader

from torch_brain.data.sampler import RandomFixedWindowSampler, SequentialFixedWindowSampler

from src.models import BaseNeuronEncoder
from src.losses import Loss
from src.dataset import Dataset, ViewData
from src.utils import (
    Precision,
    get_num_params,
    get_num_trainable_params,
    validate_weights,
)
from src.trainers.trainer import Trainer, CheckpointDict


class FinetuningTrainer(Trainer):
    def setup(self, ckpt: Optional[CheckpointDict]):
        if dist.is_initialized():
            raise RuntimeError(
                "Finetuning script only supports non-DDP mode"
            )

        if ckpt is None:
            self.logger.warn(
                f"No checkpoint provided. Will initialize model from scratch"
            )

        self._setup_model(ckpt)
        self._setup_train_loader()
        self._setup_val_loader()

        self.model = self.make_ddp(self.model)
        self.loss = self.make_ddp(self.loss)

    def train_epoch(self):
        self.model.train()
        self.loss.train()

        for batch in self.logger.get_pbar(self.train_loader, "Train"):
            batch: ViewData
            batch.to(self.device)

            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                with torch.no_grad():
                    y = self.model(**batch.enc_input)

                loss, logits = self.loss(y, batch, prefix="train", logger=self.logger)

                assert y.isfinite().all(), "Y is not finite"
                assert logits.isfinite().all(), "Logits are not finite"
                assert loss.isfinite().all(), "Loss is not finite"

                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            validate_weights(self.model)
            validate_weights(self.loss)
            self.log_weight_stats()
            self.log_grad_stats()
            self.log_lr()
            self.push_logs()

            self.train_step += 1

    @torch.inference_mode  # type: ignore
    def val_epoch(self):  # type: ignore
        if self.cfg.debug == "train":
            return

        self.model.eval()
        self.loss.eval()

        self.loss.init_cache(self.val_unit_ids, self.device)
        running_loss_dict = defaultdict(lambda: 0.0)
        for batch in self.logger.get_pbar(self.val_loader, "Test"):
            batch: ViewData
            batch.to(self.device)

            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                y = self.model(**batch.enc_input)
                self.loss.update(y, batch)
        
        loss_dict = self.loss.compute(prefix="val")
        self.logger.log_dict(loss_dict)
        self.push_logs()

    def _setup_model(self, ckpt: Optional[CheckpointDict]):

        # Setup model
        self.precision = Precision(self.cfg.precision)
        self.cfg.model.ctx_duration = self.cfg.views.duration
        self.model: BaseNeuronEncoder = instantiate(
            self.cfg.model, precision=self.precision
        ).to(self.device)
        self.loss: Loss = instantiate(self.cfg.loss, model=self.model).to(self.device)

        # Setup optimizer
        params = [
            {"name": "model", "params": self.model.parameters(), "lr": 1e-6},
            {"name": "loss", "params": self.loss.parameters()},
        ]
        self.optimizer: torch.optim.Optimizer = instantiate(self.cfg.optim, params)

        # Apply checkpoint
        if ckpt is not None:
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        
        # Log info
        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.logger.info(f"Loss: {self.loss.__class__}")
        self.logger.info(f"Optimizer: {self.optimizer.__class__}")

        num_params = get_num_params(self.model) + get_num_params(self.loss)
        self.logger.info(f"Num params: {num_params:,}")

        num_trainable_params = get_num_trainable_params(self.model)
        num_trainable_params += get_num_trainable_params(self.loss)
        self.logger.info(f"Num trainable params: {num_trainable_params:,}")

    def _setup_train_loader(self):
        ds = Dataset(root=self.cfg.data.root, config=self.cfg.data.train_dataset)
        ds.transform = self.model.tokenize

        sampler=RandomFixedWindowSampler(
            sampling_intervals=ds.get_sampling_intervals(),
            window_length=self.cfg.views.duration,
            generator=torch.Generator().manual_seed(self.cfg.seed),
            drop_short=True,
        )

        multi_wrkrs = self.cfg.num_workers > 0
        self.train_loader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            collate_fn=ViewData.collate,
            multiprocessing_context=mp.get_context("fork") if multi_wrkrs else None,
            persistent_workers=multi_wrkrs,
        )

        self.logger.info(f"Train sessions: {len(self.train_session_ids)}")
        self.logger.info(f"Train units: {len(self.train_unit_ids)}")
        self.logger.info(f"Train steps/epoch: {len(self.train_loader)}")

    def _setup_val_loader(self):
        ds = Dataset(root=self.cfg.data.root, config=self.cfg.data.val_dataset)
        ds.transform = self.model.tokenize  # type: ignore

        sampler=SequentialFixedWindowSampler(
            sampling_intervals=ds.get_sampling_intervals(),
            window_length=self.cfg.views.duration,
            step=0.5 * self.cfg.views.duration,
            drop_short=False,
        )

        multi_wrkrs = self.cfg.num_workers > 0
        self.val_loader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            collate_fn=ViewData.collate,
            multiprocessing_context=mp.get_context("fork") if multi_wrkrs else None,
            persistent_workers=multi_wrkrs,
        )

        self.logger.info(f"Val sessions: {len(self.val_session_ids)}")
        self.logger.info(f"Val units: {len(self.val_unit_ids)}")
        self.logger.info(f"Val steps/epoch: {len(self.val_loader)}")
