from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from torch_brain.data import Dataset
from torch_brain.data.sampler import SequentialFixedWindowSampler
from torch_brain.transforms import Compose

from src.models import BaseNeuronEncoder
from src.losses import Loss
from src.dataset import TwoViewDataset, TwoViewData, ViewData
from src.samplers import (
    RandomFixedWindowSampler,
    DistributedSamplerWrapper,
)
from src.utils import (
    Precision,
    rank_zero_only,
    get_num_params,
    get_num_trainable_params,
    validate_weights,
)
from src.evaluators.collector import EmbeddingCollector
from src.trainers.trainer import Trainer, CheckpointDict
from src.utils import instantiate


class ContrastiveTrainer(Trainer):
    def setup(self, ckpt: Optional[CheckpointDict]):
        self.cfg.model.ctx_duration = self.cfg.views.duration

        self._setup_model()
        self._setup_train_loader()
        self._setup_val_loader()
        self._setup_embedding_collectors()
        self._setup_evaluator()

        self._setup_optimizer()
        self._apply_ckpt(ckpt)
        self.model = self.make_ddp(self.model)  # type: ignore
        self.loss = self.make_ddp(self.loss)  # type: ignore

    def train_epoch(self):
        self.model.train()
        self.loss.train()
        self.train_emb_collector.reset()
        self.train_loader.sampler.set_epoch(self.epoch)  # pyright: ignore

        for batch in self.logger.get_pbar(self.train_loader, "Train"):
            self.lr_step()

            batch: TwoViewData
            batch.to(self.device)

            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                y1 = self.model(**batch.view1.data["enc_input"])
                y2 = self.model(**batch.view2.data["enc_input"])

                self.train_emb_collector.update(y1, batch.view1.unit_ids)

                loss = self.loss(
                    y1,
                    y2,
                    batch,
                    prefix="train",
                    logger=self.logger,
                )

                assert y1.isfinite().all(), "Y1 is not finite"
                assert y2.isfinite().all(), "Y2 is not finite"
                assert loss.isfinite().all(), "Loss is not finite"

                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg.grad_clip.enable:
                    max_norm = self.cfg.grad_clip.max_norm
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    torch.nn.utils.clip_grad_norm_(self.loss.parameters(), max_norm)
                self.optimizer.step()

            validate_weights(self.model)
            validate_weights(self.loss)
            self.log_weight_stats()
            self.log_grad_stats()
            self.log_lr()
            self.push_logs()

            self.train_step += 1

    def lr_step(self):
        """Currently only implements linear-rise, cosine decay schedule"""
        lr = self.cfg.lr.max

        if self.cfg.lr.rise is not None:
            rise_until_step = self.cfg.lr.rise * self.steps_per_epoch
            if self.train_step < rise_until_step:
                lr = lr * self.train_step / rise_until_step

        if self.cfg.lr.decay is not None:
            decay_start_epoch = self.cfg.lr.decay
            if isinstance(decay_start_epoch, float):
                decay_start_epoch = decay_start_epoch * self.cfg.num_epochs

            if self.epoch >= decay_start_epoch:
                decay_start_step = decay_start_epoch * self.steps_per_epoch
                decay_step = self.train_step - decay_start_step
                total_steps = self.steps_per_epoch * self.cfg.num_epochs
                total_decay_steps = total_steps - decay_start_step
                lr = lr * (1 + np.cos(np.pi * decay_step / total_decay_steps)) * 0.5

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @torch.inference_mode()
    def val_epoch(self):
        if self.cfg.debug == "train":
            return

        self.model.eval()
        self.loss.eval()
        self.val_emb_collector.reset()

        if not self.cfg.val.loss:
            for batch in self.logger.get_pbar(self.val_loader, "Test"):
                assert isinstance(batch, ViewData)
                batch.to(self.device)

                with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                    y1 = self.model(**batch.data["enc_input"])
                    self.val_emb_collector.update(y1, batch.unit_ids)

        else:
            running_loss_dict = defaultdict(lambda: 0.0)
            steps = 0
            for batch in self.logger.get_pbar(self.val_loader, "Test"):
                assert isinstance(batch, TwoViewData)
                batch.to(self.device)

                with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                    y1 = self.model(**batch.view1.data["enc_input"])
                    y2 = self.model(**batch.view2.data["enc_input"])
                    self.val_emb_collector.update(y1, batch.view1.unit_ids)
                    loss_dict = self.loss(
                        y1,
                        y2,
                        batch,
                        return_loss_dict=True,
                        prefix="val",
                        logger=self.logger,
                    )

                for k, v in loss_dict.items():
                    running_loss_dict[k] = v + running_loss_dict[k]
                steps += 1

            # Average loss and log
            for k, v in running_loss_dict.items():
                running_loss_dict[k] /= steps
            self.logger.log_dict(running_loss_dict)

        self._run_evaluator()
        self.push_logs()

    def _setup_model(self):
        self.precision = Precision(self.cfg.precision)
        self.model: BaseNeuronEncoder = instantiate(
            self.cfg.model, precision=self.precision
        ).to(self.device)
        self.loss: Loss = instantiate(self.cfg.loss, model=self.model).to(self.device)

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.logger.info(f"Loss: {self.loss.__class__}")

        num_params = get_num_params(self.model) + get_num_params(self.loss)
        self.logger.info(f"Num params: {num_params:,}")

        num_trainable_params = get_num_trainable_params(self.model)
        num_trainable_params += get_num_trainable_params(self.loss)
        self.logger.info(f"Num trainable params: {num_trainable_params:,}")

        self.add_checkpoint_items(
            model=self.model,
            loss=self.loss,
        )

    def _setup_optimizer(self):
        decay = []
        no_decay = []

        def update_params(module: torch.nn.Module):
            for name, param in module.named_parameters():
                if "norm" in name or "bias" in name:
                    no_decay.append(param)
                else:
                    decay.append(param)

        update_params(self.model)
        update_params(self.loss)

        params = [
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        self.optimizer: torch.optim.Optimizer = instantiate(self.cfg.optim, params)
        self.logger.info(f"Optimizer: {self.optimizer.__class__}")

        self.add_checkpoint_items(
            optimizer=self.optimizer,
        )

    def _apply_ckpt(self, ckpt: Optional[CheckpointDict]):
        if ckpt is None:
            return

        self.model.load_state_dict(ckpt["model_state_dict"])
        if self.cfg.ckpt.resume:
            self.loss.load_state_dict(ckpt["loss_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        elif self.cfg.ckpt.model_only:
            self.loss.load_state_dict(ckpt["loss_state_dict"])

    def _setup_train_loader(self):
        ds = TwoViewDataset(
            root=self.cfg.data.root,
            config=self.cfg.data.train_dataset,
            unit_id_prefix_fn=lambda _: "",
        )
        ds.transform = Compose(
            [
                instantiate(self.cfg.train_transform),
                self.model.tokenize,
            ]
        )
        ds.common_transform = Compose([TwoViewData.from_two_views, self.loss.tokenize])  # type: ignore
        ds.set_second_view_props(
            window_length=self.cfg.views.duration,
            max_time_diff=self.cfg.views.max_distance,
            seed=self.cfg.seed,
        )

        sampler = DistributedSamplerWrapper(
            RandomFixedWindowSampler(
                sampling_intervals=ds.sanitized_sampling_intervals,
                window_length=self.cfg.views.duration,
                seed=self.cfg.seed,
            ),
        )

        multi_wrkrs = self.cfg.num_workers > 0
        self.train_loader = DataLoader(
            dataset=ds,
            sampler=sampler,
            collate_fn=TwoViewData.collate,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            multiprocessing_context=mp.get_context("fork") if multi_wrkrs else None,
            persistent_workers=multi_wrkrs,
            drop_last=True,
        )

        self.steps_per_epoch = len(self.train_loader)

        self.logger.info(f"Train sessions: {len(self.train_session_ids)}")
        self.logger.info(f"Train units: {len(self.train_unit_ids)}")
        self.logger.info(f"Train steps/epoch: {self.steps_per_epoch}")

        if hasattr(self.loss, "init_recording_classifier"):
            self.loss.init_recording_classifier(self.train_session_ids)  # type: ignore
            self.loss.to(self.device)

    def _setup_val_loader(self):
        if not self.cfg.val.loss:
            ds = Dataset(
                root=self.cfg.data.root,
                config=self.cfg.data.val_dataset,
                unit_id_prefix_fn=lambda _: "",
            )
            ds.transform = self.model.tokenize  # type: ignore
            collate_fn = ViewData.collate
        else:
            # Need two view dataset if we want to compute loss
            ds = TwoViewDataset(
                root=self.cfg.data.root,
                config=self.cfg.data.val_dataset,
                unit_id_prefix_fn=lambda _: "",
            )
            ds.transform = self.model.tokenize  # type: ignore
            ds.common_transform = Compose([TwoViewData.from_two_views, self.loss.tokenize])  # type: ignore
            ds.set_second_view_props(
                window_length=self.cfg.views.duration,
                max_time_diff=self.cfg.views.max_distance,
                seed=self.cfg.seed,
            )
            collate_fn = TwoViewData.collate

        sampler = DistributedSamplerWrapper(
            SequentialFixedWindowSampler(
                sampling_intervals=ds.get_sampling_intervals(),
                window_length=self.cfg.views.duration,
                step=None,
                drop_short=True,
            )
        )

        multi_wrkrs = self.cfg.num_workers > 0
        self.val_loader = DataLoader(
            dataset=ds,
            sampler=sampler,
            collate_fn=collate_fn,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            multiprocessing_context=mp.get_context("fork") if multi_wrkrs else None,
            persistent_workers=multi_wrkrs,
        )

        self.logger.info(f"Val sessions: {len(self.val_session_ids)}")
        self.logger.info(f"Val units: {len(self.val_unit_ids)}")
        self.logger.info(f"Val steps/epoch: {len(self.val_loader)}")

        if hasattr(self.loss, "init_recording_classifier"):
            raise NotImplementedError(
                "Val loss with Domain-Adversarial loss not implemented"
            )

    def _setup_embedding_collectors(self):
        self.train_emb_collector = EmbeddingCollector(
            self.train_unit_ids,
            dim=self.model.emb_dim,
            device=self.device,
        )
        self.val_emb_collector = EmbeddingCollector(
            self.val_unit_ids,
            dim=self.model.emb_dim,
            device=self.device,
        )

    @rank_zero_only
    def _setup_evaluator(self):
        if self.cfg.debug is not None:
            return

        self.evaluator = instantiate(
            self.cfg.data.evaluator,
            world_size=self.world_size,
        )

    def _run_evaluator(self):
        if self.cfg.debug is not None:
            return

        if self.cfg.val.first:
            return

        train_unit_embs, train_unit_ids = self.train_emb_collector.compute()
        val_unit_embs, val_unit_ids = self.val_emb_collector.compute()
        if self.rank == 0:
            metrics = self.evaluator(
                X_train=train_unit_embs,
                train_unit_ids=train_unit_ids,
                X_val=val_unit_embs,
                val_unit_ids=val_unit_ids,
            )
            self.logger.log_dict(metrics)
