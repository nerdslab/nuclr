from typing import Optional

from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from torch_brain.data import Dataset
from torch_brain.data.sampler import SequentialFixedWindowSampler
from temporaldata import Interval

from src.models import BaseNeuronEncoder
from src.dataset import ViewData
from src.samplers import (
    DistributedSamplerWrapper,
)
from src.utils import Precision
from src.evaluators.collector import EmbeddingCollector
from src.trainers.trainer import Trainer, CheckpointDict
from src.utils import instantiate, expand_path


class ForwardPass(Trainer):
    """Does a forward pass over the train_dataset,
    and stores the embeddings in a .pt file located
    at cfg.embs_root / run_id / embs_epoch_{epoch_num}.pt
    """

    def setup(self, ckpt: Optional[CheckpointDict]):
        if ckpt is None:
            raise ValueError("Must provide ckpt")

        embs_dir = expand_path(self.cfg.embs_root) / ckpt["run_id"]
        embs_dir.mkdir(exist_ok=True, parents=True)
        self.embs_save_filepath: Path = (
            embs_dir / f"embs_{self.cfg.ckpt.load_from.name}"
        )
        self.logger.info(f"Will save embeddings at {self.embs_save_filepath}")
        if self.embs_save_filepath.exists():
            if self.cfg.overwrite:
                self.logger.warn(f"Embeddings exist, will overwrite")
            else:
                raise ValueError(
                    f"Embeddings exist at that location."
                    f" Use `overwrite=true` if you want to overwrite them."
                )

        self._setup_model(ckpt)
        self._setup_loader()
        self._setup_embedding_collector()

        self.model = self.make_ddp(self.model)  # type: ignore

    def _setup_model(self, ckpt):
        self.precision = Precision(self.cfg.precision)
        self.model: BaseNeuronEncoder = instantiate(
            ckpt["cfg"]["model"],
            precision=self.precision,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.eval().to(self.device)

    def train(self):
        self.val_epoch()
        unit_emb, unit_ids = self.emb_collector.compute()
        if self.rank == 0:
            save_data = {
                "embs": unit_emb,
                "ids": unit_ids,
            }
            torch.save(save_data, self.embs_save_filepath)
            self.logger.info(f"Saved to {self.embs_save_filepath}")

    @torch.inference_mode()
    def val_epoch(self):
        if self.cfg.debug == "train":
            return

        self.emb_collector.reset()

        for batch in self.logger.get_pbar(self.loader, "Forward Pass"):
            assert isinstance(batch, ViewData)
            batch.to(self.device)

            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                y1 = self.model(**batch.data["enc_input"])
                self.emb_collector.update(y1, batch.unit_ids)

    def _setup_loader(self):
        ds = Dataset(
            root=self.cfg.data.root,
            config=self.cfg.data.train_dataset,
            unit_id_prefix_fn=lambda _: "",
        )
        ds.transform = self.model.tokenize  # type: ignore
        collate_fn = ViewData.collate

        sampling_intervals = ds.get_sampling_intervals()

        if self.cfg.limit_observation_time is not None:
            self.logger.info(
                f"Limiting observation time to {self.cfg.limit_observation_time}s"
            )
            for sid, samp_interval in sampling_intervals.items():
                start = samp_interval.start[0]
                orig_duration = (samp_interval.end - samp_interval.start).sum()

                window = Interval(start, start + self.cfg.limit_observation_time)
                new_interval = samp_interval & window
                new_duration = (new_interval.end - new_interval.start).sum()
                sampling_intervals[sid] = new_interval

                self.logger.info(f"{sid}: {orig_duration} -> {new_duration}")

        sampler = DistributedSamplerWrapper(
            SequentialFixedWindowSampler(
                sampling_intervals=sampling_intervals,
                window_length=self.model.ctx_duration,
                step=None,
                drop_short=True,
            )
        )

        multi_wrkrs = self.cfg.num_workers > 0
        self.loader = DataLoader(
            dataset=ds,
            sampler=sampler,
            collate_fn=collate_fn,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            multiprocessing_context=mp.get_context("fork") if multi_wrkrs else None,
            persistent_workers=multi_wrkrs,
        )

        self.unit_ids = np.sort(ds.get_unit_ids())

        self.logger.info(f"Num sessions: {len(ds.get_session_ids())}")
        self.logger.info(f"Num units: {len(self.unit_ids)}")
        self.logger.info(f"Val steps/epoch: {len(self.loader)}")

    def _setup_embedding_collector(self):
        self.emb_collector = EmbeddingCollector(
            self.unit_ids,
            dim=self.model.emb_dim,
            device=self.device,
        )
