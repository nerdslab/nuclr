from typing import Tuple
import numpy as np
import torch
import torch.distributed as dist
from ..logger import get_cli_logger

logger = get_cli_logger()


class EmbeddingCollector:
    def __init__(self, unit_ids: np.ndarray, dim: int, device: torch.device):
        self.device = device
        self.unit_ids = np.sort(unit_ids)
        self.dim = dim
        self.reset()

    def update(self, x: torch.Tensor, unit_ids: np.ndarray) -> None:
        with torch.inference_mode():
            idx = np.searchsorted(self.unit_ids, unit_ids)
            idx = torch.tensor(idx, device=self.device, dtype=torch.long)

            self.emb_sum.index_add_(0, idx, x.detach().to(torch.float64))
            self.count.index_add_(0, idx, torch.ones_like(idx))

    def compute(self) -> Tuple[torch.Tensor, np.ndarray]:
        with torch.inference_mode():
            if dist.is_initialized():
                dist.all_reduce(self.emb_sum)
                dist.all_reduce(self.count)

            mask = self.count > 0
            if not mask.all():
                logger.warning("Some units have no embedding")

            unit_emb = self.emb_sum[mask].clone() / self.count[mask, None]
            unit_ids = self.unit_ids[mask.cpu().numpy()]

            self.reset()
            return unit_emb, unit_ids

    def reset(self):
        self.emb_sum = torch.zeros(
            (len(self.unit_ids), self.dim),
            requires_grad=False,
            device=self.device,
            dtype=torch.float64,
        )
        self.count = torch.zeros(
            len(self.unit_ids),
            requires_grad=False,
            device=self.device,
            dtype=torch.long,
        )
