from typing import Optional
import torch
from torch import nn
from torch import Tensor
from einops import einsum

from ..dataset import TwoViewData
from ..logger import Logger
from ..distributed import AllReduce
from . import Loss
from ..models import BaseNeuronEncoder


class VICReg(Loss):
    def __init__(
        self,
        inv_factor: float,
        var_factor: float,
        cov_factor: float,
        var_cutoff: float,
        model: BaseNeuronEncoder,
        eps: float = 1e-4,
        projector: bool = True,
    ):
        super().__init__()

        self.inv_factor = inv_factor
        self.var_factor = var_factor
        self.cov_factor = cov_factor
        self.var_cutoff = var_cutoff
        self.eps = eps

        dim_in = model.emb_dim

        if projector:
            self.projector = nn.Sequential(
                nn.Linear(dim_in, 4 * dim_in),
                nn.ReLU(),
                nn.Linear(4 * dim_in, 4 * dim_in),
                nn.ReLU(),
                nn.Linear(4 * dim_in, 4 * dim_in),
            )
        else:
            self.projector = nn.Identity()

    def __call__(
        self,
        x1: Tensor,
        x2: Tensor,
        batch: TwoViewData,
        prefix: str,
        logger: Logger | None = None,
        return_loss_dict: bool = False,
    ) -> Optional[Tensor]:
        assert batch.collated
        p1: Tensor = self.projector(x1)
        p2: Tensor = self.projector(x2)

        match = batch.get_match_idx_with_offset()

        inv_loss: Tensor = (p1[match[0]] - p2[match[1]]).square_().mean()
        var_loss: Tensor = self._var_loss_fn(p1) + self._var_loss_fn(p2)
        cov_loss: Tensor = self._cov_loss_fn(p1) + self._cov_loss_fn(p2)

        inv_loss = AllReduce.apply(inv_loss)  # type: ignore
        var_loss = AllReduce.apply(var_loss)  # type: ignore
        cov_loss = AllReduce.apply(cov_loss)  # type: ignore

        loss = self.inv_factor * inv_loss
        loss += self.var_factor * var_loss
        loss += self.cov_factor * cov_loss

        if return_loss_dict:
            raise NotImplementedError()
        elif logger is not None:
            logger.log(f"{prefix}/loss_inv", inv_loss.item(), pbar="LossInv")
            logger.log(f"{prefix}/loss_var", var_loss.item(), pbar="LossVar")
            logger.log(f"{prefix}/loss_cov", cov_loss.item(), pbar="LossCov")
            logger.log(f"{prefix}/loss", loss.item(), pbar="Loss")
            return loss

    def _var_loss_fn(self, z: Tensor) -> Tensor:
        var = torch.sqrt(z.var(0) + self.eps)
        loss = (self.var_cutoff - var).clip(min=0).mean(0)
        return loss

    def _cov_loss_fn(self, z: Tensor) -> Tensor:
        z_cntrd = z - z.mean(0, keepdim=True)
        cov = einsum(z_cntrd, z_cntrd, "n d1, n d2 -> d1 d2") / (len(z) - 1)
        loss = cov.fill_diagonal_(0).square_().sum() / z.size(1)
        return loss
