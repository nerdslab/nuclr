from __future__ import annotations
from typing import Union, Dict, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from ..dataset import TwoViewData
from ..logger import Logger
from ..distributed import AllReduce
from . import Loss
from ..models import BaseNeuronEncoder


class SampleWiseContrastiveLoss(Loss):
    def __init__(
        self,
        tau: float,
        dcl: bool,
        model: BaseNeuronEncoder,
        projector: bool = True,
        full_denom: bool = False,
    ):
        super().__init__()
        self.tau = tau
        self.dcl = dcl
        self.full_denom = full_denom

        dim_in = model.emb_dim

        if projector:
            self.projector = nn.Sequential(
                nn.LayerNorm(dim_in),
                nn.Linear(dim_in, dim_in),
                nn.ReLU(),
                nn.Linear(dim_in, dim_in),
            )
        else:
            self.projector = nn.Identity()

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        batch: TwoViewData,
        prefix: str,
        logger: Logger | None = None,
        return_loss_dict: bool = False,
    ) -> Union[Tensor, Dict]:
        assert batch.collated

        p1 = F.normalize(self.projector(x1), dim=-1)
        p2 = F.normalize(self.projector(x2), dim=-1)

        B = batch.batch_size
        # losses = torch.zeros(B, device=x1.device)
        loss_numers, loss_denoms = x1.new_zeros(B), x2.new_zeros(B)
        running_idx1, running_idx2 = 0, 0
        for b in range(B):
            n1 = batch.view1.unit_seqlen[b]
            n2 = batch.view2.unit_seqlen[b]

            if batch.match_idx[b].size(1) > 0:  # only do if matches exist
                loss_numers[b], loss_denoms[b] = self.std_contrastive_loss(
                    x1=p1[running_idx1 : running_idx1 + n1],
                    x2=p2[running_idx2 : running_idx2 + n2],
                    match_idx=batch.match_idx[b],
                )

            running_idx1 += n1
            running_idx2 += n2

        # Compute loss weight as num_pos_pairs_in_view / total_pos_pairs_in_all_views
        num_matches = torch.tensor(
            data=[m.size(1) for m in batch.match_idx],
            dtype=torch.long,
            device=batch.device,
        )
        total_matches: Tensor = AllReduce.apply(
            num_matches.sum()
        )  # avg across GPUs # type: ignore
        loss_weights = num_matches / total_matches

        # Compute weighted-averaged loss
        loss_numer = self._weighted_avg(loss_numers, loss_weights)
        loss_denom = self._weighted_avg(loss_denoms, loss_weights)
        loss = loss_numer + loss_denom

        if return_loss_dict:
            loss_dict = {
                f"{prefix}/loss_numer": loss_numer.item(),
                f"{prefix}/loss_denom": loss_denom.item(),
                f"{prefix}/loss": loss.item(),
            }
            return loss_dict

        if logger is not None:
            logger.log(f"{prefix}/loss_numer", loss_numer.item(), pbar="Loss_N")
            logger.log(f"{prefix}/loss_denom", loss_denom.item(), pbar="Loss_D")
            logger.log(f"{prefix}/loss", loss.item(), pbar="Loss")
            # logger.log(f"{prefix}/sim", self.tau * loss_numer.item(), pbar="Sim")

        return loss

    def _weighted_avg(self, x: Tensor, weights: Tensor) -> Tensor:
        ans = (x * weights).sum()
        ans = AllReduce.apply(ans)
        return ans  # type: ignore

    def std_contrastive_loss(
        self,
        x1: Tensor,
        x2: Tensor,
        match_idx: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        # -- numerator
        sim12 = (x1 @ x2.T) / self.tau
        numer = sim12[match_idx[0], match_idx[1]].mean()
        numer = (1 / self.tau) - numer

        # -- denomenator
        if self.dcl:
            sim12[match_idx[0], match_idx[1]] = float("-inf")

        if self.full_denom:
            sim11 = (x1 @ x1.T).fill_diagonal_(float("-inf"))
            sim1 = torch.cat((sim12, sim11), dim=1)[match_idx[0]]

            sim22 = (x2 @ x2.T).fill_diagonal_(float("-inf"))
            sim2 = torch.cat((sim12.T, sim22), dim=1)[match_idx[1]]
        else:
            sim1 = sim12.clone()[match_idx[0]]
            sim2 = sim12.clone().T[match_idx[1]]

        denom1 = torch.mean(sim1.logsumexp(1) - math.log(sim1.size(1)))
        denom2 = torch.mean(sim2.logsumexp(1) - math.log(sim2.size(1)))

        denom = (denom1 + denom2) / 2

        # denom = self.tau * (denom1 + denom2) / 2.0

        return numer, denom
