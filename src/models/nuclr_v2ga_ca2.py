# v2g + flipped ST layers

from typing import Iterable
import math
from einops import rearrange
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import xformers.ops as xops

from torch_brain.data import chain
from temporaldata import Data

from . import BaseNeuronEncoder
from ..utils import Precision
from ..dataset import ViewData
from ..nn.attention import RotarySelfAttention
from ..nn.rotary_embedding import RotaryEmbedding
from ..utils import get_cli_logger


logger = get_cli_logger()


class NuclrV2gaCa2(BaseNeuronEncoder):
    def __init__(
        self,
        ctx_duration: float,
        patch_size: int,
        precision: Precision,
        dim: int,
        self_heads: int,
        dim_head: int,
        atn_dropout: float,
        lin_dropout: float,
        t_layers: int,
        st_layers: int,
        rot_ratio: float = 0.5,
    ):
        super().__init__()

        self.ctx_duration = ctx_duration
        self.patch_size = patch_size
        self.precision = precision
        self.emb_dim = self.dim = dim

        samp_rate = 1.0 / 0.23344201114422702
        num_samples = math.floor(self.ctx_duration * samp_rate)
        self.num_latents = math.floor(num_samples / self.patch_size)

        logger.info(f"Num latents = {self.num_latents}, Patch size = {patch_size}")

        t_min, t_max = 1.0, 8.0 * self.num_latents
        self.rotary_emb = RotaryEmbedding(
            head_dim=dim_head,
            rotate_dim=int(dim_head * rot_ratio),
            t_min=t_min,
            t_max=t_max,
        )

        self.read_in = nn.Linear(patch_size, dim)

        self.t_blocks: Iterable[Iterable[nn.Module]] = nn.ModuleList(  # type: ignore
            [
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim,
                            self_heads,
                            dim_head,
                            atn_dropout,
                            rotate_value=True,
                        ),
                        FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                    ]
                )
                for _ in range(t_layers)
            ]
        )

        self.st_blocks: Iterable[Iterable[nn.Module]] = nn.ModuleList(  # type: ignore
            [
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim,
                            self_heads,
                            dim_head,
                            atn_dropout,
                            rotate_value=True,
                        ),
                        FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                        RotarySelfAttention(
                            dim,
                            self_heads,
                            dim_head,
                            atn_dropout,
                            rotate_value=True,
                        ),
                        FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                    ]
                )
                for _ in range(st_layers)
            ]
        )

        self.dp = nn.Dropout(lin_dropout)

    def forward(
        self,
        bins: Tensor,
        unit_seqlen: Tensor,
    ) -> Tensor:

        device = bins.device
        num_units = int(unit_seqlen.sum())

        spax_attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=unit_seqlen.tolist() * self.num_latents,
            device=device,
        )

        x = self.read_in(bins.float())
        x = rearrange(x, "(U L) D -> U L D", U=num_units, L=self.num_latents)
        t = torch.arange(self.num_latents, device=device).repeat(num_units, 1).float()
        x_rot = self.rotary_emb(t)

        # Temporal attention layers
        for t_attn, t_ffn in self.t_blocks:
            x = x + self.dp(t_attn(x=x, rotary=x_rot))
            x = x + self.dp(t_ffn(x))

        # Spatio-temporal attention layers
        for t_attn, t_ffn, s_attn, s_ffn in self.st_blocks:
            # Spatial
            x = rearrange(x, "U L D -> (L U) D", L=self.num_latents, U=num_units)
            x = x + self.dp(s_attn(x, attn_bias=spax_attn_bias))
            x = x + self.dp(s_ffn(x))
            x = rearrange(x, "(L U) D -> U L D", U=num_units, L=self.num_latents)

            # Temporal
            x = x + self.dp(t_attn(x=x, rotary=x_rot))
            x = x + self.dp(t_ffn(x))

        y = x.mean(1)
        return y

    def tokenize(self, view: Data) -> ViewData:
        unit_ids: np.ndarray = view.units.id  # pyright: ignore
        traces = view.calcium_traces  # type: ignore
        num_units = len(unit_ids)

        values = torch.from_numpy(traces.df_over_f_norm).to(torch.float32).T
        assert values.size(0) == num_units
        values = values[:, : self.num_latents * self.patch_size]
        values = rearrange(
            values, "U (L P) -> (U L) P", L=self.num_latents, P=self.patch_size
        )

        enc_input = {
            "bins": chain(values),
            "unit_seqlen": num_units,
        }

        recording_ids = np.array([view.session.id for _ in range(num_units)])  # type: ignore
        data = {"enc_input": enc_input}
        return ViewData(data=data, unit_ids=unit_ids, recording_ids=recording_ids)


class FFN(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float, pre_norm: bool):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.in_proj = nn.Linear(dim, 2 * dim * mult)
        self.out_proj = nn.Linear(dim * mult, dim)
        self.dp = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.norm(x)
        y, gate = self.in_proj(y).chunk(2, dim=-1)
        y = self.act(gate) * y
        y = self.dp(y)
        return self.out_proj(y)
