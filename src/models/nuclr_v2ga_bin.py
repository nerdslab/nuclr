# v2g + flipped ST layers

from typing import Iterable
from einops import rearrange
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import xformers.ops as xops

from torch_brain.data import chain

from . import BaseNeuronEncoder
from ..utils import Precision, is_divisible
from ..dataset import ViewData, SpikeData
from ..nn.attention import RotarySelfAttention
from ..nn.rotary_embedding import RotaryEmbedding


class NuclrV2gaBin(BaseNeuronEncoder):
    def __init__(
        self,
        ctx_duration: float,
        latent_step: float,
        bin_size: float,
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

        assert is_divisible(latent_step, bin_size)
        assert is_divisible(ctx_duration, latent_step)
        self.ctx_duration = ctx_duration
        self.latent_step = latent_step
        self.bin_size = bin_size
        self.precision = precision
        self.emb_dim = self.dim = dim

        # derived params
        self.bins_per_latent = int(latent_step / bin_size)  # num of bins per latent
        self.num_latents = int(np.ceil(ctx_duration / latent_step))

        t_min, t_max = 1.0, 8.0 * self.num_latents
        self.rotary_emb = RotaryEmbedding(
            head_dim=dim_head,
            rotate_dim=int(dim_head * rot_ratio),
            t_min=t_min,
            t_max=t_max,
        )

        self.read_in = nn.Linear(self.bins_per_latent, dim)

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

    def tokenize(self, view: SpikeData) -> ViewData:
        spike_times = torch.tensor(view.spikes.timestamps, dtype=torch.float32)
        spike_units = torch.tensor(view.spikes.unit_index, dtype=torch.long)
        active_units = spike_units.unique()
        assert torch.all(
            active_units[:-1] <= active_units[1:]
        ), "active_units must be sorted"

        # bin spikes
        num_bins = int(self.ctx_duration / self.bin_size)
        rate = 1 / self.bin_size
        bins = torch.zeros((len(view.units), num_bins + 1), dtype=torch.int16)
        bins.index_put_(
            indices=(spike_units, torch.floor(spike_times * rate).long()),
            values=torch.ones(len(spike_times), dtype=torch.int16),
            accumulate=True,
        )
        bins = bins[active_units][:, : self.num_latents * self.bins_per_latent]
        bins = rearrange(
            bins, "U (L B) -> (U L) B", L=self.num_latents, B=self.bins_per_latent
        )

        enc_input = {
            "bins": chain(bins),
            "unit_seqlen": len(active_units),
        }

        recording_ids = np.array([view.session.id for _ in range(len(active_units))])  # type: ignore
        unit_ids = view.units.id[active_units]  # type: ignore

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
