from typing import Iterable
from einops import rearrange, repeat
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import xformers.ops as xops

from torch_brain.data import chain

from . import BaseNeuronEncoder
from ..utils import Precision
from ..dataset import ViewData, SpikeData
from ..nn.attention import RotaryCrossAttention, RotarySelfAttention
from ..nn.rotary_embedding import RotaryEmbedding


class NuclrV2gaSpikes(BaseNeuronEncoder):
    def __init__(
        self,
        ctx_duration: float,
        latent_step: float,
        precision: Precision,
        dim: int,
        cross_heads: int,
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
        self.latent_step = latent_step
        self.precision = precision
        self.emb_dim = self.dim = dim

        t_min, t_max = 1e-4, 8 * ctx_duration
        # self.time_emb = SinusoidalTimeEmb(dim=dim, t_min=t_min, t_max=t_max)

        self.rotary_emb = RotaryEmbedding(
            head_dim=dim_head,
            rotate_dim=int(dim_head * rot_ratio),
            t_min=t_min,
            t_max=t_max,
        )

        self.num_latents = num_latents = int(np.ceil(ctx_duration / latent_step))
        latent_times = (torch.arange(num_latents) + 0.5) * latent_step
        self.register_buffer("latent_times", latent_times)

        self.cross_attn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dim_head=dim_head,
            rotate_value=True,
            pre_norm_q=False,
            pre_norm_kv=False,
        )
        self.cross_ffn = FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True)

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
                        RotaryCrossAttention(
                            dim=dim,
                            heads=cross_heads,
                            dim_head=dim_head,
                            atn_dropout=atn_dropout,
                            rotate_value=True,
                            pre_norm_kv=False,
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
                        RotaryCrossAttention(
                            dim=dim,
                            heads=cross_heads,
                            dim_head=dim_head,
                            atn_dropout=atn_dropout,
                            rotate_value=True,
                            pre_norm_kv=False,
                        ),
                        FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                        RotarySelfAttention(
                            dim,
                            self_heads,
                            dim_head,
                            atn_dropout,
                            rotate_value=True,
                        ),
                        RotaryCrossAttention(
                            dim=dim,
                            heads=cross_heads,
                            dim_head=dim_head,
                            atn_dropout=atn_dropout,
                            rotate_value=True,
                            pre_norm_kv=False,
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
        spikes: Tensor,
        spike_seqlen: Tensor,
        unit_seqlen: Tensor,
    ) -> Tensor:

        device = spikes.device
        num_units = len(spike_seqlen)
        temp_latent_seqlen = [self.num_latents for _ in range(num_units)]
        spax_latent_seqlen = unit_seqlen.tolist() * self.num_latents

        cross_attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=temp_latent_seqlen,
            kv_seqlen=spike_seqlen.tolist(),
            device=device,
        )
        spax_attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=spax_latent_seqlen,
            device=device,
        )
        temp_attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=temp_latent_seqlen,
            device=device,
        )

        with torch.autocast(device_type="cuda", enabled=False):
            ctx_rot = self.rotary_emb(spikes)
            x_rot = self.rotary_emb(self.latent_times)

        ctx = spikes.new_zeros((len(spikes), self.dim))
        x = spikes.new_zeros((num_units * self.num_latents, self.dim))  # (U L) D
        x_rot = repeat(x_rot, "L H -> (U L) H", U=num_units)

        # Input cross attention
        x = x + self.cross_attn(x, ctx, x_rot, ctx_rot, cross_attn_bias)
        x = x + self.dp(self.cross_ffn(x))

        # Temporal attention layers
        for t_attn, tc_attn, t_ffn in self.t_blocks:
            x = x + self.dp(t_attn(x=x, rotary=x_rot, attn_bias=temp_attn_bias))
            x = x + self.dp(tc_attn(x, ctx, x_rot, ctx_rot, cross_attn_bias))
            x = x + self.dp(t_ffn(x))

        # Spatio-temporal attention layers
        for t_attn, tc_attn, t_ffn, s_attn, sc_attn, s_ffn in self.st_blocks:
            # Spatial
            x = rearrange(x, "(U L) D -> (L U) D", L=self.num_latents, U=num_units)
            x = x + self.dp(s_attn(x, attn_bias=spax_attn_bias))
            x = rearrange(x, "(L U) D -> (U L) D", U=num_units, L=self.num_latents)
            x = x + self.dp(sc_attn(x, ctx, x_rot, ctx_rot, cross_attn_bias))
            x = x + self.dp(s_ffn(x))

            # Temporal
            x = x + self.dp(t_attn(x=x, rotary=x_rot, attn_bias=temp_attn_bias))
            x = x + self.dp(tc_attn(x, ctx, x_rot, ctx_rot, cross_attn_bias))
            x = x + self.dp(t_ffn(x))

        x = rearrange(x, "(U L) D -> U L D", U=num_units, L=self.num_latents)
        y = x.mean(1)
        return y

    def tokenize(self, view: SpikeData) -> ViewData:
        spike_times = torch.tensor(view.spikes.timestamps, dtype=torch.float32)
        spike_units = torch.tensor(view.spikes.unit_index, dtype=torch.long)
        active_units, spike_seqlen = spike_units.unique(return_counts=True)
        assert torch.all(
            active_units[:-1] <= active_units[1:]
        ), "active_units must be sorted"

        # Sort spikes by unit indices
        sort_idx = torch.argsort(spike_units)
        spike_times = spike_times[sort_idx]

        enc_input = {
            "spikes": chain(spike_times),
            "spike_seqlen": chain(spike_seqlen),
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
