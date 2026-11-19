from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.varlen import varlen_attn

from .rotary_embedding import apply_rotary_pos_emb, invert_rotatry_pos_emb


@dataclass
class BlockDiagonalMask:
    r"""Block-diagonal attention over sequences packed along the token dimension.

    The tokens of several sequences are concatenated without padding. A query
    token attends only to the key tokens of its own sequence. ``from_seqlens``
    takes the same arguments as ``xformers.ops.fmha.BlockDiagonalMask.from_seqlens``.
    """

    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int

    @classmethod
    def from_seqlens(
        cls,
        q_seqlen: Sequence[int],
        kv_seqlen: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
    ) -> "BlockDiagonalMask":
        if kv_seqlen is None:
            kv_seqlen = q_seqlen
        assert len(q_seqlen) == len(kv_seqlen)

        def cumulative(seqlen: Sequence[int]) -> torch.Tensor:
            cu = torch.zeros(len(seqlen) + 1, dtype=torch.int32)
            cu[1:] = torch.tensor(seqlen, dtype=torch.int32).cumsum(0)
            return cu.to(device)

        return cls(
            cu_seqlens_q=cumulative(q_seqlen),
            cu_seqlens_k=cumulative(kv_seqlen),
            max_seqlen_q=max(q_seqlen),
            max_seqlen_k=max(kv_seqlen),
        )


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[BlockDiagonalMask] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    r"""FlashAttention-2 from PyTorch on inputs of shape ``(b, n, h, d)``.

    Without ``attn_bias``, every token of a batch element attends to every other
    token of it. With ``attn_bias``, the batch size must be 1 and the tokens are
    split into sequences as described by the mask.
    """
    if attn_bias is None:
        # Force the flash backend so that both paths run the same kernel
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=dropout_p,
            )
        return out.transpose(1, 2)

    if dropout_p > 0:
        raise NotImplementedError(
            "Attention dropout is not supported with a block-diagonal mask, "
            "because torch.nn.attention.varlen.varlen_attn has no dropout."
        )
    assert q.size(0) == 1, "A block-diagonal mask needs a batch size of 1"
    out = varlen_attn(
        q[0],
        k[0],
        v[0],
        attn_bias.cu_seqlens_q,
        attn_bias.cu_seqlens_k,
        attn_bias.max_seqlen_q,
        attn_bias.max_seqlen_k,
    )
    return out[None]


class RotaryCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        ctx_dim: int = None,
        heads: int = 8,
        dim_head: Optional[int] = None,
        atn_dropout: float = 0.0,
        rotate_value: bool = False,
        to_kv_bias: bool = True,
        to_q_bias: bool = True,
        to_out_bias: bool = True,
        pre_norm_q: bool = True,
        pre_norm_kv: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.atn_dropout = atn_dropout
        self.rotate_value = rotate_value
        dim_head = dim_head or (dim // heads)

        # build networks
        ctx_dim = ctx_dim or dim
        self.norm = nn.LayerNorm(dim) if pre_norm_q else nn.Identity()
        self.norm_context = nn.LayerNorm(ctx_dim) if pre_norm_kv else nn.Identity()

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=to_q_bias)
        self.to_kv = nn.Linear(ctx_dim, inner_dim * 2, bias=to_kv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=to_out_bias)

    def forward(
        self,
        x_q: torch.Tensor,
        x_ctx: torch.Tensor,
        rotary_q: Optional[torch.Tensor] = None,
        rotary_ctx: Optional[torch.Tensor] = None,
        attn_bias: Optional[BlockDiagonalMask] = None,
    ):

        q = self.to_q(self.norm(x_q))
        k, v = self.to_kv(self.norm_context(x_ctx)).chunk(2, dim=-1)

        if q.ndim == 2:
            q = rearrange(q, "n (h d) -> 1 n h d", h=self.heads)
            k = rearrange(k, "n (h d) -> 1 n h d", h=self.heads)
            v = rearrange(v, "n (h d) -> 1 n h d", h=self.heads)
            batched_input = False
        elif q.ndim == 3:
            q = rearrange(q, "b n (h d) -> b n h d", h=self.heads)
            k = rearrange(k, "b n (h d) -> b n h d", h=self.heads)
            v = rearrange(v, "b n (h d) -> b n h d", h=self.heads)
            batched_input = True
        else:
            raise ValueError("Unknown input format")

        # apply rotary embeddings
        if rotary_q is not None:
            assert rotary_ctx is not None
            if not batched_input:
                assert rotary_q.ndim == 2
                assert rotary_ctx.ndim == 2
                rotary_q = rotary_q[None, :, :]
                rotary_ctx = rotary_ctx[None, :, :]

            q = apply_rotary_pos_emb(rotary_q, q)
            k = apply_rotary_pos_emb(rotary_ctx, k)
            if self.rotate_value:
                v = apply_rotary_pos_emb(rotary_ctx, v)

        out = flash_attention(
            q,
            k,
            v,
            attn_bias=attn_bias,
            dropout_p=self.atn_dropout if self.training else 0.0,
        )

        if rotary_ctx is not None and self.rotate_value:
            out = apply_rotary_pos_emb(invert_rotatry_pos_emb(rotary_q), out)

        # project back to output
        if batched_input:
            out = rearrange(out, "b n h d -> b n (h d)")
        else:
            out = rearrange(out, "b n h d -> (b n) (h d)", b=1)

        out = self.to_out(out)
        return out


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        atn_dropout: float = 0.0,
        rotate_value: bool = False,
        to_qkv_bias: bool = True,
        to_out_bias: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.atn_dropout = atn_dropout
        self.rotate_value = rotate_value
        dim_head = dim_head or (dim // heads)

        # build networks
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=to_qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=to_out_bias)

    def forward(
        self,
        x: torch.Tensor,
        rotary: Optional[torch.Tensor] = None,
        attn_bias: Optional[BlockDiagonalMask] = None,
    ):

        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=-1)

        if q.ndim == 2:
            q = rearrange(q, "n (h d) -> 1 n h d", h=self.heads)
            k = rearrange(k, "n (h d) -> 1 n h d", h=self.heads)
            v = rearrange(v, "n (h d) -> 1 n h d", h=self.heads)
            batched_input = False
        elif q.ndim == 3:
            q = rearrange(q, "b n (h d) -> b n h d", h=self.heads)
            k = rearrange(k, "b n (h d) -> b n h d", h=self.heads)
            v = rearrange(v, "b n (h d) -> b n h d", h=self.heads)
            batched_input = True
        else:
            raise ValueError("Unknown input format")

        # apply rotary embeddings
        if rotary is not None:
            if not batched_input:
                assert rotary.ndim == 2
                rotary = rotary[None, :, :]

            q = apply_rotary_pos_emb(rotary, q)
            k = apply_rotary_pos_emb(rotary, k)
            if self.rotate_value:
                v = apply_rotary_pos_emb(rotary, v)

        out = flash_attention(
            q,
            k,
            v,
            attn_bias=attn_bias,
            dropout_p=self.atn_dropout if self.training else 0.0,
        )

        if rotary is not None and self.rotate_value:
            out = apply_rotary_pos_emb(invert_rotatry_pos_emb(rotary), out)

        # project back to output
        if batched_input:
            out = rearrange(out, "b n h d -> b n (h d)")
        else:
            out = rearrange(out, "b n h d -> (b n) (h d)", b=1)

        out = self.to_out(out)
        return out
