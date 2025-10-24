from einops import rearrange
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops


class ConvAttention(nn.Module):
    def __init__(self, in_ch: int, n_heads: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.to_qkv = nn.Conv1d(in_ch, n_heads * 3, kernel_size, stride=1)
        self.to_out = nn.Conv1d(n_heads, in_ch, kernel_size, stride=1)

    def forward(self, X: Tensor, attn_bias: xops.AttentionBias) -> Tensor:

        q, k, v = self.to_qkv(X).chunk(3, dim=-2)
        reshape = lambda x: x[None, ...]
        q, k, v = reshape(q), reshape(k), reshape(v)

        extra_dim = 0
        if q.size(-1) % 8 != 0:
            extra_dim = 8 - (q.size(-1) % 8)
            pad = lambda x: F.pad(x, (0, extra_dim))
            q, k, v = pad(q), pad(k), pad(v)

        out = xops.memory_efficient_attention(
            query=q,
            key=k,
            value=v,
            attn_bias=attn_bias,
            p=0.0,
        )
        out = out[0, ..., :-extra_dim]
        out = F.relu(out)
        out = self.to_out(out)
        out = out + X[..., self.kernel_size - 1 : -self.kernel_size + 1]
        return out
