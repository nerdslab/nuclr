import torch
import torch.nn as nn
from torch import Tensor
from einops import repeat, rearrange
import numpy as np

from src.utils import get_cli_logger
from src.nn.sine_emb import SinusoidalTimeEmb


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, rotate_dim, t_min=1e-4, t_max=4.0):
        super().__init__()

        assert head_dim % 2 == 0
        assert rotate_dim % 2 == 0

        periods = SinusoidalTimeEmb.get_periods(rotate_dim // 2, t_min, t_max)
        omega = torch.zeros(head_dim // 2)
        omega[: rotate_dim // 2] = 2 * torch.pi / periods
        self.register_buffer("omega", omega)

        # Log periods
        periods_str = [f"{p:.2e}" for p in periods]
        cli_logger = get_cli_logger()
        cli_logger.info("Rotary periods: [")
        per_line = 10
        for i in range(0, len(periods_str), per_line):
            cli_logger.info("  " + ", ".join(periods_str[i : i + per_line]) + ",")
        cli_logger.info("]")

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, timestamps: Tensor) -> Tensor:
        r"""Computes the rotation matrices for given timestamps.

        Args:
            timestamps (torch.Tensor): timestamps tensor.
        """
        angles = timestamps.unsqueeze(-1) * self.omega
        angles = repeat(angles, "... n -> ... (n r)", r=2)
        pos_emb = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return pos_emb


def rotate_half(x: Tensor) -> Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(pos_emb: Tensor, x: Tensor, head_dim: int = 2) -> Tensor:
    r"""Apply the rotary positional embedding to the input data.

    Args:
        pos_emb (torch.Tensor): Angles for different rotations.
        x (torch.Tensor): Input data.
        head_dim (int, optional): Dimension of the head. Defaults to 2.
    """
    pos_emb = pos_emb.unsqueeze(head_dim).to(x.dtype)
    pos_cos, pos_sin = pos_emb.chunk(chunks=2, dim=-1)
    return (x * pos_cos) + (rotate_half(x) * pos_sin)


def invert_rotatry_pos_emb(pos_emb: Tensor) -> Tensor:
    pos_cos, pos_sin = pos_emb.chunk(chunks=2, dim=-1)
    return torch.cat((pos_cos, -pos_sin), dim=-1)
