import torch
import torch.nn as nn

from ..logger import get_cli_logger


class SinusoidalTimeEmb(nn.Module):
    def __init__(
        self,
        dim: int,
        t_min: float,
        t_max: float,
    ):
        super().__init__()
        assert dim % 2 == 0

        periods = self.get_periods(dim // 2, t_min, t_max)
        omegas = 2 * torch.pi / periods
        self.register_buffer("periods", periods)
        self.register_buffer("omegas", omegas)

        # Log periods
        periods_str = [f"{p:.2e}" for p in periods]
        cli_logger = get_cli_logger()
        cli_logger.info("Sinusoid periods: [")
        per_line = 10
        for i in range(0, len(periods_str), per_line):
            cli_logger.info("  " + ", ".join(periods_str[i : i + per_line]) + ",")
        cli_logger.info("]")

    @torch.no_grad
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t[..., None] * self.omegas
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    @staticmethod
    def get_periods(num: int, t_min: float, t_max: float):
        exponents = torch.linspace(0, 1.0, num, dtype=torch.float32)
        t_min, t_max = torch.tensor(t_min), torch.tensor(t_max)
        periods = torch.exp(torch.lerp(t_min.log(), t_max.log(), exponents))
        assert torch.isclose(periods[0], t_min)
        assert torch.isclose(periods[-1], t_max)
        return periods
