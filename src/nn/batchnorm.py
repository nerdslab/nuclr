import torch
from torch import nn, Tensor
from .functional import seqlen_to_batch_idx, seqlen_zscore


class SeqlenBatchNorm1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x: Tensor, seqlen: Tensor) -> Tensor:
        assert seqlen.device == x.device
        assert seqlen.ndim == 1
        assert seqlen.sum() == x.size(0)
        assert (seqlen > 0).all()
        # assert x.ndim == 2
        assert x.size(-1) == self.dim

        eps = 1e-5
        dim = self.dim
        batch_idx = seqlen_to_batch_idx(seqlen)

        def expand(inp):
            return inp.view(-1, *([1] * (x.ndim - 1)))

        mean = x.new_zeros((len(seqlen), *x.shape[1:]))
        mean.index_add_(dim=0, index=batch_idx, source=x)
        mean = mean / (expand(seqlen) + eps)
        x = x - mean.repeat_interleave(seqlen, dim=0)

        std = x.new_zeros((len(seqlen), *x.shape[1:]))
        std.index_add_(dim=0, index=batch_idx, source=torch.square(x))
        std = torch.sqrt(std / (expand(seqlen) + eps))
        x = x / (std.repeat_interleave(seqlen, dim=0) + eps)

        return (x * self.weight.unsqueeze(0)) + self.bias.unsqueeze(0)
