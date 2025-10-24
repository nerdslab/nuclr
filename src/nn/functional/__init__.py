import torch
from torch import Tensor


@torch.no_grad()
def seqlen_to_batch_idx(seqlen: Tensor) -> Tensor:
    """Convert seqlen to an index tensor
    Example: seqlen = [2, 3] -> index = [0, 0, 1, 1, 1]
    """
    indices = torch.arange(len(seqlen), device=seqlen.device)
    return indices.repeat_interleave(seqlen, dim=0)


def seqlen_mean(x: Tensor, seqlen: Tensor) -> Tensor:
    """
    Args:
        x: (N, D)
        seqlen: (B,): [seqlen_1, seqlen_2, ...]
    """

    assert seqlen.device == x.device
    assert seqlen.ndim == 1
    assert seqlen.sum() == x.size(0)
    # assert (seqlen > 0).all()
    assert x.ndim == 2

    eps = 1e-5
    dim = x.size(1)
    batch_idx = seqlen_to_batch_idx(seqlen)

    ans = x.new_zeros((len(seqlen), dim))
    ans.index_add_(dim=0, index=batch_idx, source=x)
    ans = ans / (seqlen.unsqueeze(1) + eps)
    # ans = ans / seqlen.unsqueeze(1)
    return ans


def seqlen_zscore(x: Tensor, seqlen: Tensor) -> Tensor:
    assert seqlen.device == x.device
    assert seqlen.ndim == 1
    assert seqlen.sum() == x.size(0)
    assert (seqlen > 0).all()
    assert x.ndim == 2

    eps = 1e-5
    dim = x.size(1)
    batch_idx = seqlen_to_batch_idx(seqlen)

    mean = x.new_zeros((len(seqlen), dim))
    mean.index_add_(dim=0, index=batch_idx, source=x)
    mean = mean / seqlen.unsqueeze(1)
    x = x - mean.repeat_interleave(seqlen, dim=0)

    std = x.new_zeros((len(seqlen), dim))
    std.index_add_(dim=0, index=batch_idx, source=torch.square(x))
    std = torch.sqrt(std / seqlen.unsqueeze(1))
    x = x / (std.repeat_interleave(seqlen, dim=0) + eps)

    return x
