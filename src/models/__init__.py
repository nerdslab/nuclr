import abc
import torch

from ..dataset import ViewData, SpikeData


class BaseNeuronEncoder(abc.ABC, torch.nn.Module):
    ctx_duration: float
    emb_dim: int

    @abc.abstractmethod
    def tokenize(self, view: SpikeData) -> ViewData:
        raise NotImplementedError
