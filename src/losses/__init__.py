from abc import ABC
from torch import nn

from ..dataset import TwoViewData


class Loss(nn.Module, ABC):

    def tokenize(self, views: TwoViewData) -> TwoViewData:
        return views
