from torch_brain.data import Data


class NOP:
    def __call__(self, data: Data) -> Data:
        return data
