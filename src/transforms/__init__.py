from temporaldata import Data


class NOP:
    def __call__(self, data: Data) -> Data:
        return data
