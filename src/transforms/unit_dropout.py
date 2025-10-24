from typing import cast
import numpy as np
from temporaldata import Data

from ..dataset import SpikeData, CalciumData
from ..utils import apply_unit_mask_


class UnitDropout:
    def __init__(self, min_units: int | float):
        if isinstance(min_units, float):
            assert min_units <= 1.0
        self.min_units = min_units

    def __call__(self, data: SpikeData | CalciumData) -> Data:
        num_units_before = len(data.units)

        if isinstance(self.min_units, int):
            min_units = self.min_units
        elif isinstance(self.min_units, float):
            min_units = int(num_units_before * self.min_units)
        else:
            raise ValueError(
                f"Unknown type of min_units: "
                f"{type(self.min_units)}, {self.min_units=}"
            )

        if num_units_before <= min_units:
            return data

        num_units_after = np.random.randint(min_units, num_units_before + 1)
        sampled_unit_idx = np.random.permutation(num_units_before)[:num_units_after]
        sampled_unit_mask = np.zeros(num_units_before, dtype=bool)
        sampled_unit_mask[sampled_unit_idx] = True

        if hasattr(data, "spikes"):
            return apply_unit_mask_(data, sampled_unit_mask)
        elif hasattr(data, "calcium_traces"):
            data = cast(CalciumData, data)
            df_over_f = data.calcium_traces.df_over_f
            assert df_over_f.shape[1] == num_units_before
            data.calcium_traces.df_over_f = df_over_f[:, sampled_unit_mask]
            return data
        else:
            raise NotImplementedError()
