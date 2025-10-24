# pyright: reportAttributeAccessIssue=false

from pathlib import Path
import copy
import h5py
import numpy as np
import sys

from temporaldata import Data, IrregularTimeSeries, Interval
from brainsets.core import serialize_fn_map

root = Path("../data/processed/steinmetz_2019")
new_root = Path("../data/processed/steinmetz_2019_probes")
new_brainset_id = "steinmetz_2019_probes"

new_root.mkdir(exist_ok=True, parents=True)

if len(sys.argv) < 2:
    print("Usage: python allen_split_probes.py <session_id>")

fname = root / sys.argv[1]
with h5py.File(fname, "r") as f:
    data = Data.from_hdf5(f, lazy=False)
probes = np.unique(data.units.probe_id)
print(f"Opened {fname}, found {len(probes)} probes")

for i, probe in enumerate(probes):
    new_data = copy.deepcopy(data)

    new_data.brainset.id = new_brainset_id
    new_data.session.id = f"{data.session.id}_p{i}"

    unit_mask = new_data.units.probe_id == probe
    new_data.units = new_data.units.select_by_mask(unit_mask)

    old_unit_idx = np.arange(len(data.units))[unit_mask]

    unit_idx_remap = np.zeros(len(data.units), dtype=int)
    unit_idx_remap[old_unit_idx] = np.arange(len(old_unit_idx))

    # Spikes
    spike_mask = np.isin(new_data.spikes.unit_index, old_unit_idx)
    new_timestamps = data.spikes.timestamps[spike_mask]
    new_unit_index = unit_idx_remap[data.spikes.unit_index[spike_mask]]

    # Find places where probe was dropped
    step = 1.0
    invalid_domain = Interval(0.0, 0.0)
    nspikes_bin = np.bincount(np.floor(new_timestamps / step).astype(int))
    for i, nspikes in zip(np.arange(len(nspikes_bin)), nspikes_bin):
        if nspikes == 0:
            invalid_domain |= Interval((i - 1) * step, (i + 2) * step)
    invalid_domain = invalid_domain.coalesce(1.0)

    valid_domain = Interval(new_timestamps[0], new_timestamps[-1])
    valid_domain = valid_domain.difference(invalid_domain)
    valid_time = (valid_domain.end - valid_domain.start).sum()
    assert valid_time > 0.0

    new_data.spikes = IrregularTimeSeries(
        timestamps=new_timestamps,
        unit_index=new_unit_index,
        domain=valid_domain,
    )
    setattr(new_data, "_domain", valid_domain)

    new_fname = new_root / f"{new_data.session.id}.h5"
    with h5py.File(new_fname, "w") as f:
        new_data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    print(f"Written to {new_fname}: {valid_time=}, {len(new_data.units)=}")
