# Run like:
# cat preprocess/allen_vc_2019/session_ids.txt | parallel -j 16 python utils/allen_split_probes.py

from pathlib import Path
import copy
import h5py
import numpy as np
import argparse

from temporaldata import Data, IrregularTimeSeries, Interval
from brainsets.core import serialize_fn_map


def split(input_fname: Path, output_root: Path):
    data = Data.load(input_fname, lazy=False)

    old_brainset_id = data.brainset.id
    new_brainset_id = old_brainset_id + "_probes"

    probes = np.unique(data.units.probe_id)
    print(f"Opened {input_fname}, found {len(probes)} probes")

    for i, probe in enumerate(probes):
        new_data = copy.deepcopy(data)

        new_data.brainset.id = new_brainset_id
        new_data.session.id = f"{data.session.id}_p{i}"

        # Mask units
        unit_mask = new_data.units.probe_id == probe
        new_data.units = new_data.units.select_by_mask(unit_mask)

        # Prep unit index mapper
        old_unit_idx = np.arange(len(data.units))[unit_mask]
        unit_idx_remap = np.full(len(data.units), fill_value=-1, dtype=int)
        unit_idx_remap[old_unit_idx] = np.arange(len(old_unit_idx))

        # Spikes
        spike_mask = np.isin(new_data.spikes.unit_index, old_unit_idx)
        new_timestamps = data.spikes.timestamps[spike_mask]
        new_unit_index = unit_idx_remap[data.spikes.unit_index[spike_mask]]
        assert (new_unit_index >= 0).all()

        # Find places where probe was dropped
        step = 1.0
        invalid_domain = Interval(0.0, 0.0)
        nspikes_bin = np.bincount(np.floor(new_timestamps / step).astype(int))
        for i, nspikes in zip(np.arange(len(nspikes_bin)), nspikes_bin):
            if nspikes == 0:
                # add an extra +/- 1s for safety
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

        new_fname = output_root / f"{new_data.session.id}.h5"
        with h5py.File(new_fname, "w") as f:
            new_data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

        print(f"Written to {new_fname}: {valid_time=}, {len(new_data.units)=}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fname", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    if args.output_root == None:
        args.output_root = Path(f"{str(args.input_fname.parent)}_probes")
        print(f"Output root: {args.output_root}")

    args.output_root.mkdir(exist_ok=True, parents=True)
    split(args.input_fname, args.output_root)


if __name__ == "__main__":
    main()
