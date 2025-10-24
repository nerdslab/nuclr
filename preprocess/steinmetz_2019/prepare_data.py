# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false

from pathlib import Path
import argparse
import datetime
import numpy as np
import pandas as pd
import h5py

from temporaldata import Data, IrregularTimeSeries, ArrayDict
from brainsets.taxonomy import Species
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.core import serialize_fn_map


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path)
    parser.add_argument("--processed_dir", type=Path)
    parser.add_argument("--session_dir", type=str)

    args = parser.parse_args()
    datadir = args.raw_dir / args.session_dir
    print(f"Processing {args.session_dir}")

    # Brainset
    brainset = BrainsetDescription(
        id="steinmetz_2019",
        origin_version="0.0.0",
        derived_version="0.0.0",
        source="https://figshare.com/articles/dataset/Dataset_from_Steinmetz_et_al_2019/9598406",
        description="Distributed coding of choice, action and engagement across the mouse brain",
    )

    # Session
    subject_id, date = args.session_dir.split("_")
    subject_id = subject_id.lower()
    session_id = f"{subject_id}_{''.join(date.split('-'))}"
    session = SessionDescription(
        id=session_id,
        recording_date=datetime.datetime.strptime(date, "%Y-%m-%d"),
    )

    # Subject
    subject = SubjectDescription(
        id=subject_id,
        species=Species.UNKNOWN,
    )

    # Spikes
    spikes = IrregularTimeSeries(
        timestamps=np.load(datadir / "spikes.times.npy").flatten().astype(np.float64),
        unit_index=np.load(datadir / "spikes.clusters.npy").flatten().astype(int),
        domain="auto",
    )
    spikes.sort()
    spikes = spikes.slice(0, spikes.timestamps.max())
    unique_unit_indices = np.unique(spikes.unit_index)  # type: ignore
    assert unique_unit_indices[0] == 0
    assert unique_unit_indices[-1] == len(unique_unit_indices) - 1
    num_units = len(unique_unit_indices)

    channels_location_df = pd.read_csv(datadir / "channels.brainLocation.tsv", sep="\t")

    # Units
    units = ArrayDict(
        id=np.array([f"{session_id}_u{x}" for x in range(num_units)]),
        probe_id=np.load(datadir / "clusters.probes.npy").flatten().astype(int),
        channel=np.load(datadir / "clusters.peakChannel.npy").flatten().astype(int) - 1,
        depths=np.load(datadir / "clusters.depths.npy").flatten().astype(int),
    )
    units.brain_area = channels_location_df.loc[units.channel].allen_ontology.values
    units.ccf_ap = channels_location_df.loc[units.channel].ccf_ap.values
    units.ccf_dv = channels_location_df.loc[units.channel].ccf_dv.values
    units.ccf_lr = channels_location_df.loc[units.channel].ccf_lr.values

    data = Data(
        brainset=brainset,
        subject=subject,
        session=session,
        spikes=spikes,
        units=units,
        domain=spikes.domain,
    )

    args.processed_dir.mkdir(exist_ok=True, parents=True)
    output_fname = args.processed_dir / f"{session_id}.h5"
    with h5py.File(output_fname, "w") as f:
        data.to_hdf5(f, serialize_fn_map=serialize_fn_map)
    print(f"Written to {output_fname}")


if __name__ == "__main__":
    main()
