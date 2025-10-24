from argparse import ArgumentParser
import logging
from pathlib import Path
import datetime

import h5py
import numpy as np
import pandas as pd

from temporaldata import Data, IrregularTimeSeries, ArrayDict, Interval
from brainsets import serialize_fn_map
from brainsets.taxonomy import RecordingTech, Species, Sex, Task
from brainsets.descriptions import (
    BrainsetDescription,
    SubjectDescription,
    SessionDescription,
)
from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
    EcephysProjectCache,
    EcephysSession,
)

logging.basicConfig(level=logging.INFO)


def extract_spikes(session):
    units = session.units
    spiketimes_dict = session.spike_times

    spikes = []
    unit_index = []
    types = []
    # waveforms = []
    unit_meta = []

    unit_number = 0
    for unit_id in spiketimes_dict.keys():
        metadata = units.loc[unit_id]
        if not metadata["structure_acronym"].startswith("VIS"):
            continue

        spiketimes = spiketimes_dict[unit_id]
        spikes.append(spiketimes)
        unit_index.append([unit_number] * len(spiketimes))
        types.append(np.ones_like(spiketimes) * int(RecordingTech.NEUROPIXELS_SPIKES))

        peak_channel = session.units.loc[unit_id].peak_channel_id
        waveform = session.mean_waveforms[unit_id].sel(channel_id=peak_channel)
        waveform = np.array(waveform, dtype=np.float64)

        unit_meta.append(
            {
                "count": len(spiketimes),
                "probe_id": metadata["probe_id"],
                "electrode_row": metadata["probe_horizontal_position"],
                "electrode_col": metadata["probe_vertical_position"],
                "id": str(unit_id),
                "area_name": metadata["structure_acronym"],
                "channel_number": metadata["probe_channel_number"],
                "unit_number": unit_number,
                "type": int(RecordingTech.NEUROPIXELS_SPIKES),
                "peak_channel": peak_channel,
                "mean_waveform_on_peak_channel": waveform,
            }
        )

        unit_number += 1

    spikes = np.concatenate(spikes)
    # waveforms = np.concatenate(waveforms)
    unit_index = np.concatenate(unit_index)
    types = np.concatenate(types)

    # convert unit metadata to a Data object
    unit_meta_df = pd.DataFrame(unit_meta)  # list of dicts to dataframe
    units = ArrayDict.from_dataframe(unit_meta_df, unsigned_to_long=True)

    sorted = np.argsort(spikes)
    spikes = spikes[sorted]
    # waveforms = waveforms[sorted]
    unit_index = unit_index[sorted]
    types = types[sorted]

    spikes = IrregularTimeSeries(
        domain="auto",
        timestamps=spikes,
        # waveforms=waveforms,
        unit_index=unit_index,
        types=types,
    )

    return spikes, units


def extract_invalid_interval(session: EcephysSession):
    ret = Interval(0, 0)
    for _, x in session.get_invalid_times().iterrows():
        curr_interval = Interval(x.start_time, x.stop_time)
        ret |= curr_interval
    return ret


def process_session(raw_data: EcephysSession, sess_md: pd.DataFrame) -> Data:

    brainset = BrainsetDescription(
        id="allen_vc_2019_vis",
        origin_version="0.3.0",
        derived_version="nuclr_v0.1",
        source="allensdk",
        description="Allen Visual Coding Neuropixels 2019",
        session_type=sess_md.session_type,
        unit_count=sess_md.unit_count,
        channel_count=sess_md.channel_count,
        probe_count=sess_md.probe_count,
        published_at=sess_md.published_at,
    )

    time = sess_md.published_at.split("T")[0]
    session = SessionDescription(
        id=str(sess_md.name),
        recording_date=datetime.datetime.strptime(time, "%Y-%m-%d"),
        task=Task.DISCRETE_VISUAL_CODING,
    )

    animal = f"mouse_{sess_md['specimen_id']}"
    sex_map = {"M": Sex.MALE, "F": Sex.FEMALE}
    subject = SubjectDescription(
        id=str(sess_md["specimen_id"]),
        species=Species.MUS_MUSCULUS,
        age=sess_md["age_in_days"],
        sex=sex_map[sess_md["sex"]],
        genotype=sess_md["full_genotype"],
    )

    spikes, units = extract_spikes(raw_data)

    invalid_interval = extract_invalid_interval(raw_data)

    try:
        optotag_start = sorted(raw_data.optogenetic_stimulation_epochs.start_time)[0]
        spikes = spikes.slice(0, optotag_start - 300.0)  # 5 minute gap
    except:
        pass

    data = Data(
        brainset=brainset,
        session=session,
        subject=subject,
        spikes=spikes,
        units=units,
        domain=spikes.domain.difference(invalid_interval),
        invalid_interval=invalid_interval,
    )

    return data


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=Path,
        required=True,
        help="where the raw data is stored/will be stored",
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        required=True,
        help="where processed data will be stored",
    )
    parser.add_argument("--session_id", type=int, required=True)
    args = parser.parse_args()

    print(f"Preprocessing {args.session_id}")

    # get the project cache from the warehouse
    manifest_path: Path = args.raw_dir / "manifest.json"
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()
    sess_id = args.session_id
    sess_md = sessions.loc[sess_id]

    raw_data = cache.get_session_data(sess_id)
    data = process_session(raw_data, sess_md)

    output_dir: Path = args.processed_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"{sess_id}.h5"
    with h5py.File(output_path, "w") as f:
        data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    logging.info(f"Saved session data to {output_path}")


if __name__ == "__main__":
    main()
