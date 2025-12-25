# /// brainset-pipeline
# python-version = "3.10"
# dependencies = ["allensdk==2.16.2"]
# ///

from argparse import ArgumentParser
import logging
from pathlib import Path
import datetime

import h5py
import numpy as np
import pandas as pd

from temporaldata import Data, IrregularTimeSeries, ArrayDict, Interval
from brainsets.pipeline import BrainsetPipeline
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

parser = ArgumentParser()
parser.add_argument("--reprocess", action="store_true")


class Pipeline(BrainsetPipeline):
    brainset_id = "allen_vc_2019_vis"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir: Path, args):
        cache = EcephysProjectCache.from_warehouse(manifest=raw_dir / "manifest.json")
        sessions = cache.get_session_table()
        # get_session_table() returns a pandas DataFrame, with index being session ids
        # but the index is integers, brainsets would like index to be strings
        sessions["session_id"] = sessions.index
        sessions.index = [str(x) for x in sessions.index]
        return sessions

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        cache = EcephysProjectCache.from_warehouse(
            manifest=self.raw_dir / "manifest.json"
        )
        raw_data = cache.get_session_data(manifest_item.session_id)  # pyright: ignore
        return raw_data, manifest_item

    def process(self, download_output):
        assert self.args is not None

        self.update_status("PROCESSING")
        raw_data, sess_md = download_output

        self.processed_dir.mkdir(exist_ok=True, parents=True)

        brainset = BrainsetDescription(
            id=self.brainset_id,
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
            id=str(sess_md.Index),
            recording_date=datetime.datetime.strptime(time, "%Y-%m-%d"),
            task=Task.DISCRETE_VISUAL_CODING,
        )

        store_path = self.processed_dir / f"{session.id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        sex_map = {"M": Sex.MALE, "F": Sex.FEMALE}
        subject = SubjectDescription(
            id=str(sess_md.specimen_id),
            species=Species.MUS_MUSCULUS,
            age=sess_md.age_in_days,
            sex=sex_map[sess_md.sex],
            genotype=sess_md.full_genotype,
        )

        self.update_status("Extracting Spikes")
        spikes, units = extract_spikes(raw_data)

        invalid_interval = extract_invalid_interval(raw_data)

        self.update_status("Extracting Stimulus Times")
        stimulus_epochs = extract_stimulus_epochs(raw_data)

        try:
            self.update_status("Extracting Optotagging Info")
            optotag_md = raw_data.optogenetic_stimulation_epochs
            optotag_start = sorted(optotag_md.start_time)[0]
            spikes = spikes.slice(0, optotag_start - 300.0)  # 5 minute gap
            logging.info(f"Optotagging start time {optotag_start}")
        except:
            logging.info(f"No optotagging info found")

        data = Data(
            brainset=brainset,
            session=session,
            subject=subject,
            spikes=spikes,
            units=units,
            stimulus_epochs=stimulus_epochs,
            domain=spikes.domain.difference(invalid_interval),
            invalid_interval=invalid_interval,
        )

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        logging.info(f"Saved session data to {store_path}")


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


def extract_invalid_interval(raw_data: EcephysSession):
    ans = Interval(0, 0)
    for _, x in raw_data.get_invalid_times().iterrows():
        ans |= Interval(x.start_time, x.stop_time)
    return ans


def extract_stimulus_epochs(raw_data: EcephysSession):
    epochs = raw_data.get_stimulus_epochs()
    ans = Interval(
        start=epochs.start_time.to_numpy(),
        end=epochs.stop_time.to_numpy(),
        type=epochs.stimulus_name.to_numpy().astype(str),
    )
    return ans
