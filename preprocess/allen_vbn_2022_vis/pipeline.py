# /// brainset-pipeline
# python-version = "3.10"
# dependencies = ["allensdk==2.16.2"]
# ///

from brainsets.pipeline import BrainsetPipeline
from brainsets import serialize_fn_map
from brainsets.descriptions import BrainsetDescription, SubjectDescription, SessionDescription
from brainsets.taxonomy import Species, Sex
from temporaldata import Data, IrregularTimeSeries, ArrayDict, Interval

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import logging
import datetime
from typing import Literal, get_args

from session_extractor import extract_session_data

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache

# Setup argument extensions
parser = ArgumentParser()

parser.add_argument("--reprocess", action="store_true")

# Unit filtering configuration 
# Can override like: --isi-violations-max 0.3 --presence-ratio-min 0.95
DEFAULT_FILTERS = [
    "isi_violations",
    "amplitude_cutoff", 
    "presence_ratio", 
    "firing_rate", 
    "quality", 
    "snr"
    ]
FILTER_SPECS = {
    "isi_violations": {"arg": "isi_violations_max", "default": 0.5, "type": float},
    "amplitude_cutoff": {"arg": "amplitude_cutoff_max", "default": 0.1, "type": float},
    "presence_ratio": {"arg": "presence_ratio_min", "default": 0.9, "type": float},
    "firing_rate" : {"arg" : "firing_rate", "default": 0.1, "type": float},
    "quality" : {"arg" : "quality" , "default" : "good", "type": str},
    "snr" : {"arg" : "snr" , "default" : 1.0, "type": float}
}
for name, spec in FILTER_SPECS.items():
    cli_flag = "--" + spec["arg"].replace("_", "-")
    parser.add_argument(cli_flag, type=spec["type"], default=spec["default"])

DEFAULT_TAGGING_PARAMS = [
    "time_before",
    "duration",
    "bin_size",
    "increase_in_fr",
    "min_evoked_rate",
    "baseline_start_ms",
    "baseline_end_ms",
    "evoked_start_ms",
    "evoked_end_ms",
    "max_pulse_duration",
    "pulse_level",
]
TAGGING_SPECS = {
    "time_before": {"arg": "tagging_time_before", "default": 0.5, "type": float},
    "duration": {"arg": "tagging_duration", "default": 1.5, "type": float},
    "bin_size": {"arg": "tagging_bin_size", "default": 0.001, "type": float},
    "increase_in_fr": {"arg": "tagging_increase_in_fr", "default": 3.0, "type": float},
    "min_evoked_rate": {"arg": "tagging_min_evoked_rate", "default": 50.0, "type": float},
    "baseline_start_ms": {"arg": "tagging_baseline_start_ms", "default": -10.0, "type": float},
    "baseline_end_ms": {"arg": "tagging_baseline_end_ms", "default": -2.0, "type": float},
    "evoked_start_ms": {"arg": "tagging_evoked_start_ms", "default": 1.0, "type": float},
    "evoked_end_ms": {"arg": "tagging_evoked_end_ms", "default": 9.0, "type": float},
    "max_pulse_duration": {"arg": "tagging_max_pulse_duration", "default": 0.1, "type": float},
    "pulse_level": {"arg": "tagging_pulse_level", "default": 1.7, "type": float},
}
for name, spec in TAGGING_SPECS.items():
    cli_flag = "--" + spec["arg"].replace("_", "-")
    parser.add_argument(cli_flag, type=spec["type"], default=spec["default"])

class Pipeline(BrainsetPipeline):
    brainset_id = "allen_vbn_2022"
    parser = parser

    @classmethod
    # Passes in the raw data directory and additional arguments 
    def get_manifest(cls, raw_dir : Path, args) -> pd.DataFrame:
        # Use given directory to find cache
        cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
            cache_dir = raw_dir
        )

        # Get sessions table from cache
        sessions_table = cache.get_ecephys_session_table()

        # Ensure manifest index is a string
        # All the rows in the ecephys_session_table correspond 
        # to a single ecephys_session_id
        # Conversion from Int64Dtype -> String
        sessions_table["session_id"] = sessions_table.index
        sessions_table.index = [str(x) for x in sessions_table.index]

        return sessions_table

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")

        # Find cache
        cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
            cache_dir = self.raw_dir
        )

        # Extract single session information
        # This downloads the raw session data from AllenSDK
        session = cache.get_ecephys_session(
            ecephys_session_id = manifest_item.session_id
        )
        
        return session, manifest_item

    def process(self, download_output):
        self.update_status("PROCESSING")

        # Receive from download()
        session , manifest_item = download_output

        self.processed_dir.mkdir(exist_ok=True, parents=True)

        # Output file name is session-id-based
        store_path = self.processed_dir / f"{manifest_item.Index}.h5"

        # Return if session is already processed and user doesn't want reprocessing
        if (store_path.exists() and not self.args.reprocess):
            return
        
        # Grab unit filters - either default or overidden with arguments
        unit_filter_config = {
            name: getattr(self.args, FILTER_SPECS[name]["arg"])
            for name in DEFAULT_FILTERS
        }

        tagging_config = {
            name: getattr(self.args, TAGGING_SPECS[name]["arg"])
            for name in DEFAULT_TAGGING_PARAMS
        }

        # Set Descriptions
        # Brainset Description
        brainset_description = BrainsetDescription(
            id = self.brainset_id,
            origin_version = "0.1.0",
            derived_version = "0.1.0",
            source = "allensdk",
            description = "Allen Institute Visual Behavior Neuropixels Dataset",
            )

        # Session Description
        session_description = SessionDescription(
            id = str(manifest_item.Index),
            recording_date = datetime.datetime.strptime(
                manifest_item.date_of_acquisition.split(" ")[0],
                "%Y-%m-%d"
                ),
            #TODO: add DISCRETE_VISUAL_BEHAVIOR to task.py in brainsets
            task = None,
            image_set = manifest_item.image_set,
            channel_count = manifest_item.channel_count,
            prior_exposure_to_image_set = manifest_item.prior_exposures_to_image_set,
            experience_level = manifest_item.experience_level,
            # Structure acronmys and others may be useful
            )

        # Subject Description
        subject_description = SubjectDescription(
            id = str(manifest_item.mouse_id),
            species = Species.MUS_MUSCULUS,
            genotype = manifest_item.genotype,
            sex = (Sex.MALE if manifest_item.sex == "M" else Sex.FEMALE),
            age = manifest_item.age_in_days,
            )

        # Extract Session Information
        self.update_status("EXTRACTING SESSION DATA")
        unit_data, spikes, intervals, domain = extract_session_data(
            session,
            manifest_item, 
            unit_filter_config,
            tagging_config,
            manifest_item.genotype
            )


        # Initialize Data object

        # SpikeData requires:
        #    spikes: Spikes : td.IrregularTimeSeries
        #    session: bsd.SessionDescription
        #    brainset: bsd.BrainsetDescription
        #    units: td.ArrayDict
        
        # The Data object will contain:
        # spikes : ITS of spikes with unit id per spike index
        # units : ArrayDict of filtered units with metadata 
        # intervals : Data object of Intervals for images, tasks, and optotagging times 
        # domain : Interval of first spike in session to last in session
        
        data = Data(
            session = session_description,
            brainset = brainset_description,
            subject = subject_description,
            spikes = spikes, # IrregularTimeSeries
            units = unit_data, # ArrayDict
            intervals = intervals, # Data
            domain = domain # Interval
        )

        # Save as hdf5
        self.update_status("STORING")

        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        logging.info(f"Saved session data to {store_path}")
