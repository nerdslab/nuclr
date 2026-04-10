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

from session_extractor import extract_behavior, extract_channels, extract_probes, extract_units

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache

# Setup argument extensions
parser = ArgumentParser()
parser.add_argument("--reprocess", action="store_true")
# TODO: Implement extended arguments
# Unit extraction filters? Spike quality filters? 


class Pipeline(BrainsetPipeline):

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

        # Output file name is session-id-based
        store_path = self.processed_dir / f"{session.id}.h5"

        # Return if session is already processed and user doesn't want reprocessing
        if (store_path.exists() and not self.args.reprocess):
            return

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
            id = self.brainset_id,
            recording_date = datetime.strptime(
                session.date_of_acquisition.split(" ")[0],
                "%Y-%m-%d"
                ),
            #TODO: add DISCRETE_VISUAL_BEHAVIOR to task.py in brainsets
            # Tracks session type for now (e.g., EPHYS_1_images_H_3uL_reward
            task = session.session_type,
            image_set = session.image_set,
            channel_count = session.channel_count,
            prior_exposure_to_image_set = session.prior_exposure_to_image_set,
            experience_level = session.experience_level,
            # Structure acronmys and others may be useful
            )

        # Subject Description
        subject_description = SubjectDescription(
            id = self.brainset_id,
            species = Species.MUS_MUSCULUS,
            mouse_id = session.mouse_id,
            genotype = session.genotype,
            sex = (Sex.MALE if session.sex == "M" else Sex.FEMALE),
            age = session.age_in_days,
            )

        # Extract Session Information



        # Initialize Data object
        # SpikeData requires:
        #    spikes: Spikes
        #    session: bsd.SessionDescription
        #    brainset: bsd.BrainsetDescription
        #    units: td.ArrayDict
        data = Data(
            session = session_description,
            brainset = brainset_description,
            subject = subject_description,
            spikes = ...,
            units = ...,
            stimulus_epochs = ...,
            domain = ...,
        )

        # Save as hdf5
        self.update_status("STORING")

        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        logging.info(f"Saved session data to {store_path}")