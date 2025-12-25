# /// brainset-pipeline
# python-version = "3.10"
# dependencies = ["ONE-api==2.11.2", "ibllib==2.39.1"]
# ///

# Mainly taken from:
# https://github.com/neuro-galaxy/brainsets/tree/181f25bdf931606117bd004683cb6812b69c2a58/brainsets_pipelines/ibl_reproducible_ephys_2022

from argparse import ArgumentParser
from datetime import datetime
import logging
import h5py
from pathlib import Path

import numpy as np
import pandas as pd
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader

from temporaldata import Data, IrregularTimeSeries, Interval, ArrayDict
from brainsets.descriptions import (
    BrainsetDescription,
    SubjectDescription,
    SessionDescription,
    DeviceDescription,
)

from brainsets.taxonomy import RecordingTech, Sex, Species
from brainsets import serialize_fn_map
from brainsets.pipeline import BrainsetPipeline

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--reprocess", action="store_true")


class Pipeline(BrainsetPipeline):
    brainset_id = "ibl_brainwide_map_qc"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir: Path, args):
        # Use a precomputed list of eids
        pipeline_dir = Path(__file__).resolve().parent
        with open(pipeline_dir / "eids.txt") as fh:
            manifest_list = [{"id": line.strip(), "eid": line.strip()} for line in fh]
        manifest = pd.DataFrame(manifest_list).set_index("id")

        # Connect to the ONE sdk
        # do this once only, and copy the handle for all manifest items
        # otherwise: error from token file being overwritten
        one = ONE(
            base_url="https://openalyx.internationalbrainlab.org",
            password="international",
            cache_dir=raw_dir,
        )
        manifest["one"] = one

        return manifest

    def download(self, manifest_item):
        return manifest_item

    def process(self, download_output):
        manifest_item = download_output
        eid = manifest_item.eid
        one = manifest_item.one
        logging.info(f"Processing session eid: {eid}")

        self.processed_dir.mkdir(exist_ok=True, parents=True)

        store_path = self.processed_dir / f"{eid}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        # intiantiate a DatasetBuilder which provides utilities for processing data
        brainset = BrainsetDescription(
            id="ibl_brainwide_map_qc",
            origin_version="",
            derived_version="1.0.0",
            source="https://int-brain-lab.github.io/iblenv/notebooks_external/data_release_repro_ephys.html",
            description="Repeatedly inserted Neuropixels multi-electrode probes targeting "
            "the same brain locations (called the repeated site, including posterior "
            "parietal cortex, hippocampus, and thalamus) in mice performing a behavioral "
            "task.",
        )

        # get subject metadata
        self.update_status("Extracting metadata")
        subject_id = one.get_details(eid)["subject"]

        subject = SubjectDescription(
            id=subject_id,
            species=Species.MUS_MUSCULUS,
            sex=Sex.MALE,
        )

        # extract experiment metadata
        recording_date = datetime.fromisoformat(
            one.get_details(eid)["start_time"]
        ).strftime("%Y%m%d")
        device_id = f"{subject.id}_{recording_date}"

        # register session
        session_description = SessionDescription(
            id=eid,
            recording_date=datetime.strptime(recording_date, "%Y%m%d"),
        )

        # register device
        device_description = DeviceDescription(
            id=device_id,  # TODO: update
            recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,  # TODO: update
        )

        # extract spiking activity
        self.update_status("Extracting Spikes")
        spikes, units = extract_spikes(one, eid)

        domain = Interval(spikes.domain.start[0], spikes.domain.end[0] - 300.0)
        # sometimes last few minutes dont have any spikes

        # register session
        data = Data(
            brainset=brainset,
            subject=subject,
            session=session_description,
            device=device_description,
            # neural activity
            spikes=spikes,
            units=units,
            domain=domain,
        )

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        logging.info(f"Saved session data to {store_path}")


def extract_spikes(one, eid):
    # get probes
    pids, probe_names = one.eid2pid(eid)

    spikes_list = []
    units_list = []
    unit_ptr = 0

    # iterate over probes and load spikes and clusters
    for pid, probe_name in zip(pids, probe_names):
        spike_loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=probe_name)
        spikes, clusters, channels = spike_loader.load_spike_sorting()

        clusters = SpikeSortingLoader.merge_clusters(
            spikes, clusters, channels, compute_metrics=False
        )
        assert clusters is not None

        clusters["probe_id"] = pid

        spikes["clusters"] += unit_ptr

        assert len(clusters["cluster_id"]) == len(
            np.unique(clusters["cluster_id"])
        ), f"There are duplicate units in {eid}"

        assert len(clusters["cluster_id"]) == len(
            np.unique(spikes["clusters"])
        ), f"There are units that have no spikes in {eid}"

        num_units = len(clusters["cluster_id"])
        unit_ptr += num_units

        spikes_list.append(spikes)
        units = pd.DataFrame(clusters).rename(columns={"uuids": "id"})

        units_list.append(units)

    units = ArrayDict.from_dataframe(
        pd.concat(units_list, ignore_index=True),
        unsigned_to_long=True,
    )

    timestamps = np.concatenate([s["times"] for s in spikes_list])
    unit_index = np.concatenate([s["clusters"] for s in spikes_list])

    # ignore units with firing rates less than 2Hz
    # unit_mask = units.firing_rate >= 2
    # if "ks2_label" in units.keys():
    #    unit_mask = unit_mask & (units.ks2_label == "good")

    # QC metric. Taken from: https://neurostars.org/t/kilosort-cluster-labels/29520/2
    unit_mask = units.label == 1.0
    print(f"QC: {len(unit_mask)=}, {sum(unit_mask)=}, {np.mean(unit_mask)=}")

    valid_unit_idx = np.where(unit_mask)[0]
    unit_remap = np.zeros(len(units), dtype=int)
    unit_remap[unit_mask] = np.arange(unit_mask.sum())

    units = units.select_by_mask(unit_mask)
    spike_mask = np.isin(unit_index, valid_unit_idx)
    timestamps = timestamps[spike_mask]
    unit_index = unit_remap[unit_index[spike_mask]]

    spikes = IrregularTimeSeries(
        timestamps=timestamps,
        unit_index=unit_index,
        # amplitude=np.concatenate([s["amps"] for s in spikes_list]),
        # depth=np.concatenate([s["depths"] for s in spikes_list]),
        domain="auto",
    )
    spikes.sort()

    # check that id is unique
    assert len(units.id) == len(np.unique(units.id)), "uuids is not unique"

    return spikes, units
