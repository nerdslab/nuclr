'''

A set of functions designed for extracting session-specific information
from the Allen Institute Visual Behavior Neuropixels Dataset

'''

import pandas as pd
import numpy as np

from temporaldata import IrregularTimeSeries, ArrayDict, Interval, Data

def extract_units(session : pd.DataFrame, unit_filter_config):
    # Extract units & channels
    units = session.get_units()
    channels = session.get_channels()

    # Combine units & channels data
    units = units.merge(channels, left_on = "peak_channel_id", right_index = True)

    # Filter by filter_config
    unit_mask = (
        (units.isi_violations < unit_filter_config["isi_violations"]) &
        (units.amplitude_cutoff < unit_filter_config["amplitude_cutoff"]) &
        (units.presence_ratio > unit_filter_config["presence_ratio"]) &
        (units.firing_rate > unit_filter_config["firing_rate"]) & 
        (units.quality == unit_filter_config["quality"]) & 
        (units.snr > unit_filter_config["snr"])
    )
    # Filter to visual cortex only 
    vis_mask = units["structure_acronym"].astype(str).str.startswith("VIS")
    
    units = units[unit_mask & vis_mask]

    return units


def extract_spikes(session : pd.DataFrame, selected_units : pd.DataFrame):
    # Extract extract_spikes
    spikes = session.spike_times
    
    spike_times = [] # Keep an array of all units spike times
    spike_unit_indices = [] # Track indices for spike times

    # Filter by selected units
    # spikes is type `dict`
    for df_id, unit_id in enumerate(selected_units.index):
        # Extract unit_id's spikes
        unit_spikes = spikes[unit_id]

        # Extend the array with spikes
        spike_times.extend(unit_spikes)

        # Extend index tracker
        spike_unit_indices.extend([df_id] * len(unit_spikes))

    # Turn into IrregularTimeSeries
    # Includes timestamps and unit_index attributes
    spikes = IrregularTimeSeries(
        timestamps = np.array(spike_times),
        unit_index = np.array(spike_unit_indices),
        domain = "auto",
    )

    # Sort the spikes
    spikes.sort()

    # Turn selected units into an ArrayDict
    # Make sure index is the unit id
    units = ArrayDict.from_dataframe(df = selected_units)
    units.id = selected_units.index.values 

    return spikes, units

def interval_sorter(session):
    # Data object for storing task intervals
    # Useful for downstream window selection
    stimulus_table = session.stimulus_presentations
    
    # Active
    active_rows = stimulus_table[stimulus_table["stimulus_block"] == 0]
    active_intervals = Interval(
        start = np.array([active_rows["start_time"].min()]),
        end = np.array([active_rows["end_time"].max()]),
        block = np.array([0]),
        task = np.array(["active_behavior"]),
        stimulus_name = np.array([active_rows["stimulus_name"].iloc[0]]).astype(str)
    )

    # Spontaneous
    spontaneous_rows = stimulus_table[stimulus_table["stimulus_block"].isin([1, 3])]
    spontaneous_intervals = Interval(
        start=spontaneous_rows["start_time"].values,
        end=spontaneous_rows["end_time"].values,
        block=spontaneous_rows["stimulus_block"].values,
        task=np.array(["spontaneous"] * len(spontaneous_rows)),
        stimulus_name=spontaneous_rows["stimulus_name"].values.astype(str),
        duration=spontaneous_rows["duration"].values,
    )

    # Gabor
    gabor_rows = stimulus_table[stimulus_table["stimulus_block"] == 2]
    gabor_intervals = Interval(
        start = np.array([gabor_rows["start_time"].min()]),
        end = np.array([gabor_rows["end_time"].max()]),
        block = np.array([2]),
        task = np.array(["gabor"]),
        stimulus_name = np.array([gabor_rows["stimulus_name"].iloc[0]]).astype(str),
    )

    # Flash
    flash_rows = stimulus_table[stimulus_table["stimulus_block"] == 4]
    flash_intervals = Interval(
        start = np.array([flash_rows["start_time"].min()]),
        end = np.array([flash_rows["end_time"].max()]),
        block = np.array([4]),
        task = np.array(["flash"]),
        stimulus_name = np.array([flash_rows["stimulus_name"].iloc[0]]).astype(str),
    )

    # Passive
    passive_rows = stimulus_table[stimulus_table["stimulus_block"] == 5]
    passive_intervals = Interval(
        start=np.array([passive_rows["start_time"].min()]),
        end=np.array([passive_rows["end_time"].max()]),
        block=np.array([5]),
        task=np.array(["passive_replay"]),
        stimulus_name=np.array([passive_rows["stimulus_name"].iloc[0]]).astype(str),
    )

    # Optotagging
    optotagging_table = session.optotagging_table
    optotagging_intervals = Interval(
        start = optotagging_table["start_time"].values,
        end = optotagging_table["stop_time"].values,
        laser_type = optotagging_table["stimulus_name"].values.astype(str),
        laser_level = optotagging_table["level"].values,
        laser_duration = optotagging_table["duration"].values,
    )
    
    intervals = Data(
        active = active_intervals,
        spontaneous = spontaneous_intervals,
        gabor= gabor_intervals,
        flash = flash_intervals,
        passive = passive_intervals,
        optotagging = optotagging_intervals,
        domain = "auto"
    )

    return intervals

def domain_setter(spikes, intervals, safety_time):
    # Get first and last spike in session
    start_time = spikes.domain.start[0] + safety_time
    end_time = spikes.domain.end[0] - safety_time

    domain = Interval(
        start = start_time,
        end = end_time
    )
    return domain

def extract_session_data(session, manifest_item, unit_filter_config):

    # Find units
    selected_units = extract_units(session, unit_filter_config)
    
    # Find spikes
    spikes, units = extract_spikes(session, selected_units)

    # Find intervals
    intervals = interval_sorter(session)

    # Calculate domain
    domain = domain_setter(spikes, intervals, 0)

    return units, spikes, intervals, domain