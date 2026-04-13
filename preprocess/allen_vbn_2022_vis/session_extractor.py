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
        (units.presence_ratio > unit_filter_config["presence_ratio"])
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
    # Should be stored as a Data object
    # Used as downstream information for which windows are valid to select
    # from for temporaldata operations (e.g., AND, OR)
    # we encode task label and intervals for that task

    # TODO: More task intervals

    optotagging_table = session.optotagging_table
    opto_intervals = Interval(
        start = optotagging_table["start_time"].values,
        end = optotagging_table["stop_time"].values,
        laser_type = optotagging_table["stimulus_name"].values.astype(str),
        laser_level = optotagging_table["level"].values,
        laser_duration = optotagging_table["duration"].values,
    )
    
    intervals = Data(
        optotagging = opto_intervals,
        domain = "auto"
    )

    return intervals

def domain_setter(spikes, intervals, safety_time):
    # Get first spike in session
    start_time = spikes.domain.start[0]
    # Get the time before optotagging task
    end_time = intervals.optotagging.start[0] - safety_time

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
    domain = domain_setter(spikes, intervals, 300) # 300s before optotagging

    return units, spikes, intervals, domain