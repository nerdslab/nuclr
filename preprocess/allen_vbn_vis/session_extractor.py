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

def interval_sorter():
    # Should be stored as a Data object
    return

def extract_session_data(session, manifest_item, unit_filter_config):

    # Find units
    selected_units = extract_units(session, unit_filter_config)
    
    # Find spikes
    spikes, units = extract_spikes(session, selected_units)

    # Find intervals
    intervals = interval_sorter()

    # Calculate domain
    domain = Interval(start = spikes.domain.start[0],
                      end = ...) #TODO: Should be end of optotagging task

    return units, spikes, intervals, domain