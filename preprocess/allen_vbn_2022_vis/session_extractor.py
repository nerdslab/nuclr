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

def standardize_genotype(genotype):
    genotype = str(genotype).lower()
    if "sst" in genotype:
        return "Sst"
    if "vip" in genotype:
        return "Vip"
    return "wt"


def compute_trial_reliability_and_latency(
    spike_times_by_unit,
    unit_ids,
    pulse_times,
    evoked_start_s,
    evoked_end_s,
):
    """
    Compute trial-level optotagging consistency for each unit.

    Reliability is the fraction of selected laser pulses that evoke at least one
    spike in the configured evoked window. Latency is the mean first-spike time,
    in milliseconds after laser onset, over trials with at least one evoked
    spike. Trials without evoked spikes are excluded from the latency average
    and contribute only to lower reliability.
    """
    unit_ids = list(unit_ids)
    n_trials = len(pulse_times)
    reliability = np.zeros(len(unit_ids), dtype=float)
    first_spike_latency_ms = np.full(len(unit_ids), np.nan, dtype=float)

    # No trials
    if n_trials == 0:
        return reliability, first_spike_latency_ms

    # Calculate reliability & FSL
    for unit_idx, unit_id in enumerate(unit_ids):
        spikes = spike_times_by_unit.get(unit_id)
        if spikes is None or len(spikes) == 0:
            continue

        first_spikes = []
        for pulse_time in pulse_times:
            evoked_spikes = spikes[
                (spikes >= pulse_time + evoked_start_s) &
                (spikes < pulse_time + evoked_end_s)
            ]
            if len(evoked_spikes) > 0:
                first_spikes.append(float(np.min(evoked_spikes) - pulse_time))

        reliability[unit_idx] = len(first_spikes) / n_trials
        if first_spikes:
            first_spike_latency_ms[unit_idx] = np.mean(first_spikes) * 1000

    return reliability, first_spike_latency_ms


def add_cell_types(
    units,
    spike_times_by_unit,
    optotagging_table,
    tagging_config: dict,
    genotype_label,
):
    units = units.copy()

    # Determine genotype
    genotype = standardize_genotype(genotype_label)

    # PSTH settings
    time_before = tagging_config.get("time_before")
    duration = tagging_config.get("duration")
    bin_size = tagging_config.get("bin_size")
    # Tagging settings
    increase_in_fr = tagging_config.get("increase_in_fr")
    min_evoked_rate = tagging_config.get("min_evoked_rate")
    baseline_start_ms = tagging_config.get("baseline_start_ms")
    baseline_end_ms = tagging_config.get("baseline_end_ms")
    evoked_start_ms = tagging_config.get("evoked_start_ms")
    evoked_end_ms = tagging_config.get("evoked_end_ms")
    min_trial_reliability = tagging_config.get("min_trial_reliability", 0.0)
    max_first_spike_latency_ms = tagging_config.get(
        "max_first_spike_latency_ms",
        None
    )
    # Laser settings
    max_pulse_duration = tagging_config.get("max_pulse_duration")
    pulse_level = tagging_config.get("pulse_level")

    # Find start times of pulses
    selected_pulses = optotagging_table[
        optotagging_table["duration"] <= max_pulse_duration
    ]
    if pulse_level is None:
        pulse_level = selected_pulses["level"].max()
    selected_pulses = selected_pulses[
        np.isclose(selected_pulses["level"], pulse_level)
    ]
    pulse_times = selected_pulses['start_time'].values
    unit_ids = list(units.index)
    n_units = len(unit_ids)

    # Create population PSTH 
    # optotagging array (units x bins x trials)
    opto_array, time_bins, unit_ids = make_population_psth(
        spike_times_by_unit, unit_ids, pulse_times, time_before, duration, bin_size
    )

    # Calculate average baseline and evoked rates from the PSTH
    # These rates capture response magnitude, while reliability below
    # captures whether that response is repeatable across laser pulses
    baseline_idx = (
        (time_bins >= baseline_start_ms / 1000) &
        (time_bins < baseline_end_ms / 1000)
    )
    evoked_idx = (
        (time_bins >= evoked_start_ms / 1000) & 
        (time_bins < evoked_end_ms / 1000)
    )

    if len(pulse_times) == 0:
        baseline_rate = np.full(n_units, np.nan)
        evoked_rate = np.full(n_units, np.nan)
    else:
        mean_opto = np.nanmean(opto_array, axis=2) # (units, time_bins)
        baseline_rate = np.mean(mean_opto[:, baseline_idx], axis=1)
        evoked_rate = np.mean(mean_opto[:, evoked_idx], axis=1)

    response_ratio = evoked_rate / (baseline_rate + 1)

    trial_reliability, first_spike_latency_ms = compute_trial_reliability_and_latency(
        spike_times_by_unit=spike_times_by_unit,
        unit_ids=unit_ids,
        pulse_times=pulse_times,
        evoked_start_s=evoked_start_ms / 1000,
        evoked_end_s=evoked_end_ms / 1000,
    )

    # A unit is optotagged only if it has a large, baseline-relative evoked
    # response that is repeatable across pulses and fast enough to be consistent
    # with direct optogenetic activation.
    latency_gate = np.ones(n_units, dtype=bool)
    if max_first_spike_latency_ms is not None:
        latency_gate = (
            np.isfinite(first_spike_latency_ms) &
            (first_spike_latency_ms <= max_first_spike_latency_ms)
        )

    cre_pos_idx = (
        (genotype != "wt") &
        (evoked_rate > min_evoked_rate) &
        (response_ratio > increase_in_fr) &
        (trial_reliability >= min_trial_reliability) &
        latency_gate
    )

    optotagged_cell_type = np.full(len(units), "wt", dtype=object)
    optotagged_cell_type[cre_pos_idx] = genotype

    units["optotagging_baseline_rate"] = baseline_rate
    units["optotagging_evoked_rate"] = evoked_rate
    units["optotagging_response_ratio"] = response_ratio
    units["optotagging_trial_reliability"] = trial_reliability
    units["optotagging_first_spike_latency_ms"] = first_spike_latency_ms
    units["optotagging_cre_positive"] = cre_pos_idx
    # Keep the existing column name as a per-unit compatibility alias.
    units["cre_positive"] = cre_pos_idx
    units["optotagged_cell_type"] = optotagged_cell_type

    return units

def make_population_psth(spike_times_by_unit, unit_ids, start_times, time_before, duration, bin_size,
                         mean_over_trials=False):
    """
    Population level of make_psth function.

    Parameters
    ----------
    mean_over_trials : bool
        If True, return shape (n_units, n_bins) averaged over trials instead of
        the full (n_units, n_bins, n_trials) array.  Use this in batch scripts
        to avoid allocating the full 3-D array in memory.
    """
    bins = np.arange(-time_before, duration - time_before + bin_size, bin_size)
    n_bins = len(bins) - 1
    n_trials = len(start_times)
    unit_ids = list(unit_ids)

    if mean_over_trials:
        psth_array = np.zeros((len(unit_ids), n_bins))
        for i, uid in enumerate(unit_ids):
            spikes = spike_times_by_unit.get(uid)
            if spikes is None:
                continue
            for t in start_times:
                window_spikes = spikes[(spikes >= t - time_before) & (spikes < t + duration - time_before)]
                psth_array[i] += np.histogram(window_spikes - t, bins=bins)[0] / bin_size
        psth_array /= n_trials
        return psth_array, bins[:-1], unit_ids

    psth_array = np.zeros((len(unit_ids), n_bins, n_trials))
    for i, uid in enumerate(unit_ids):
        spikes = spike_times_by_unit.get(uid)
        if spikes is None:
            continue
        for j, t in enumerate(start_times):
            window_spikes = spikes[(spikes >= t - time_before) & (spikes < t + duration - time_before)]
            psth_array[i, :, j] = np.histogram(window_spikes - t, bins=bins)[0] / bin_size

    return psth_array, bins[:-1], unit_ids

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

def extract_session_data( session, manifest_item, unit_filter_config, tagging_config, genotype):

    # Find units
    selected_units = extract_units(session, unit_filter_config)

    # Classify unit cell types
    classified_units = add_cell_types(
        selected_units, session.spike_times, session.optotagging_table, tagging_config, genotype
        )
    
    # Find spikes
    spikes, units = extract_spikes(session, classified_units)

    # Find intervals
    intervals = interval_sorter(session)

    # Calculate domain
    domain = domain_setter(spikes, intervals, 0)

    return units, spikes, intervals, domain
