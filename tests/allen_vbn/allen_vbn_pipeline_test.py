"""

Allen VBN session-level pipeline validation:

-> This script validates one processed Allen Visual Behavior Neuropixels session
HDF5 file

-> Run this after the preprocessing pipeline has written the session-level dataset

"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from temporaldata import Data
from torch_brain.data import Dataset


DEFAULT_SESSION_ID = "1095138995"
DEFAULT_BRAINSET_ID = "allen_vbn_2022"
DEFAULT_WINDOW_SECONDS = 1.0
DEFAULT_CHUNK_SIZE = 1_000_000


REQUIRED_TOP_LEVEL_GROUPS = {
    "brainset",
    "session",
    "subject",
    "spikes",
    "units",
    "intervals",
    "domain",
}

REQUIRED_SPIKE_FIELDS = {"timestamps", "unit_index", "domain"}

REQUIRED_UNIT_FIELDS = {
    "id",
    "probe_id",
    "structure_acronym",
    "isi_violations",
    "amplitude_cutoff",
    "presence_ratio",
    "firing_rate",
    "quality",
    "snr",
}


@dataclass(frozen=True)
class UnitFilters:
    isi_violations_max: float
    amplitude_cutoff_max: float
    presence_ratio_min: float
    firing_rate_min: float
    quality: str
    snr_min: float

    def as_pipeline_kwargs(self) -> dict[str, Any]:
        return {
            "isi_violations": self.isi_violations_max,
            "amplitude_cutoff": self.amplitude_cutoff_max,
            "presence_ratio": self.presence_ratio_min,
            "firing_rate": self.firing_rate_min,
            "quality": self.quality,
            "snr": self.snr_min,
        }


@dataclass(frozen=True)
class TaskIntervalSpec:
    name: str
    stimulus_blocks: tuple[int, ...]
    expected_rows: int


TASK_INTERVALS = (
    TaskIntervalSpec("active", (0,), 1),
    TaskIntervalSpec("spontaneous", (1, 3), 2),
    TaskIntervalSpec("gabor", (2,), 1),
    TaskIntervalSpec("flash", (4,), 1),
    TaskIntervalSpec("passive", (5,), 1),
)

REQUIRED_INTERVALS = {spec.name for spec in TASK_INTERVALS} | {"optotagging"}


def as_array(value: Any) -> np.ndarray:
    """Materialize lazy array-like values into a NumPy array."""
    if hasattr(value, "__array__") and not isinstance(value, np.ndarray):
        value = value[:]
    return np.asarray(value)


def as_float_array(value: Any) -> np.ndarray:
    """Convert numeric array-like values to float arrays for comparisons."""
    return as_array(value).astype(float)


def as_str_array(value: Any) -> np.ndarray:
    """Normalize bytes/object arrays into plain string arrays."""
    arr = as_array(value)

    if arr.dtype.kind == "S":
        return arr.astype(str)

    if arr.dtype.kind == "O":
        return np.array(
            [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]
        )

    return arr.astype(str)


def print_ok(message: str) -> None:
    print(f"[ok] {message}")


def repo_root() -> Path:
    """Return the repository root regardless of the current working directory."""
    return Path(__file__).resolve().parents[2]


def default_processed_dir() -> Path:
    """
    Resolve the default processed-data root from the repo location.

    This keeps the script stable even after moving it under `tests/allen_vbn`
    and avoids relying on whatever the current working directory happens to be.
    """
    return repo_root().parent / "data" / "processed"


def resolve_h5_path(processed_dir: Path, brainset_id: str, session_id: str) -> Path:
    """
    Find the target session HDF5.

    Support both:
    - a processed root that contains `<brainset_id>/<session_id>.h5`
    - a brainset-specific directory that directly contains `<session_id>.h5`
    """
    direct = processed_dir / f"{session_id}.h5"
    nested = processed_dir / brainset_id / f"{session_id}.h5"

    if nested.exists():
        return nested
    if direct.exists():
        return direct

    raise FileNotFoundError(
        "Could not find processed HDF5 at either "
        f"{nested} or {direct}. Pass --processed-dir as the processed root "
        "or as the brainset-specific directory."
    )


def resolve_processed_root(processed_dir: Path, brainset_id: str) -> Path:
    """
    Recover the processed-data root expected by `torch_brain.Dataset`.

    The Dataset API wants the directory that contains the brainset folder, not
    the individual brainset folder itself.
    """
    if (processed_dir / brainset_id).exists():
        return processed_dir
    return processed_dir.parent


def assert_group_contains(
    h5_group: h5py.Group,
    required_names: set[str],
    description: str,
) -> None:
    """Assert that an HDF5 group contains the required child names."""
    missing = required_names - set(h5_group.keys())
    assert not missing, f"missing {description}: {sorted(missing)}"


def validate_hdf5_layout(path: Path) -> None:
    """Check the high-level HDF5 structure written by the pipeline."""
    with h5py.File(path, "r") as f:
        assert_group_contains(f, REQUIRED_TOP_LEVEL_GROUPS, "top-level HDF5 groups")
        assert_group_contains(f["spikes"], REQUIRED_SPIKE_FIELDS, "spike datasets")
        assert_group_contains(f["units"], REQUIRED_UNIT_FIELDS, "unit datasets")
        assert_group_contains(f["intervals"], REQUIRED_INTERVALS, "interval groups")

    print_ok("HDF5 group layout has required fields")


def validate_core_metadata(data: Data, session_id: str, brainset_id: str) -> None:
    """Validate the core brainset/session/subject metadata fields."""
    assert data.brainset.id == brainset_id, (
        f"brainset id mismatch: {data.brainset.id} != {brainset_id}"
    )
    assert data.session.id == session_id, (
        f"session id mismatch: {data.session.id} != {session_id}"
    )
    assert str(data.subject.id), "subject.id is empty"
    assert str(data.subject.species) == "MUS_MUSCULUS", (
        f"unexpected species: {data.subject.species}"
    )
    assert str(data.subject.sex) in {"MALE", "FEMALE"}, (
        f"unexpected sex: {data.subject.sex}"
    )
    assert float(data.subject.age) > 0, "subject age should be positive"

    print_ok("Core brainset/session/subject metadata is present")


def validate_units(data: Data, filters: UnitFilters) -> None:
    """
    Validate saved units against the pipeline's selection assumptions.

    This checks both structural properties, like unique ids, and the filtering
    thresholds that decide which units survive preprocessing.
    """
    unit_ids = as_array(data.units.id).astype(str)
    probe_ids = as_array(data.units.probe_id)
    structures = as_str_array(data.units.structure_acronym)
    qualities = as_str_array(data.units.quality)

    assert len(unit_ids) > 0, "no units were saved"
    assert len(np.unique(unit_ids)) == len(unit_ids), "unit IDs duplicate"
    assert len(np.unique(probe_ids)) > 0, "no probe IDs found"
    assert np.all(np.char.startswith(structures, "VIS")), (
        "non-visual-cortex units are present"
    )

    assert np.all(as_float_array(data.units.isi_violations) < filters.isi_violations_max)
    assert np.all(
        as_float_array(data.units.amplitude_cutoff) < filters.amplitude_cutoff_max
    )
    assert np.all(as_float_array(data.units.presence_ratio) > filters.presence_ratio_min)
    assert np.all(as_float_array(data.units.firing_rate) > filters.firing_rate_min)
    assert np.all(as_float_array(data.units.snr) > filters.snr_min)
    assert np.all(qualities == filters.quality), "unit quality filter mismatch"

    print_ok(
        f"Units pass filters: {len(unit_ids)} units across "
        f"{len(np.unique(probe_ids))} probes"
    )


def validate_sorted_dataset(dset: h5py.Dataset, chunk_size: int) -> None:
    """Verify a large HDF5 dataset is globally sorted without loading it all at once."""
    if len(dset) < 2:
        return

    previous = None
    for start in range(0, len(dset), chunk_size):
        chunk = dset[start : start + chunk_size]
        if previous is not None:
            assert previous <= chunk[0], "spike timestamps are not sorted"
        assert np.all(np.diff(chunk) >= 0), "spike timestamps are not sorted"
        previous = chunk[-1]


def interval_start_end(interval: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return interval start/end arrays as float arrays for easy comparison."""
    return as_float_array(interval.start), as_float_array(interval.end)


def validate_spikes_and_domains(path: Path, data: Data, chunk_size: int) -> None:
    """
    Validate spike storage and the top-level time domain.

    The session-level file should contain sorted spikes, valid local unit
    indexing, and a domain that encloses the saved spike timestamps.
    """
    with h5py.File(path, "r") as f:
        timestamps = f["spikes/timestamps"]
        unit_index = f["spikes/unit_index"][:]

        assert len(timestamps) > 0, "no spikes were saved"
        assert len(timestamps) == len(unit_index), "spike timestamps/unit_index mismatch"
        validate_sorted_dataset(timestamps, chunk_size)

        num_units = len(data.units.id)
        assert unit_index.min() >= 0, "spike unit_index contains negative values"
        assert unit_index.max() < num_units, "spike unit_index exceeds units length"
        counts = np.bincount(unit_index, minlength=num_units)
        assert np.all(counts > 0), "at least one saved unit has no spikes"

        first_ts = float(timestamps[0])
        last_ts = float(timestamps[-1])

    spike_start, spike_end = interval_start_end(data.spikes.domain)
    data_start, data_end = interval_start_end(data.domain)

    assert len(spike_start) == 1 and len(spike_end) == 1, "spikes.domain should be 1 row"
    assert len(data_start) >= 1 and len(data_end) >= 1, "data.domain is empty"
    assert spike_start[0] <= first_ts <= spike_end[-1]
    assert spike_start[0] <= last_ts <= spike_end[-1]
    assert data_start[0] >= spike_start[0] - 1e-6, "data.domain starts before spikes"
    assert data_end[-1] <= spike_end[-1] + 1e-6, "data.domain extends past spikes"

    print_ok(
        f"Spikes are sorted and indexed: {len(unit_index):,} spikes, "
        f"unit_index range [{unit_index.min()}, {unit_index.max()}]"
    )
    print_ok(
        f"Domain valid: data [{data_start[0]:.3f}, {data_end[-1]:.3f}], "
        f"spikes [{spike_start[0]:.3f}, {spike_end[-1]:.3f}]"
    )


def validate_interval(
    interval: Any,
    name: str,
    *,
    expected_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one Interval object and return its start/end arrays."""
    start, end = interval_start_end(interval)

    assert start.ndim == 1, f"{name}.start should be 1D"
    assert end.ndim == 1, f"{name}.end should be 1D"
    assert len(start) == len(end), f"{name}.start/end length mismatch"
    assert len(start) > 0, f"{name} interval is empty"
    if expected_rows is not None:
        assert len(start) == expected_rows, f"{name} should have {expected_rows} rows"
    assert np.all(np.isfinite(start)), f"{name}.start contains non-finite values"
    assert np.all(np.isfinite(end)), f"{name}.end contains non-finite values"
    assert np.all(end > start), f"{name} has non-positive duration"
    assert np.all(np.diff(start) >= 0), f"{name}.start is not sorted"

    return start, end


def validate_task_intervals(data: Data) -> None:
    """
    Validate the coarse task blocks saved in `data.intervals`.

    These checks make sure each expected task block is present, correctly
    labeled, ordered in time, and split into the expected number of rows.
    """
    starts: dict[str, np.ndarray] = {}
    ends: dict[str, np.ndarray] = {}

    for spec in TASK_INTERVALS:
        interval = getattr(data.intervals, spec.name)
        start, end = validate_interval(
            interval,
            spec.name,
            expected_rows=spec.expected_rows,
        )
        starts[spec.name] = start
        ends[spec.name] = end

        saved_blocks = as_array(interval.block)
        assert np.array_equal(saved_blocks, spec.stimulus_blocks), (
            f"{spec.name} block labels mismatch"
        )

    opto_start, _ = validate_interval(data.intervals.optotagging, "optotagging")

    assert ends["active"][0] <= starts["spontaneous"][0], (
        "active overlaps first spontaneous block"
    )
    assert ends["spontaneous"][0] <= starts["gabor"][0], (
        "first spontaneous overlaps gabor"
    )
    assert ends["gabor"][0] <= starts["spontaneous"][1], (
        "gabor overlaps second spontaneous block"
    )
    assert ends["spontaneous"][1] <= starts["flash"][0], (
        "second spontaneous overlaps flash"
    )
    assert ends["flash"][0] <= starts["passive"][0], "flash overlaps passive replay"
    assert ends["passive"][0] <= opto_start[0], "passive replay overlaps optotagging"

    print_ok(
        "Task intervals are ordered and labeled: "
        "active, spontaneous, gabor, flash, passive, optotagging"
    )


def validate_dataset_readback(
    processed_root: Path,
    brainset_id: str,
    session_id: str,
    window_seconds: float,
) -> None:
    """
    Perform one downstream smoke test through `torch_brain.Dataset`.

    This catches cases where the HDF5 looks valid on disk but fails when loaded
    through the project's actual sampling path.
    """
    recording_id = f"{brainset_id}/{session_id}"
    dataset = Dataset(root=str(processed_root), recording_id=recording_id)
    intervals = dataset.get_sampling_intervals()[recording_id]

    start = float(intervals.start[0])
    end = start + window_seconds
    assert end <= float(intervals.end[0]), "sampling domain too short for test window"

    sample = dataset.get(recording_id, start, end)
    assert sample.session.id == recording_id
    assert len(sample.units.id) > 0

    if len(sample.spikes.timestamps) > 0:
        assert float(sample.spikes.timestamps.min()) >= 0.0
        assert float(sample.spikes.timestamps.max()) <= window_seconds
        assert int(sample.spikes.unit_index.min()) >= 0
        assert int(sample.spikes.unit_index.max()) < len(sample.units.id)

    print_ok(
        f"torch_brain Dataset can sample {window_seconds:g}s window from {recording_id}"
    )


def validate_against_allensdk(
    raw_dir: Path,
    data: Data,
    session_id: str,
    filters: UnitFilters,
) -> None:
    """
    Compare the saved session file against the original AllenSDK session object.

    This is the strongest end-to-end check in the script: it verifies that the
    processed file agrees with the source metadata, selected units, and interval
    timing.
    """
    pipeline_dir = repo_root() / "preprocess" / "allen_vbn_2022_vis"
    sys.path.insert(0, str(pipeline_dir))

    from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_neuropixels_project_cache import (  # noqa: E501
        VisualBehaviorNeuropixelsProjectCache,
    )
    from session_extractor import extract_units

    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=raw_dir)
    session_table = cache.get_ecephys_session_table()
    manifest = session_table.loc[int(session_id)]
    session = cache.get_ecephys_session(ecephys_session_id=int(session_id))

    assert str(data.subject.id) == str(manifest.mouse_id)
    assert str(data.subject.sex) == ("MALE" if manifest.sex == "M" else "FEMALE")
    assert float(data.subject.age) == float(manifest.age_in_days)
    assert str(data.subject.genotype) == str(manifest.genotype)
    assert str(data.session.recording_date)[:10] == str(manifest.date_of_acquisition)[:10]

    selected_units = extract_units(session, filters.as_pipeline_kwargs())
    saved_unit_ids = set(as_array(data.units.id).astype(str))
    expected_unit_ids = set(selected_units.index.astype(str))
    assert saved_unit_ids == expected_unit_ids, (
        f"saved unit IDs do not match extract_units output: "
        f"{len(saved_unit_ids)} saved vs {len(expected_unit_ids)} expected"
    )

    validate_task_intervals_against_allensdk(session, data)
    print_ok("Saved HDF5 matches AllenSDK metadata, units, and interval timing")


def validate_task_intervals_against_allensdk(session: Any, data: Data) -> None:
    """Cross-check saved task intervals against AllenSDK stimulus tables."""
    stimulus_presentations = session.stimulus_presentations

    for spec in TASK_INTERVALS:
        expected_start = []
        expected_end = []
        for block in spec.stimulus_blocks:
            block_rows = stimulus_presentations[
                stimulus_presentations["stimulus_block"] == block
            ]
            expected_start.append(block_rows["start_time"].min())
            expected_end.append(block_rows["end_time"].max())

        saved_start, saved_end = interval_start_end(getattr(data.intervals, spec.name))
        assert np.allclose(saved_start, expected_start), (
            f"{spec.name} start mismatch vs AllenSDK"
        )
        assert np.allclose(saved_end, expected_end), (
            f"{spec.name} end mismatch vs AllenSDK"
        )

    optotagging = session.optotagging_table
    saved_start, saved_end = interval_start_end(data.intervals.optotagging)
    assert np.allclose(saved_start, optotagging["start_time"].values)
    assert np.allclose(saved_end, optotagging["stop_time"].values)


def build_unit_filters(args: argparse.Namespace) -> UnitFilters:
    """Convert CLI threshold arguments into the filter bundle used by the checks."""
    return UnitFilters(
        isi_violations_max=args.isi_violations_max,
        amplitude_cutoff_max=args.amplitude_cutoff_max,
        presence_ratio_min=args.presence_ratio_min,
        firing_rate_min=args.firing_rate_min,
        quality=args.quality,
        snr_min=args.snr_min,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one Allen VBN 2022 processed HDF5 file."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed_dir(),
        help=(
            "Processed-data root or the brainset-specific directory containing "
            "the session HDF5 files."
        ),
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--brainset-id", default=DEFAULT_BRAINSET_ID)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--isi-violations-max", type=float, default=0.5)
    parser.add_argument("--amplitude-cutoff-max", type=float, default=0.1)
    parser.add_argument("--presence-ratio-min", type=float, default=0.9)
    parser.add_argument("--firing-rate-min", type=float, default=0.1)
    parser.add_argument("--quality", type=str, default="good")
    parser.add_argument("--snr-min", type=float, default=1.0)
    parser.add_argument(
        "--skip-allensdk",
        action="store_true",
        help="Skip AllenSDK comparison even if --raw-dir is provided.",
    )
    return parser.parse_args()


def main() -> None:
    # Build the unit filter bundle
    args = parse_args()
    filters = build_unit_filters(args)

    # Resolve the target session file and the processed root used by the
    # downstream Dataset loader 
    h5_path = resolve_h5_path(args.processed_dir, args.brainset_id, args.session_id)
    processed_root = resolve_processed_root(args.processed_dir, args.brainset_id)

    print(f"Validating {h5_path}")

    # Cheap structural checks directly on the HDF5 layout 
    validate_hdf5_layout(h5_path)

    with h5py.File(h5_path, "r") as f:
        data = Data.from_hdf5(f, lazy=True)

        # Semantic checks on metadata, units, spikes, and task intervals 
        validate_core_metadata(data, args.session_id, args.brainset_id)
        validate_units(data, filters)
        validate_spikes_and_domains(h5_path, data, args.chunk_size)
        validate_task_intervals(data)

        # Optional source-of-truth comparison against the AllenSDK cache 
        if args.raw_dir is not None and not args.skip_allensdk:
            validate_against_allensdk(args.raw_dir, data, args.session_id, filters)

    # Final downstream readback through the project dataloader 
    validate_dataset_readback(
        processed_root=processed_root,
        brainset_id=args.brainset_id,
        session_id=args.session_id,
        window_seconds=args.window,
    )
    print("Allen VBN pipeline validation passed!")


if __name__ == "__main__":
    main()
