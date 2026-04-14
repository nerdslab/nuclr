"""
Allen VBN split-probe validation.

Run this suite only after:
1. generating the session-level Allen VBN HDF5 files with the preprocessing
   pipeline,
2. splitting those files into probe-level HDF5s with `utils/split_probes.py`,
3. generating the unit metadata CSV with `utils/neuron_md_gen.py`.

This script validates that the split probe files are a correct transformation of
the parent session files, that the metadata CSV matches those probe files, and
that the downstream Dataset loader can read from the split-probe brainset.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from temporaldata import Data
from torch_brain.data import Dataset


DEFAULT_PARENT_BRAINSET_ID = "allen_vbn_2022"
DEFAULT_PROBE_BRAINSET_ID = "allen_vbn_2022_probes"
DEFAULT_METADATA_CSV = "allen_vbn_2022.csv"
DEFAULT_WINDOW_SECONDS = 1.0


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

REQUIRED_METADATA_COLUMNS = {
    "id",
    "session_id",
    "subject_id",
    "probe_id",
    "structure_acronym",
}

PROBE_FILENAME_RE = re.compile(r"^(?P<session_id>\d+)_p(?P<probe_idx>\d+)$")


@dataclass(frozen=True)
class ProbeFileGroup:
    """All split probe files derived from one parent session."""

    parent_session_id: str
    probe_paths: list[Path]


def print_ok(message: str) -> None:
    print(f"[ok] {message}")


def repo_root() -> Path:
    """Return the repository root independent of the current working directory."""
    return Path(__file__).resolve().parents[2]


def default_parent_dir() -> Path:
    """Default directory containing parent session-level HDF5 files."""
    return repo_root().parent / "data" / "processed" / DEFAULT_PARENT_BRAINSET_ID


def default_probe_dir() -> Path:
    """Default directory containing split probe-level HDF5 files."""
    return repo_root().parent / "data" / "processed" / DEFAULT_PROBE_BRAINSET_ID


def default_metadata_csv() -> Path:
    """Default metadata CSV generated from the split probe files."""
    return repo_root() / "neuron_metadata" / DEFAULT_METADATA_CSV


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


def arrays_equal(expected: Any, actual: Any) -> bool:
    """Compare numeric or string arrays with sensible handling for NaNs/bytes."""
    expected_arr = as_array(expected)
    actual_arr = as_array(actual)

    if expected_arr.shape != actual_arr.shape:
        return False

    if (
        expected_arr.dtype.kind in {"S", "U", "O"}
        or actual_arr.dtype.kind in {"S", "U", "O"}
    ):
        return np.array_equal(as_str_array(expected_arr), as_str_array(actual_arr))

    if expected_arr.dtype.kind in {"f", "c"} or actual_arr.dtype.kind in {"f", "c"}:
        return np.allclose(expected_arr, actual_arr, equal_nan=True)

    return np.array_equal(expected_arr, actual_arr)


def assert_arrays_equal(expected: Any, actual: Any, label: str) -> None:
    """Assert that two arrays match exactly enough for split-file validation."""
    assert arrays_equal(expected, actual), f"{label} mismatch"


def arraydict_to_dataframe(arraydict: Any) -> pd.DataFrame:
    """Convert an ArrayDict-like object into a pandas DataFrame."""
    data = {}
    for key in arraydict.keys():
        value = getattr(arraydict, key)
        arr = as_array(value)
        if arr.dtype.kind in {"S", "U", "O"}:
            data[key] = as_str_array(arr)
        else:
            data[key] = arr
    return pd.DataFrame(data)


def resolve_brainset_dir(path: Path, brainset_id: str) -> Path:
    """
    Resolve a brainset directory from either a processed-data root or a direct
    brainset directory path.
    """
    nested = path / brainset_id
    if nested.exists():
        return nested
    return path


def resolve_processed_root(path: Path, brainset_id: str) -> Path:
    """Recover the processed-data root expected by `torch_brain.Dataset`."""
    nested = path / brainset_id
    if nested.exists():
        return path
    return path.parent


def parse_probe_filename(path: Path) -> tuple[str, int]:
    """Parse `<session_id>_p<probe_idx>.h5` and validate the naming convention."""
    assert path.suffix == ".h5", f"{path.name} is not an HDF5 file"

    match = PROBE_FILENAME_RE.match(path.stem)
    assert match is not None, f"{path.name} does not match <session_id>_p<idx>.h5"

    session_id = match.group("session_id")
    probe_idx = int(match.group("probe_idx"))
    assert probe_idx >= 0, f"{path.name} has a negative probe index"

    return session_id, probe_idx


def resolve_parent_h5_path(parent_dir: Path, session_id: str) -> Path:
    """Find the parent session-level HDF5 for one VBN session."""
    parent_dir = resolve_brainset_dir(parent_dir, DEFAULT_PARENT_BRAINSET_ID)
    h5_path = parent_dir / f"{session_id}.h5"
    assert h5_path.exists(), f"missing parent session HDF5: {h5_path}"
    return h5_path


def resolve_probe_file_groups(
    probe_dir: Path,
    expected_sessions: int | None = None,
    expected_probe_files: int | None = None,
    session_id: str | None = None,
) -> list[ProbeFileGroup]:
    """
    Discover and group all split probe HDF5 files by parent session.

    This validates naming, grouping, uniqueness, and optional full-dataset
    counts before any expensive content-level checks begin.
    """
    probe_dir = resolve_brainset_dir(probe_dir, DEFAULT_PROBE_BRAINSET_ID)
    assert probe_dir.exists(), f"probe directory does not exist: {probe_dir}"

    grouped_paths: dict[str, dict[int, Path]] = {}
    total_probe_files = 0

    for probe_path in sorted(probe_dir.glob("*.h5")):
        parent_session_id, probe_idx = parse_probe_filename(probe_path)
        if session_id is not None and parent_session_id != session_id:
            continue

        grouped_paths.setdefault(parent_session_id, {})
        assert probe_idx not in grouped_paths[parent_session_id], (
            f"duplicate split probe file for {parent_session_id}_p{probe_idx}"
        )
        grouped_paths[parent_session_id][probe_idx] = probe_path
        total_probe_files += 1

    assert grouped_paths, f"no split probe files found in {probe_dir}"

    groups: list[ProbeFileGroup] = []
    for parent_session_id in sorted(grouped_paths):
        probe_map = grouped_paths[parent_session_id]
        observed_indices = sorted(probe_map)
        expected_indices = list(range(len(observed_indices)))
        assert observed_indices == expected_indices, (
            f"non-contiguous probe indices for {parent_session_id}: "
            f"{observed_indices} != {expected_indices}"
        )
        groups.append(
            ProbeFileGroup(
                parent_session_id=parent_session_id,
                probe_paths=[probe_map[idx] for idx in observed_indices],
            )
        )

    if expected_sessions is not None:
        assert len(groups) == expected_sessions, (
            f"unexpected session count: {len(groups)} != {expected_sessions}"
        )
    if expected_probe_files is not None:
        assert total_probe_files == expected_probe_files, (
            f"unexpected probe file count: {total_probe_files} != {expected_probe_files}"
        )

    print_ok(
        f"Discovered {total_probe_files} split probe files across {len(groups)} sessions"
    )
    return groups


def load_metadata_csv(metadata_csv: Path) -> pd.DataFrame:
    """Load and sanity-check the metadata CSV generated from the split probe files."""
    assert metadata_csv.exists(), f"metadata CSV does not exist: {metadata_csv}"

    metadata_df = pd.read_csv(metadata_csv)
    missing = REQUIRED_METADATA_COLUMNS - set(metadata_df.columns)
    assert not missing, f"metadata CSV is missing required columns: {sorted(missing)}"

    metadata_df["id"] = metadata_df["id"].astype(str)
    metadata_df["session_id"] = metadata_df["session_id"].astype(str)
    metadata_df["subject_id"] = metadata_df["subject_id"].astype(str)

    assert not metadata_df["id"].duplicated().any(), "metadata CSV has duplicate unit ids"

    print_ok(
        f"Loaded metadata CSV with {len(metadata_df)} rows and unique unit ids"
    )
    return metadata_df


def validate_probe_hdf5_layout(probe_path: Path, data: Data) -> None:
    """Verify a split probe HDF5 is structurally valid on its own."""
    with h5py.File(probe_path, "r") as f:
        missing = REQUIRED_TOP_LEVEL_GROUPS - set(f.keys())
        assert not missing, f"{probe_path.name} missing top-level groups: {sorted(missing)}"

        missing_spikes = REQUIRED_SPIKE_FIELDS - set(f["spikes"].keys())
        assert not missing_spikes, (
            f"{probe_path.name} missing spike datasets: {sorted(missing_spikes)}"
        )

        missing_units = REQUIRED_UNIT_FIELDS - set(f["units"].keys())
        assert not missing_units, (
            f"{probe_path.name} missing unit datasets: {sorted(missing_units)}"
        )

    expected_session_id, _ = parse_probe_filename(probe_path)
    assert data.brainset.id == DEFAULT_PROBE_BRAINSET_ID, (
        f"{probe_path.name} has wrong brainset id: {data.brainset.id}"
    )
    assert data.session.id == probe_path.stem, (
        f"{probe_path.name} session id mismatch: {data.session.id} != {probe_path.stem}"
    )
    assert data.session.id.startswith(expected_session_id), (
        f"{probe_path.name} session id does not match parent session"
    )
    assert len(data.units.id) > 0, f"{probe_path.name} has no units"
    assert len(data.spikes.timestamps) > 0, f"{probe_path.name} has no spikes"
    assert len(data.spikes.timestamps) == len(data.spikes.unit_index), (
        f"{probe_path.name} spike timestamps/unit_index length mismatch"
    )
    assert len(data.intervals.keys()) > 0, f"{probe_path.name} has no intervals"


def validate_single_probe_identity(
    probe_path: Path,
    parent_data: Data,
    probe_data: Data,
) -> tuple[int, Any]:
    """
    Verify the probe file contains exactly one biological probe and that its
    filename index matches the probe ordering used by `split_probes.py`.
    """
    _, probe_idx = parse_probe_filename(probe_path)
    parent_probe_ids = np.unique(as_array(parent_data.units.probe_id))
    assert probe_idx < len(parent_probe_ids), (
        f"{probe_path.name} probe index {probe_idx} exceeds parent probe count"
    )

    expected_probe_id = parent_probe_ids[probe_idx]
    observed_probe_ids = np.unique(as_array(probe_data.units.probe_id))
    assert len(observed_probe_ids) == 1, (
        f"{probe_path.name} contains multiple probe ids: {observed_probe_ids}"
    )
    assert_arrays_equal(
        np.array([expected_probe_id]),
        observed_probe_ids,
        f"{probe_path.name} probe id",
    )

    return probe_idx, expected_probe_id


def validate_units_against_parent(
    parent_data: Data,
    probe_data: Data,
    probe_id: Any,
) -> np.ndarray:
    """
    Verify the split file's units are exactly the subset of parent units for one probe.

    Returns the parent unit indices for the saved probe units so the spike check
    can reuse the same mapping.
    """
    parent_probe_ids = as_array(parent_data.units.probe_id)
    parent_unit_mask = parent_probe_ids == probe_id
    parent_unit_indices = np.flatnonzero(parent_unit_mask)
    assert len(parent_unit_indices) == len(probe_data.units.id), (
        "parent/probe unit counts disagree"
    )

    parent_unit_keys = set(parent_data.units.keys())
    probe_unit_keys = set(probe_data.units.keys())
    assert parent_unit_keys == probe_unit_keys, "unit fields changed during probe split"

    for key in sorted(parent_unit_keys):
        parent_values = as_array(getattr(parent_data.units, key))[parent_unit_mask]
        probe_values = as_array(getattr(probe_data.units, key))
        assert_arrays_equal(parent_values, probe_values, f"units.{key}")

    return parent_unit_indices


def validate_spikes_against_parent(
    parent_data: Data,
    probe_data: Data,
    parent_unit_indices: np.ndarray,
) -> None:
    """Verify the split file's spikes are the expected subset/remapping of parent spikes."""
    parent_timestamps = as_float_array(parent_data.spikes.timestamps)
    parent_unit_index = as_array(parent_data.spikes.unit_index).astype(int)

    probe_timestamps = as_float_array(probe_data.spikes.timestamps)
    probe_unit_index = as_array(probe_data.spikes.unit_index).astype(int)

    spike_mask = np.isin(parent_unit_index, parent_unit_indices)
    expected_timestamps = parent_timestamps[spike_mask]

    remap = np.full(len(parent_data.units.id), fill_value=-1, dtype=int)
    remap[parent_unit_indices] = np.arange(len(parent_unit_indices))
    expected_unit_index = remap[parent_unit_index[spike_mask]]

    assert len(probe_timestamps) > 0, "split probe file has no spikes"
    assert np.all(np.diff(probe_timestamps) >= 0), "split probe spike timestamps are not sorted"
    assert_arrays_equal(expected_timestamps, probe_timestamps, "spike timestamps")
    assert_arrays_equal(expected_unit_index, probe_unit_index, "spike unit_index")

    assert probe_unit_index.min() >= 0, "split probe unit_index contains negative values"
    assert probe_unit_index.max() < len(probe_data.units.id), (
        "split probe unit_index exceeds local unit count"
    )
    assert np.array_equal(np.unique(probe_unit_index), np.arange(len(probe_data.units.id))), (
        "split probe unit_index is not contiguous"
    )


def timestamps_within_domain(
    timestamps: np.ndarray,
    domain_start: np.ndarray,
    domain_end: np.ndarray,
) -> np.ndarray:
    """Return a boolean mask telling whether each timestamp lies inside the domain union."""
    interval_idx = np.searchsorted(domain_start, timestamps, side="right") - 1
    valid = interval_idx >= 0
    valid_indices = interval_idx[valid]
    valid[valid] = timestamps[valid] <= domain_end[valid_indices]
    return valid


def validate_domain_after_split(probe_data: Data) -> None:
    """Verify the split probe domain is internally consistent."""
    data_start, data_end = as_float_array(probe_data.domain.start), as_float_array(
        probe_data.domain.end
    )
    spike_start, spike_end = as_float_array(
        probe_data.spikes.domain.start
    ), as_float_array(probe_data.spikes.domain.end)

    assert_arrays_equal(data_start, spike_start, "data.domain.start")
    assert_arrays_equal(data_end, spike_end, "data.domain.end")

    assert len(data_start) > 0, "split probe domain is empty"
    assert len(data_start) == len(data_end), "split probe domain start/end mismatch"
    assert np.all(data_end > data_start), "split probe domain has non-positive duration"
    assert np.all(np.diff(data_start) >= 0), "split probe domain starts are not sorted"
    if len(data_start) > 1:
        assert np.all(data_start[1:] >= data_end[:-1]), (
            "split probe domain intervals overlap"
        )

    probe_timestamps = as_float_array(probe_data.spikes.timestamps)
    in_domain = timestamps_within_domain(probe_timestamps, data_start, data_end)
    if not np.all(in_domain):
        outside_timestamps = probe_timestamps[~in_domain]
        raise AssertionError(
            "some split probe spikes lie outside the saved domain: "
            f"{len(outside_timestamps)} spikes outside support, "
            f"first outside spike={outside_timestamps[0]:.6f}, "
            f"domain starts at {data_start[0]:.6f}"
        )


def validate_intervals_preserved(parent_data: Data, probe_data: Data) -> None:
    """Verify that task intervals were preserved exactly by `split_probes.py`."""
    parent_interval_names = set(parent_data.intervals.keys())
    probe_interval_names = set(probe_data.intervals.keys())
    assert parent_interval_names == probe_interval_names, (
        "interval groups changed during probe split"
    )

    for interval_name in sorted(parent_interval_names):
        parent_interval = getattr(parent_data.intervals, interval_name)
        probe_interval = getattr(probe_data.intervals, interval_name)

        parent_keys = set(parent_interval.keys())
        probe_keys = set(probe_interval.keys())
        assert parent_keys == probe_keys, f"{interval_name} fields changed during probe split"

        for key in sorted(parent_keys):
            parent_values = getattr(parent_interval, key)
            probe_values = getattr(probe_interval, key)
            assert_arrays_equal(
                parent_values,
                probe_values,
                f"intervals.{interval_name}.{key}",
            )


def validate_probe_file_against_parent(
    probe_path: Path,
    parent_data: Data,
    probe_data: Data,
) -> Any:
    """Run the full parent-vs-probe validation for one split HDF5 file."""
    validate_probe_hdf5_layout(probe_path, probe_data)
    _, probe_id = validate_single_probe_identity(probe_path, parent_data, probe_data)
    parent_unit_indices = validate_units_against_parent(parent_data, probe_data, probe_id)
    validate_spikes_against_parent(parent_data, probe_data, parent_unit_indices)
    validate_domain_after_split(probe_data)
    validate_intervals_preserved(parent_data, probe_data)
    return probe_id


def validate_metadata_row_coverage(
    metadata_df: pd.DataFrame,
    probe_data: Data,
    probe_session_id: str,
    probe_id: Any,
) -> None:
    """Verify metadata rows match one split probe HDF5 exactly."""
    probe_units_df = arraydict_to_dataframe(probe_data.units).copy()
    probe_units_df["id"] = probe_units_df["id"].astype(str)
    probe_units_df["session_id"] = str(probe_session_id)
    probe_units_df["subject_id"] = str(probe_data.subject.id)

    metadata_subset = metadata_df[metadata_df["session_id"] == str(probe_session_id)].copy()
    assert len(metadata_subset) == len(probe_units_df), (
        f"metadata row count mismatch for {probe_session_id}"
    )

    probe_unit_ids = set(probe_units_df["id"])
    metadata_unit_ids = set(metadata_subset["id"])
    assert probe_unit_ids == metadata_unit_ids, (
        f"metadata unit ids do not match HDF5 units for {probe_session_id}"
    )

    metadata_subset["subject_id"] = metadata_subset["subject_id"].astype(str)
    metadata_subset = metadata_subset.set_index("id").sort_index()
    probe_units_df = probe_units_df.set_index("id").sort_index()

    assert np.all(metadata_subset["session_id"] == str(probe_session_id)), (
        f"metadata session ids are wrong for {probe_session_id}"
    )
    assert np.all(as_str_array(metadata_subset["probe_id"]) == str(probe_id)), (
        f"metadata probe ids are wrong for {probe_session_id}"
    )
    assert np.all(metadata_subset["subject_id"] == str(probe_data.subject.id)), (
        f"metadata subject ids are wrong for {probe_session_id}"
    )
    assert np.array_equal(
        metadata_subset["structure_acronym"].astype(str).values,
        probe_units_df["structure_acronym"].astype(str).values,
    ), f"metadata structure_acronym mismatch for {probe_session_id}"


def validate_metadata_global_consistency(
    metadata_df: pd.DataFrame,
    probe_groups: list[ProbeFileGroup],
) -> None:
    """Verify the metadata CSV matches the full split-probe dataset."""
    all_unit_ids: set[str] = set()
    all_session_ids: set[str] = set()

    for probe_group in probe_groups:
        for probe_path in probe_group.probe_paths:
            with h5py.File(probe_path, "r") as f:
                probe_data = Data.from_hdf5(f, lazy=True)
                all_unit_ids.update(as_str_array(probe_data.units.id))
                all_session_ids.add(str(probe_data.session.id))

    metadata_unit_ids = set(metadata_df["id"].astype(str))
    metadata_session_ids = set(metadata_df["session_id"].astype(str))

    assert metadata_unit_ids == all_unit_ids, "metadata CSV unit ids do not match probe HDF5s"
    assert metadata_session_ids == all_session_ids, (
        "metadata CSV session ids do not match probe HDF5s"
    )
    assert not metadata_df["id"].duplicated().any(), "metadata CSV has duplicate unit ids"

    print_ok("Metadata CSV matches the split probe dataset globally")


def validate_dataset_readback(
    processed_root: Path,
    probe_session_id: str,
    *,
    brainset_id: str = DEFAULT_PROBE_BRAINSET_ID,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> None:
    """Perform one downstream smoke test through `torch_brain.Dataset`."""
    recording_id = f"{brainset_id}/{probe_session_id}"
    dataset = Dataset(root=str(processed_root), recording_id=recording_id)
    intervals = dataset.get_sampling_intervals()[recording_id]

    start = float(intervals.start[0])
    end = start + window_seconds
    assert end <= float(intervals.end[0]), (
        f"sampling domain too short for test window in {recording_id}"
    )

    sample = dataset.get(recording_id, start, end)
    assert sample.session.id == recording_id
    assert len(sample.units.id) > 0, f"{recording_id} sample has no units"

    if len(sample.spikes.timestamps) > 0:
        assert float(sample.spikes.timestamps.min()) >= 0.0
        assert float(sample.spikes.timestamps.max()) <= window_seconds
        assert int(sample.spikes.unit_index.min()) >= 0
        assert int(sample.spikes.unit_index.max()) < len(sample.units.id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Allen VBN split-probe HDF5 files and metadata."
    )
    parser.add_argument(
        "--parent-dir",
        type=Path,
        default=default_parent_dir(),
        help="Directory containing parent session-level HDF5 files.",
    )
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=default_probe_dir(),
        help="Directory containing split probe-level HDF5 files.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=default_metadata_csv(),
        help="CSV generated by utils/neuron_md_gen.py.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional parent session id to validate instead of the full dataset.",
    )
    parser.add_argument(
        "--expected-sessions",
        type=int,
        default=None,
        help="Optional full-dataset check for number of parent sessions.",
    )
    parser.add_argument(
        "--expected-probe-files",
        type=int,
        default=None,
        help="Optional full-dataset check for total number of split probe files.",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Window size for downstream Dataset sampling check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parent_dir = resolve_brainset_dir(args.parent_dir, DEFAULT_PARENT_BRAINSET_ID)
    probe_dir = resolve_brainset_dir(args.probe_dir, DEFAULT_PROBE_BRAINSET_ID)
    processed_root = resolve_processed_root(args.probe_dir, DEFAULT_PROBE_BRAINSET_ID)

    # 1. Discover all split probe HDF5 files and group them by parent session id.
    probe_groups = resolve_probe_file_groups(
        probe_dir=probe_dir,
        expected_sessions=args.expected_sessions,
        expected_probe_files=args.expected_probe_files,
        session_id=args.session_id,
    )

    # 2. Load the metadata CSV and validate its top-level structure.
    metadata_df = load_metadata_csv(args.metadata_csv)

    # 3. For each parent session, verify:
    #    - the number of split probe files matches the parent session's probe count
    #    - each probe file is a correct subset/remapping of the parent HDF5
    #    - metadata rows match the probe HDF5 contents exactly
    #    - downstream `torch_brain.Dataset` readback works
    for probe_group in probe_groups:
        parent_path = resolve_parent_h5_path(parent_dir, probe_group.parent_session_id)
        with h5py.File(parent_path, "r") as f:
            parent_data = Data.from_hdf5(f, lazy=False)

        parent_probe_count = len(np.unique(as_array(parent_data.units.probe_id)))
        assert len(probe_group.probe_paths) == parent_probe_count, (
            f"split probe count mismatch for parent session {probe_group.parent_session_id}: "
            f"{len(probe_group.probe_paths)} != {parent_probe_count}"
        )

        for probe_path in probe_group.probe_paths:
            with h5py.File(probe_path, "r") as f:
                probe_data = Data.from_hdf5(f, lazy=False)

            probe_id = validate_probe_file_against_parent(
                probe_path=probe_path,
                parent_data=parent_data,
                probe_data=probe_data,
            )
            validate_metadata_row_coverage(
                metadata_df=metadata_df,
                probe_data=probe_data,
                probe_session_id=str(probe_data.session.id),
                probe_id=probe_id,
            )
            validate_dataset_readback(
                processed_root=processed_root,
                probe_session_id=str(probe_data.session.id),
                window_seconds=args.window,
            )

        print_ok(
            f"Parent session {probe_group.parent_session_id} split correctly into "
            f"{len(probe_group.probe_paths)} probe files"
        )

    # 4. After per-file checks, validate the metadata CSV globally against all
    #    split probe HDF5 files.
    validate_metadata_global_consistency(metadata_df, probe_groups)

    print_ok("All Visual Behavior split-probe checks passed")


if __name__ == "__main__":
    main()
