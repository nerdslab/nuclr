"""
Allen VBN optotagging validation

To be ran after generating Allen VBN 2022 session-level HDF5 files
and the neuron metadata CSV. 

Test suite covers:
-> genotype labels
-> PSTH binning
-> saved unit labels
-> global label counts
-> metadata propagation

"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import pandas as pd
from temporaldata import Data


DEFAULT_BRAINSET_ID = "allen_vbn_2022"
DEFAULT_METADATA_CSV = "allen_vbn_2022.csv"
ALLOWED_CELL_TYPES = {"wt", "Sst", "Vip"}
NON_WT_CELL_TYPES = {"Sst", "Vip"}
REQUIRED_OPTO_FIELDS = {
    "optotagging_baseline_rate",
    "optotagging_evoked_rate",
    "optotagging_response_ratio",
    "optotagging_trial_reliability",
    "optotagging_first_spike_latency_ms",
    "optotagging_cre_positive",
    "optotagged_cell_type",
}
REQUIRED_UNIT_FIELDS = REQUIRED_OPTO_FIELDS | {"cre_positive"}
REQUIRED_METADATA_FIELDS = REQUIRED_OPTO_FIELDS | {"id"}
DEFAULT_MAX_POSITIVE_FRACTION = 0.75


def repo_root() -> Path:
    """Return the repository root independent of the current working directory."""
    return Path(__file__).resolve().parents[2]


PIPELINE_DIR = repo_root() / "preprocess" / "allen_vbn_2022_vis"
sys.path.insert(0, str(PIPELINE_DIR))

from session_extractor import (  # noqa: E402
    add_cell_types,
    make_population_psth,
    standardize_genotype,
)


@dataclass
class OptotaggingSummary:
    """Aggregated optotagging counts from processed session HDF5 files."""

    session_count: int = 0
    unit_count: int = 0
    genotype_session_counts: Counter[str] = field(default_factory=Counter)
    genotype_unit_counts: Counter[str] = field(default_factory=Counter)
    cell_type_counts: Counter[str] = field(default_factory=Counter)
    genotype_cell_type_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )


@dataclass
class ValidationFailure:
    """One validation failure collected during a batch test run."""

    section: str
    message: str
    path: Path | None = None
    details: str | None = None

    def format(self) -> str:
        location = f"{self.path.name}: " if self.path is not None else ""
        detail = f"\n    {self.details}" if self.details else ""
        return f"[{self.section}] {location}{self.message}{detail}"


def print_ok(message: str) -> None:
    print(f"[ok] {message}")


def record_failure(
    failures: list[ValidationFailure],
    section: str,
    message: str,
    *,
    path: Path | None = None,
    details: str | None = None,
) -> None:
    """Append one failure to the run-level failure list."""
    failures.append(
        ValidationFailure(section=section, message=message, path=path, details=details)
    )


def check(
    condition: bool,
    failures: list[ValidationFailure],
    section: str,
    message: str,
    *,
    path: Path | None = None,
    details: str | None = None,
) -> bool:
    """Record a failed condition without stopping the rest of the test run."""
    if condition:
        return True

    record_failure(
        failures,
        section,
        message,
        path=path,
        details=details,
    )
    return False


def summarize_numeric_array(
    values: np.ndarray,
    unit_ids: np.ndarray,
    *,
    max_examples: int = 5,
) -> str:
    """Return compact diagnostics for a numeric unit-level field."""
    values = np.asarray(values)
    finite = np.isfinite(values)
    nonfinite_idx = np.flatnonzero(~finite)
    examples = [
        f"{unit_ids[idx]}={values[idx]!r}"
        for idx in nonfinite_idx[:max_examples]
    ]

    parts = [
        f"finite={int(finite.sum())}/{values.size}",
        f"nan={int(np.isnan(values).sum())}",
        f"+inf={int(np.isposinf(values).sum())}",
        f"-inf={int(np.isneginf(values).sum())}",
    ]
    if examples:
        parts.append(f"examples: {', '.join(examples)}")

    return "; ".join(parts)


def print_failure_summary(failures: list[ValidationFailure]) -> None:
    """Print all collected failures in a stable, scan-friendly summary."""
    if not failures:
        return

    print("\nAllen VBN optotagging validation failed")
    print(f"Collected {len(failures)} failure(s):")

    grouped = Counter((failure.section, failure.message) for failure in failures)
    print("Failure types:")
    for (section, message), count in grouped.most_common():
        print(f"  - {section}: {message} ({count})")

    for idx, failure in enumerate(failures, start=1):
        print(f"{idx}. {failure.format()}")


def default_processed_dir() -> Path:
    """
    Resolve the default processed-data root.

    Support both repository-local `data/processed` and the sibling layout used
    by the other Allen VBN validation scripts.
    """
    candidates = [
        repo_root() / "data" / "processed",
        repo_root().parent / "data" / "processed",
    ]

    for candidate in candidates:
        brainset_dir = candidate / DEFAULT_BRAINSET_ID
        if brainset_dir.exists() and any(brainset_dir.glob("*.h5")):
            return candidate
        if candidate.exists() and any(candidate.glob("*.h5")):
            return candidate

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def default_metadata_csv() -> Path:
    """Default metadata CSV generated by `utils/neuron_md_gen.py`."""
    return repo_root() / "neuron_metadata" / DEFAULT_METADATA_CSV


def as_array(value: Any) -> np.ndarray:
    """Materialize lazy array-like values into a NumPy array."""
    if hasattr(value, "__array__") and not isinstance(value, np.ndarray):
        value = value[:]
    return np.asarray(value)


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


def as_bool_array(value: Any) -> np.ndarray:
    """Normalize bool-like arrays from HDF5 or CSV into real booleans."""
    arr = as_array(value)

    if arr.dtype.kind == "b":
        return arr.astype(bool)

    true_values = {"true", "1", "yes"}
    false_values = {"false", "0", "no"}
    normalized = np.array([str(x).strip().lower() for x in arr])
    valid = np.isin(normalized, list(true_values | false_values))
    assert np.all(valid), f"cannot parse boolean values: {np.unique(normalized[~valid])}"

    return np.isin(normalized, list(true_values))


def scalar_to_str(value: Any) -> str:
    """Convert scalar HDF5/Data values into a plain Python string."""
    if isinstance(value, bytes):
        return value.decode("utf-8")

    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)

    if arr.size == 1:
        item = arr.reshape(-1)[0]
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)

    return str(value)


def resolve_brainset_dir(processed_dir: Path, brainset_id: str) -> Path:
    """
    Resolve a processed brainset directory from either a root or direct path.
    """
    if processed_dir.suffix == ".h5":
        return processed_dir.parent

    nested = processed_dir / brainset_id
    if nested.exists():
        return nested

    return processed_dir


def resolve_h5_paths(
    processed_dir: Path,
    brainset_id: str,
    session_id: str | None,
) -> list[Path]:
    """
    Resolve session HDF5 files for validation.

    Supports a processed root, a brainset-specific directory, or a direct HDF5
    path. If `session_id` is provided, only that session is selected.
    """
    if processed_dir.suffix == ".h5":
        assert processed_dir.exists(), f"processed HDF5 does not exist: {processed_dir}"
        return [processed_dir]

    if session_id is not None:
        direct = processed_dir / f"{session_id}.h5"
        nested = processed_dir / brainset_id / f"{session_id}.h5"

        if nested.exists():
            return [nested]
        if direct.exists():
            return [direct]

        raise FileNotFoundError(
            "Could not find processed HDF5 at either "
            f"{nested} or {direct}. Pass --processed-dir as the processed root "
            "or as the brainset-specific directory."
        )

    brainset_dir = resolve_brainset_dir(processed_dir, brainset_id)
    assert brainset_dir.exists(), f"processed brainset directory does not exist: {brainset_dir}"

    h5_paths = sorted(brainset_dir.glob("*.h5"))
    assert h5_paths, f"no HDF5 files found in {brainset_dir}"

    return h5_paths


@contextmanager
def open_data(path: Path, *, lazy: bool = True) -> Iterator[Data]:
    """Open one processed HDF5 as a temporaldata `Data` object."""
    with h5py.File(path, "r") as f:
        yield Data.from_hdf5(f, lazy=lazy)


def validate_unit_field_layout(path: Path) -> None:
    """Check optotagging unit fields exist before loading through Data."""
    with h5py.File(path, "r") as f:
        assert "units" in f, f"{path.name} is missing the units group"
        missing = REQUIRED_UNIT_FIELDS - set(f["units"].keys())
        assert not missing, f"{path.name} missing unit fields: {sorted(missing)}"


def test_standardize_genotype_labels() -> None:
    """Validate raw genotype strings map to the pipeline label vocabulary."""
    examples = {
        "Sst-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt": "Sst",
        "sst-cre example": "Sst",
        "Vip-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt": "Vip",
        "VIP-cre example": "Vip",
        "wt/wt": "wt",
        "Slc17a7-IRES2-Cre": "wt",
        "": "wt",
        None: "wt",
    }

    for raw_genotype, expected in examples.items():
        observed = standardize_genotype(raw_genotype)
        assert observed == expected, (
            f"standardize_genotype({raw_genotype!r}) returned "
            f"{observed!r}, expected {expected!r}"
        )

    print_ok("Genotype strings normalize to wt/Sst/Vip")


def test_make_population_psth_shape_and_binning() -> None:
    """Validate optotagging PSTH shape and pulse-relative spike binning."""
    spike_times_by_unit = {
        101: np.array([0.95, 1.02, 1.11, 1.25, 2.05, 2.15]),
        202: np.array([1.05, 1.19, 2.01, 2.22]),
    }
    unit_ids = [101, 202, 303]
    start_times = np.array([1.0, 2.0])

    psth, time_bins, returned_unit_ids = make_population_psth(
        spike_times_by_unit=spike_times_by_unit,
        unit_ids=unit_ids,
        start_times=start_times,
        time_before=0.1,
        duration=0.3,
        bin_size=0.1,
    )

    expected_time_bins = np.array([-0.1, 0.0, 0.1])
    expected = np.array(
        [
            [[10.0, 0.0], [10.0, 10.0], [10.0, 10.0]],
            [[0.0, 0.0], [10.0, 10.0], [10.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )

    assert psth.shape == (3, 3, 2), f"unexpected PSTH shape: {psth.shape}"
    assert returned_unit_ids == unit_ids, "returned unit order changed"
    assert np.allclose(time_bins, expected_time_bins), "time bin left edges changed"
    assert np.allclose(psth, expected), "pulse-relative spike binning changed"

    mean_psth, mean_time_bins, mean_unit_ids = make_population_psth(
        spike_times_by_unit=spike_times_by_unit,
        unit_ids=unit_ids,
        start_times=start_times,
        time_before=0.1,
        duration=0.3,
        bin_size=0.1,
        mean_over_trials=True,
    )

    assert mean_psth.shape == (3, 3), f"unexpected mean PSTH shape: {mean_psth.shape}"
    assert mean_unit_ids == unit_ids, "mean PSTH unit order changed"
    assert np.allclose(mean_time_bins, expected_time_bins), "mean PSTH bins changed"
    assert np.allclose(mean_psth, psth.mean(axis=2)), (
        "mean_over_trials output does not match the trial average"
    )

    print_ok("PSTH shape, binning, and trial averaging are correct")


def test_add_cell_types_uses_max_short_pulse_level() -> None:
    """Validate classification uses each session's strongest short pulse level."""
    units = pd.DataFrame(index=[101, 202])
    spike_times_by_unit = {
        101: np.array([1.002, 2.003]),
        202: np.array([10.002, 20.003]),
    }
    optotagging_table = pd.DataFrame(
        {
            "start_time": [1.0, 2.0, 10.0, 20.0, 30.0],
            "stop_time": [1.01, 2.01, 10.01, 20.01, 31.0],
            "duration": [0.01, 0.01, 0.01, 0.01, 1.0],
            "level": [1.35, 1.35, 0.97, 0.97, 1.7],
        }
    )
    tagging_config = {
        "time_before": 0.01,
        "duration": 0.03,
        "bin_size": 0.001,
        "increase_in_fr": 3.0,
        "min_evoked_rate": 50.0,
        "baseline_start_ms": -10.0,
        "baseline_end_ms": -2.0,
        "evoked_start_ms": 1.0,
        "evoked_end_ms": 9.0,
        "min_trial_reliability": 0.5,
        "max_first_spike_latency_ms": 9.0,
        "max_pulse_duration": 0.1,
    }

    classified_units = add_cell_types(
        units=units,
        spike_times_by_unit=spike_times_by_unit,
        optotagging_table=optotagging_table,
        tagging_config=tagging_config,
        genotype_label="Vip-IRES-Cre",
    )

    assert classified_units.loc[101, "optotagging_cre_positive"], (
        "unit responding to max short-pulse level was not tagged"
    )
    assert not classified_units.loc[202, "optotagging_cre_positive"], (
        "unit responding only to lower pulse levels was tagged"
    )
    assert classified_units.loc[101, "optotagged_cell_type"] == "Vip"
    assert classified_units.loc[202, "optotagged_cell_type"] == "wt"
    assert np.isfinite(classified_units["optotagging_baseline_rate"]).all()
    assert np.isfinite(classified_units["optotagging_evoked_rate"]).all()
    assert np.isfinite(classified_units["optotagging_response_ratio"]).all()

    print_ok("Cell-type classification uses the max short-pulse level")


def test_processed_h5_optotagging_unit_fields(
    h5_paths: list[Path],
) -> tuple[OptotaggingSummary, list[ValidationFailure]]:
    """Validate optotagging-specific unit fields in processed session HDF5s."""
    summary = OptotaggingSummary()
    failures: list[ValidationFailure] = []

    for path in h5_paths:
        try:
            validate_unit_field_layout(path)
        except AssertionError as exc:
            record_failure(
                failures,
                "processed HDF5 fields",
                str(exc),
                path=path,
            )
            continue
        except Exception as exc:
            record_failure(
                failures,
                "processed HDF5 fields",
                f"could not inspect unit field layout: {exc!r}",
                path=path,
            )
            continue

        try:
            with open_data(path, lazy=True) as data:
                unit_ids = as_str_array(data.units.id)
                optotagging_cre_positive = as_bool_array(
                    data.units.optotagging_cre_positive
                )
                cre_positive = as_bool_array(data.units.cre_positive)
                cell_types = as_str_array(data.units.optotagged_cell_type)
                baseline_rate = as_array(
                    data.units.optotagging_baseline_rate
                ).astype(float)
                evoked_rate = as_array(data.units.optotagging_evoked_rate).astype(
                    float
                )
                response_ratio = as_array(
                    data.units.optotagging_response_ratio
                ).astype(float)
                trial_reliability = as_array(
                    data.units.optotagging_trial_reliability
                ).astype(float)
                first_spike_latency_ms = as_array(
                    data.units.optotagging_first_spike_latency_ms
                ).astype(float)
                genotype = standardize_genotype(scalar_to_str(data.subject.genotype))
        except Exception as exc:
            record_failure(
                failures,
                "processed HDF5 fields",
                f"could not load optotagging fields: {exc!r}",
                path=path,
            )
            continue

        if not check(
            len(unit_ids) > 0,
            failures,
            "processed HDF5 fields",
            "has no saved units",
            path=path,
        ):
            continue

        field_lengths = {
            "cre_positive": len(cre_positive),
            "optotagging_cre_positive": len(optotagging_cre_positive),
            "optotagged_cell_type": len(cell_types),
            "optotagging_baseline_rate": len(baseline_rate),
            "optotagging_evoked_rate": len(evoked_rate),
            "optotagging_response_ratio": len(response_ratio),
            "optotagging_trial_reliability": len(trial_reliability),
            "optotagging_first_spike_latency_ms": len(first_spike_latency_ms),
        }
        lengths_ok = True
        for field_name, field_length in field_lengths.items():
            lengths_ok &= check(
                field_length == len(unit_ids),
                failures,
                "processed HDF5 fields",
                f"{field_name} length does not match units.id",
                path=path,
                details=f"{field_name}={field_length}, units.id={len(unit_ids)}",
            )
        if not lengths_ok:
            continue

        check(
            np.array_equal(cre_positive, optotagging_cre_positive),
            failures,
            "processed HDF5 fields",
            "cre_positive alias does not match optotagging_cre_positive",
            path=path,
        )

        baseline_finite = check(
            np.all(np.isfinite(baseline_rate)),
            failures,
            "processed HDF5 fields",
            "baseline rates contain non-finite values",
            path=path,
            details=summarize_numeric_array(baseline_rate, unit_ids),
        )
        evoked_finite = check(
            np.all(np.isfinite(evoked_rate)),
            failures,
            "processed HDF5 fields",
            "evoked rates contain non-finite values",
            path=path,
            details=summarize_numeric_array(evoked_rate, unit_ids),
        )
        ratio_finite = check(
            np.all(np.isfinite(response_ratio)),
            failures,
            "processed HDF5 fields",
            "response ratios contain non-finite values",
            path=path,
            details=summarize_numeric_array(response_ratio, unit_ids),
        )

        if baseline_finite:
            check(
                np.all(baseline_rate >= 0),
                failures,
                "processed HDF5 fields",
                "has negative baseline rates",
                path=path,
            )
        if evoked_finite:
            check(
                np.all(evoked_rate >= 0),
                failures,
                "processed HDF5 fields",
                "has negative evoked rates",
                path=path,
            )
        if ratio_finite:
            check(
                np.all(response_ratio >= 0),
                failures,
                "processed HDF5 fields",
                "has negative response ratios",
                path=path,
            )

        reliability_finite = check(
            np.all(np.isfinite(trial_reliability)),
            failures,
            "processed HDF5 fields",
            "reliability values contain non-finite values",
            path=path,
            details=summarize_numeric_array(trial_reliability, unit_ids),
        )
        if reliability_finite:
            check(
                np.all((trial_reliability >= 0) & (trial_reliability <= 1)),
                failures,
                "processed HDF5 fields",
                "reliability values are outside [0, 1]",
                path=path,
            )

        positive_latency = first_spike_latency_ms[cre_positive]
        positive_unit_ids = unit_ids[cre_positive]
        check(
            np.all(np.isfinite(positive_latency)),
            failures,
            "processed HDF5 fields",
            "positive units have missing first-spike latencies",
            path=path,
            details=summarize_numeric_array(positive_latency, positive_unit_ids),
        )

        observed_labels = set(cell_types)
        unexpected_labels = observed_labels - ALLOWED_CELL_TYPES
        check(
            not unexpected_labels,
            failures,
            "processed HDF5 fields",
            f"has unexpected optotagged labels: {sorted(unexpected_labels)}",
            path=path,
        )

        allowed_for_genotype = {"wt"} if genotype == "wt" else {"wt", genotype}
        incompatible_labels = observed_labels - allowed_for_genotype
        check(
            not incompatible_labels,
            failures,
            "processed HDF5 fields",
            f"has labels incompatible with genotype {genotype}: "
            f"{sorted(incompatible_labels)}",
            path=path,
        )
        check(
            not np.any(cre_positive) or genotype != "wt",
            failures,
            "processed HDF5 fields",
            "has optotagged Cre+ units in a wt session",
            path=path,
        )

        expected_cell_types = np.full(len(unit_ids), "wt", dtype=object)
        expected_cell_types[cre_positive] = genotype
        check(
            np.array_equal(cell_types, expected_cell_types),
            failures,
            "processed HDF5 fields",
            "optotagged_cell_type does not match optotagging_cre_positive",
            path=path,
        )

        summary.session_count += 1
        summary.unit_count += len(unit_ids)
        summary.genotype_session_counts[genotype] += 1
        summary.genotype_unit_counts[genotype] += len(unit_ids)
        summary.cell_type_counts.update(cell_types)
        summary.genotype_cell_type_counts[genotype].update(cell_types)

    if failures:
        print(
            f"[fail] Validated optotagging unit fields in {summary.session_count} "
            f"HDF5 files ({summary.unit_count:,} units) with "
            f"{len(failures)} failure(s)"
        )
    else:
        print_ok(
            f"Validated optotagging unit fields in {summary.session_count} HDF5 files "
            f"({summary.unit_count:,} units)"
        )
    return summary, failures


def test_dataset_level_optotagging_counts(
    summary: OptotaggingSummary,
    *,
    max_positive_fraction: float,
    require_non_wt: bool,
) -> None:
    """Validate broad global counts for genotype and optotagged cell-type labels."""
    non_wt_genotypes = sorted(
        genotype
        for genotype, count in summary.genotype_session_counts.items()
        if genotype in NON_WT_CELL_TYPES and count > 0
    )

    if require_non_wt:
        assert non_wt_genotypes, "no non-wt genotype sessions were selected"

    for genotype in non_wt_genotypes:
        genotype_units = summary.genotype_unit_counts[genotype]
        positive_units = summary.genotype_cell_type_counts[genotype][genotype]
        positive_fraction = positive_units / genotype_units

        assert positive_units > 0, (
            f"{genotype} sessions produced zero {genotype} optotagged units"
        )
        assert positive_fraction <= max_positive_fraction, (
            f"{genotype} optotagged fraction is too high: "
            f"{positive_fraction:.3f} > {max_positive_fraction:.3f}"
        )

    if non_wt_genotypes:
        counts = ", ".join(
            f"{label}={summary.cell_type_counts[label]:,}"
            for label in ["wt", "Sst", "Vip"]
        )
        print_ok(f"Global optotagging counts are within sanity bounds ({counts})")
    else:
        print_ok("No non-wt genotype sessions selected; global count check skipped")


def load_metadata_csv(metadata_csv: Path) -> pd.DataFrame:
    """Load generated neuron metadata and verify required columns exist."""
    assert metadata_csv.exists(), f"metadata CSV does not exist: {metadata_csv}"

    metadata = pd.read_csv(metadata_csv)
    missing = REQUIRED_METADATA_FIELDS - set(metadata.columns)
    assert not missing, f"metadata CSV missing fields: {sorted(missing)}"

    metadata["id"] = metadata["id"].astype(str)
    assert not metadata["id"].duplicated().any(), "metadata CSV has duplicate unit IDs"

    return metadata


def collect_h5_unit_labels(
    h5_paths: list[Path],
) -> tuple[pd.DataFrame, bool]:
    """Collect optotagging unit labels from selected session HDF5 files."""
    rows = []
    has_cre_positive = False

    for path in h5_paths:
        with open_data(path, lazy=True) as data:
            unit_ids = as_str_array(data.units.id)
            session_id = scalar_to_str(data.session.id)
            subject_id = scalar_to_str(data.subject.id)

            row_data = {
                "id": unit_ids,
                "session_id": np.array([session_id] * len(unit_ids)),
                "subject_id": np.array([subject_id] * len(unit_ids)),
            }
            for field_name in REQUIRED_OPTO_FIELDS:
                value = getattr(data.units, field_name)
                if field_name == "optotagged_cell_type":
                    row_data[field_name] = as_str_array(value)
                else:
                    row_data[field_name] = as_array(value)

            if "cre_positive" in set(data.units.keys()):
                has_cre_positive = True
                row_data["cre_positive"] = as_bool_array(data.units.cre_positive)

            rows.append(pd.DataFrame(row_data))

    return pd.concat(rows, ignore_index=True), has_cre_positive


def test_metadata_preserves_optotagging_fields(
    h5_paths: list[Path],
    metadata_csv: Path,
) -> None:
    """Validate generated metadata carries optotagging labels downstream."""
    metadata = load_metadata_csv(metadata_csv)
    h5_labels, h5_has_cre_positive = collect_h5_unit_labels(h5_paths)

    if h5_has_cre_positive:
        assert "cre_positive" in metadata.columns, (
            "metadata CSV is missing cre_positive even though HDF5 units contain it"
        )

    metadata_subset = metadata[metadata["id"].isin(h5_labels["id"])].copy()
    assert len(metadata_subset) == len(h5_labels), (
        "metadata CSV does not contain one row for each selected HDF5 unit"
    )

    metadata_subset = metadata_subset.set_index("id").sort_index()
    h5_labels = h5_labels.set_index("id").sort_index()

    for field_name in sorted(REQUIRED_OPTO_FIELDS):
        if field_name == "optotagged_cell_type":
            assert np.array_equal(
                metadata_subset[field_name].astype(str).values,
                h5_labels[field_name].astype(str).values,
            ), f"metadata {field_name} values do not match HDF5 labels"
        elif field_name == "optotagging_cre_positive":
            metadata_values = as_bool_array(metadata_subset[field_name].values)
            h5_values = as_bool_array(h5_labels[field_name].values)
            assert np.array_equal(metadata_values, h5_values), (
                f"metadata {field_name} values do not match HDF5 labels"
            )
        else:
            metadata_values = pd.to_numeric(metadata_subset[field_name]).to_numpy()
            h5_values = pd.to_numeric(h5_labels[field_name]).to_numpy()
            assert np.allclose(metadata_values, h5_values, equal_nan=True), (
                f"metadata {field_name} values do not match HDF5 labels"
            )

    if h5_has_cre_positive:
        metadata_cre_positive = as_bool_array(metadata_subset["cre_positive"].values)
        h5_cre_positive = as_bool_array(h5_labels["cre_positive"].values)
        assert np.array_equal(metadata_cre_positive, h5_cre_positive), (
            "metadata cre_positive values do not match HDF5 labels"
        )

    print_ok(
        f"Metadata optotagging fields match {len(h5_labels):,} selected HDF5 units"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Allen VBN optotagging labels and metadata."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed_dir(),
        help=(
            "Processed-data root, brainset-specific directory, or one session HDF5."
        ),
    )
    parser.add_argument("--brainset-id", default=DEFAULT_BRAINSET_ID)
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session id to validate instead of all processed HDF5s.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=default_metadata_csv(),
        help="Neuron metadata CSV generated by utils/neuron_md_gen.py.",
    )
    parser.add_argument(
        "--max-positive-fraction",
        type=float,
        default=DEFAULT_MAX_POSITIVE_FRACTION,
        help="Maximum allowed optotagged fraction within each non-wt genotype.",
    )
    parser.add_argument(
        "--allow-no-non-wt",
        action="store_true",
        help="Allow selected HDF5s to contain no Sst/Vip genotype sessions.",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip metadata CSV validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_paths = resolve_h5_paths(
        processed_dir=args.processed_dir,
        brainset_id=args.brainset_id,
        session_id=args.session_id,
    )

    print(f"Validating {len(h5_paths)} Allen VBN optotagging HDF5 file(s)")

    failures: list[ValidationFailure] = []

    try:
        test_standardize_genotype_labels()
    except AssertionError as exc:
        record_failure(failures, "genotype labels", str(exc))
    except Exception as exc:
        record_failure(failures, "genotype labels", f"unexpected error: {exc!r}")

    try:
        test_make_population_psth_shape_and_binning()
    except AssertionError as exc:
        record_failure(failures, "PSTH binning", str(exc))
    except Exception as exc:
        record_failure(failures, "PSTH binning", f"unexpected error: {exc!r}")

    try:
        test_add_cell_types_uses_max_short_pulse_level()
    except AssertionError as exc:
        record_failure(failures, "max pulse level selection", str(exc))
    except Exception as exc:
        record_failure(
            failures,
            "max pulse level selection",
            f"unexpected error: {exc!r}",
        )

    summary, h5_failures = test_processed_h5_optotagging_unit_fields(h5_paths)
    failures.extend(h5_failures)

    try:
        test_dataset_level_optotagging_counts(
            summary,
            max_positive_fraction=args.max_positive_fraction,
            require_non_wt=not args.allow_no_non_wt,
        )
    except AssertionError as exc:
        record_failure(failures, "global label counts", str(exc))
    except Exception as exc:
        record_failure(failures, "global label counts", f"unexpected error: {exc!r}")

    if not args.skip_metadata:
        try:
            test_metadata_preserves_optotagging_fields(h5_paths, args.metadata_csv)
        except AssertionError as exc:
            record_failure(failures, "metadata propagation", str(exc))
        except Exception as exc:
            record_failure(
                failures,
                "metadata propagation",
                f"unexpected error: {exc!r}",
            )

    if failures:
        print_failure_summary(failures)
        raise SystemExit(1)

    print("Allen VBN optotagging validation passed!")


if __name__ == "__main__":
    main()
