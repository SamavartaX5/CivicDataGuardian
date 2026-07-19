"""Deterministic fault injection scenarios for NYC 311 data."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nyc_311_recent.csv"
INJECTED_DIRECTORY = PROJECT_ROOT / "data" / "injected"
MANIFEST_PATH = INJECTED_DIRECTORY / "fault_manifest.json"
MANIFEST_VERSION = "1.0"


def to_json_safe(value: Any) -> Any:
    """Convert common pandas, NumPy, datetime, path, and missing values for JSON."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [to_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Series):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return [to_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        is_missing = False
    if isinstance(is_missing, (bool, np.bool_)) and is_missing:
        return None
    return value


def scenario_output_path(scenario_name: str) -> Path:
    """Return the repository-relative CSV output path for a scenario."""
    return Path("data") / "injected" / f"{scenario_name}.csv"


def select_indices(
    dataframe: pd.DataFrame,
    count: int,
    random_seed: int,
    eligible_indices: pd.Index | None = None,
) -> list[object]:
    """Choose deterministic, non-repeating row indices from an eligible population."""
    population = dataframe.index if eligible_indices is None else eligible_indices
    selected_count = min(count, len(population))
    if selected_count == 0:
        return []
    generator = np.random.default_rng(random_seed)
    return list(generator.choice(population.to_numpy(), size=selected_count, replace=False))


def build_metadata(
    scenario_name: str,
    fault_type: str,
    expected_issue_types: list[str],
    affected_columns: list[str],
    affected_row_indices: list[object],
    parameters: dict[str, Any],
    random_seed: int,
    description: str,
) -> dict[str, Any]:
    """Build the common structured ground-truth metadata for one scenario."""
    return {
        "scenario_name": scenario_name,
        "fault_type": fault_type,
        "expected_issue_types": expected_issue_types,
        "affected_columns": affected_columns,
        "affected_row_indices": affected_row_indices,
        "affected_row_count": len(affected_row_indices),
        "parameters": parameters,
        "random_seed": random_seed,
        "description": description,
        "output_path": str(scenario_output_path(scenario_name)),
    }


def inject_deleted_agency(dataframe: pd.DataFrame, random_seed: int = 101) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Delete the required agency column."""
    scenario_name = "deleted_required_agency_column"
    corrupted = dataframe.drop(columns=["agency"]).copy(deep=True)
    metadata = build_metadata(
        scenario_name,
        "deleted_required_column",
        ["missing_required_column"],
        ["agency"],
        list(dataframe.index),
        {"deleted_column": "agency"},
        random_seed,
        "Deletes the required agency column from the dataset.",
    )
    return corrupted, metadata


def inject_renamed_complaint_type(
    dataframe: pd.DataFrame, random_seed: int = 102
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rename complaint_type so the expected field is missing and a new field appears."""
    scenario_name = "renamed_complaint_type_column"
    corrupted = dataframe.rename(columns={"complaint_type": "complaint_category"}).copy(deep=True)
    metadata = build_metadata(
        scenario_name,
        "renamed_column",
        ["missing_required_column", "unexpected_column"],
        ["complaint_type", "complaint_category"],
        list(dataframe.index),
        {"original_column": "complaint_type", "renamed_column": "complaint_category"},
        random_seed,
        "Renames complaint_type to complaint_category.",
    )
    return corrupted, metadata


def inject_invalid_latitude_text(
    dataframe: pd.DataFrame, random_seed: int = 103, affected_count: int = 50
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace selected latitude values with text that cannot be parsed numerically."""
    scenario_name = "invalid_latitude_text"
    corrupted = dataframe.copy(deep=True)
    affected_indices = select_indices(corrupted, affected_count, random_seed)
    corrupted["latitude"] = corrupted["latitude"].astype("object")
    corrupted.loc[affected_indices, "latitude"] = "invalid-latitude"
    metadata = build_metadata(
        scenario_name,
        "invalid_numeric_text",
        ["invalid_numeric_value"],
        ["latitude"],
        affected_indices,
        {"replacement_value": "invalid-latitude", "requested_affected_count": affected_count},
        random_seed,
        "Replaces selected latitude values with non-numeric text.",
    )
    return corrupted, metadata


def inject_duplicate_rows(
    dataframe: pd.DataFrame, random_seed: int = 104, affected_count: int = 100
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append selected existing rows, retaining their original unique-key values."""
    scenario_name = "duplicate_rows_preserved_unique_keys"
    source_indices = select_indices(dataframe, affected_count, random_seed)
    duplicate_rows = dataframe.loc[source_indices].copy(deep=True)
    corrupted = pd.concat([dataframe.copy(deep=True), duplicate_rows], ignore_index=True)
    appended_indices = list(range(len(dataframe), len(corrupted)))
    metadata = build_metadata(
        scenario_name,
        "duplicate_rows",
        ["duplicate_unique_key"],
        ["unique_key"],
        appended_indices,
        {"source_row_indices": source_indices, "appended_row_count": len(appended_indices)},
        random_seed,
        "Appends duplicate records while preserving their unique_key values.",
    )
    return corrupted, metadata


def inject_invalid_created_date(
    dataframe: pd.DataFrame, random_seed: int = 105, affected_count: int = 50
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace selected creation timestamps with invalid timestamp text."""
    scenario_name = "invalid_created_date_text"
    corrupted = dataframe.copy(deep=True)
    affected_indices = select_indices(corrupted, affected_count, random_seed)
    corrupted["created_date"] = corrupted["created_date"].astype("object")
    corrupted.loc[affected_indices, "created_date"] = "invalid-created-date"
    metadata = build_metadata(
        scenario_name,
        "invalid_timestamp_text",
        ["invalid_timestamp"],
        ["created_date"],
        affected_indices,
        {"replacement_value": "invalid-created-date", "requested_affected_count": affected_count},
        random_seed,
        "Replaces selected created_date values with unparseable timestamp text.",
    )
    return corrupted, metadata


def inject_descriptor_missingness(
    dataframe: pd.DataFrame, percentage_points: int, random_seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Increase descriptor missingness by approximately the requested percentage points."""
    scenario_name = f"descriptor_missingness_plus_{percentage_points}_points"
    corrupted = dataframe.copy(deep=True)
    descriptor_values = corrupted["descriptor"]
    initially_missing = descriptor_values.isna() | descriptor_values.astype("string").str.strip().eq("")
    baseline_percentage = float(initially_missing.mean() * 100)
    available_indices = corrupted.index[~initially_missing]
    requested_count = math.ceil(len(corrupted) * percentage_points / 100)
    affected_indices = select_indices(
        corrupted, requested_count, random_seed, eligible_indices=available_indices
    )
    corrupted.loc[affected_indices, "descriptor"] = pd.NA
    resulting_missing = corrupted["descriptor"].isna() | corrupted["descriptor"].astype("string").str.strip().eq("")
    metadata = build_metadata(
        scenario_name,
        "increased_missingness",
        ["missingness_threshold_exceeded"],
        ["descriptor"],
        affected_indices,
        {
            "baseline_missingness_percentage": round(baseline_percentage, 2),
            "target_additional_percentage_points": percentage_points,
            "resulting_missingness_percentage": round(float(resulting_missing.mean() * 100), 2),
        },
        random_seed,
        "Increases descriptor missingness relative to the clean dataset baseline.",
    )
    return corrupted, metadata


def inject_descriptor_missingness_plus_10(
    dataframe: pd.DataFrame, random_seed: int = 106
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Increase descriptor missingness by approximately 10 percentage points."""
    return inject_descriptor_missingness(dataframe, 10, random_seed)


def inject_descriptor_missingness_plus_25(
    dataframe: pd.DataFrame, random_seed: int = 107
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Increase descriptor missingness by approximately 25 percentage points."""
    return inject_descriptor_missingness(dataframe, 25, random_seed)


def inject_descriptor_missingness_plus_50(
    dataframe: pd.DataFrame, random_seed: int = 108
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Increase descriptor missingness by approximately 50 percentage points."""
    return inject_descriptor_missingness(dataframe, 50, random_seed)


def inject_daily_volume_spike(
    dataframe: pd.DataFrame, random_seed: int = 109, added_count: int = 500
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add same-day records with new keys to simulate a daily request-volume spike."""
    scenario_name = "daily_request_volume_spike"
    parsed_dates = pd.to_datetime(dataframe["created_date"].copy(), errors="coerce", utc=True)
    valid_dates = parsed_dates.dropna()
    if valid_dates.empty:
        raise ValueError("A volume spike requires at least one valid created_date value.")
    date_counts = valid_dates.dt.strftime("%Y-%m-%d").value_counts()
    target_date = date_counts.index[0]
    eligible_indices = dataframe.index[parsed_dates.dt.strftime("%Y-%m-%d").eq(target_date)]
    generator = np.random.default_rng(random_seed)
    source_indices = list(generator.choice(eligible_indices.to_numpy(), size=added_count, replace=True))
    spike_rows = dataframe.loc[source_indices].copy(deep=True)
    numeric_keys = pd.to_numeric(dataframe["unique_key"], errors="coerce").dropna()
    if numeric_keys.empty:
        synthetic_keys = [f"synthetic-volume-{random_seed}-{position}" for position in range(added_count)]
    else:
        key_start = int(numeric_keys.max()) + 1
        synthetic_keys = list(range(key_start, key_start + added_count))
    spike_rows["unique_key"] = synthetic_keys
    corrupted = pd.concat([dataframe.copy(deep=True), spike_rows], ignore_index=True)
    appended_indices = list(range(len(dataframe), len(corrupted)))
    metadata = build_metadata(
        scenario_name,
        "volume_spike",
        ["volume_anomaly"],
        ["created_date", "unique_key"],
        appended_indices,
        {
            "target_date": target_date,
            "added_record_count": added_count,
            "source_row_indices": source_indices,
            "synthetic_unique_key_start": synthetic_keys[0],
        },
        random_seed,
        "Adds same-day records with new synthetic unique_key values to create a volume spike.",
    )
    return corrupted, metadata


def inject_complaint_type_frequency_shift(
    dataframe: pd.DataFrame, random_seed: int = 110, affected_count: int = 1000
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Move complaint_type values toward one existing category without changing row count."""
    scenario_name = "complaint_type_frequency_shift"
    corrupted = dataframe.copy(deep=True)
    target_category = corrupted["complaint_type"].dropna().mode().iat[0]
    eligible_indices = corrupted.index[corrupted["complaint_type"].ne(target_category)]
    affected_indices = select_indices(
        corrupted, affected_count, random_seed, eligible_indices=eligible_indices
    )
    corrupted.loc[affected_indices, "complaint_type"] = target_category
    metadata = build_metadata(
        scenario_name,
        "category_frequency_shift",
        ["category_drift"],
        ["complaint_type"],
        affected_indices,
        {"target_category": target_category, "requested_affected_count": affected_count},
        random_seed,
        "Shifts complaint_type frequencies toward one existing category without changing row count.",
    )
    return corrupted, metadata


def inject_unseen_complaint_category(
    dataframe: pd.DataFrame, random_seed: int = 111, affected_count: int = 250
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Introduce a complaint_type category not present in the clean dataset."""
    scenario_name = "unseen_complaint_type_category"
    corrupted = dataframe.copy(deep=True)
    affected_indices = select_indices(corrupted, affected_count, random_seed)
    unseen_category = "Synthetic Unseen Complaint Type"
    corrupted.loc[affected_indices, "complaint_type"] = unseen_category
    metadata = build_metadata(
        scenario_name,
        "unseen_category",
        ["unexpected_category"],
        ["complaint_type"],
        affected_indices,
        {"unseen_category": unseen_category, "requested_affected_count": affected_count},
        random_seed,
        "Introduces a complaint_type category not present in the clean dataset.",
    )
    return corrupted, metadata


def inject_reversed_closed_dates(
    dataframe: pd.DataFrame, random_seed: int = 112, affected_count: int = 100
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Set selected closed_date values to one day before valid created_date values."""
    scenario_name = "closed_date_before_created_date"
    corrupted = dataframe.copy(deep=True)
    created_dates = pd.to_datetime(corrupted["created_date"].copy(), errors="coerce", utc=True)
    eligible_indices = corrupted.index[created_dates.notna()]
    affected_indices = select_indices(
        corrupted, affected_count, random_seed, eligible_indices=eligible_indices
    )
    earlier_dates = created_dates.loc[affected_indices] - pd.Timedelta(days=1)
    corrupted.loc[affected_indices, "closed_date"] = earlier_dates.dt.strftime("%Y-%m-%dT%H:%M:%S.000")
    metadata = build_metadata(
        scenario_name,
        "reversed_date_order",
        ["invalid_date_order"],
        ["created_date", "closed_date"],
        affected_indices,
        {"offset_days": -1, "requested_affected_count": affected_count},
        random_seed,
        "Sets selected closed_date values earlier than their valid created_date values.",
    )
    return corrupted, metadata


SCENARIO_FUNCTIONS: tuple[Callable[[pd.DataFrame], tuple[pd.DataFrame, dict[str, Any]]], ...] = (
    inject_deleted_agency,
    inject_renamed_complaint_type,
    inject_invalid_latitude_text,
    inject_duplicate_rows,
    inject_invalid_created_date,
    inject_descriptor_missingness_plus_10,
    inject_descriptor_missingness_plus_25,
    inject_descriptor_missingness_plus_50,
    inject_daily_volume_spike,
    inject_complaint_type_frequency_shift,
    inject_unseen_complaint_category,
    inject_reversed_closed_dates,
)


def load_clean_dataset(data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the clean raw NYC 311 dataset without changing it."""
    return pd.read_csv(data_path)


def save_corrupted_dataset(dataframe: pd.DataFrame, metadata: dict[str, Any]) -> Path:
    """Save one corrupted dataset at its metadata-defined path."""
    output_path = PROJECT_ROOT / Path(metadata["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def save_manifest(scenarios: list[dict[str, Any]]) -> Path:
    """Save JSON-safe ground truth for all injected fault scenarios."""
    INJECTED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at_utc": datetime.now(timezone.utc),
        "source_data_path": str(Path("data") / "raw" / "nyc_311_recent.csv"),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as manifest_file:
        json.dump(to_json_safe(manifest), manifest_file, indent=2, ensure_ascii=False)
        manifest_file.write("\n")
    return MANIFEST_PATH


def run_fault_injection() -> tuple[list[dict[str, Any]], Path]:
    """Generate, verify, save, and manifest every deterministic fault scenario."""
    clean_dataframe = load_clean_dataset()
    clean_baseline = clean_dataframe.copy(deep=True)
    scenario_metadata: list[dict[str, Any]] = []
    scenario_object_ids: set[int] = set()

    for scenario_function in SCENARIO_FUNCTIONS:
        corrupted_dataframe, metadata = scenario_function(clean_dataframe)
        if corrupted_dataframe is clean_dataframe or id(corrupted_dataframe) in scenario_object_ids:
            raise AssertionError("Each fault scenario must return an independent DataFrame object.")
        scenario_object_ids.add(id(corrupted_dataframe))
        pd.testing.assert_frame_equal(clean_dataframe, clean_baseline)
        save_corrupted_dataset(corrupted_dataframe, metadata)
        scenario_metadata.append(to_json_safe(metadata))

    pd.testing.assert_frame_equal(clean_dataframe, clean_baseline)
    manifest_path = save_manifest(scenario_metadata)
    return scenario_metadata, manifest_path


def main() -> None:
    """Generate every fault scenario and print a compact saved-artifact summary."""
    scenarios, manifest_path = run_fault_injection()
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    print("Fault injection summary")
    print(f"Scenario count: {manifest['scenario_count']}")
    print("Scenario names:")
    for scenario in manifest["scenarios"]:
        print(
            f"- {scenario['scenario_name']}: {scenario['affected_row_count']} rows | "
            f"{scenario['expected_issue_types']} | {scenario['output_path']}"
        )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
