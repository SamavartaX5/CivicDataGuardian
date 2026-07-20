"""Deterministic complaint-category and borough distribution drift detection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.fault_injection import RAW_DATA_PATH, load_clean_dataset, run_fault_injection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INJECTED_DIRECTORY = PROJECT_ROOT / "data" / "injected"
FREQUENCY_SHIFT_PATH = INJECTED_DIRECTORY / "complaint_type_frequency_shift.csv"
UNSEEN_CATEGORY_PATH = INJECTED_DIRECTORY / "unseen_complaint_type_category.csv"
DEFAULT_OVERALL_DRIFT_THRESHOLD = 0.05
DEFAULT_MINIMUM_FREQUENCY_CHANGE = 0.02
DEFAULT_MINIMUM_NEW_CATEGORY_COUNT = 10
DEFAULT_MINIMUM_NEW_CATEGORY_FREQUENCY = 0.01
DEFAULT_MINIMUM_DISAPPEARING_CATEGORY_COUNT = 10


def normalized_frequencies(dataframe: pd.DataFrame, field: str) -> tuple[dict[str, float], dict[str, int]]:
    """Return normalized non-blank category frequencies and counts without mutating input."""
    if dataframe.empty or field not in dataframe.columns:
        return {}, {}
    values = dataframe[field].copy().astype("string").str.strip()
    valid_values = values[values.notna() & values.ne("")]
    if valid_values.empty:
        return {}, {}
    counts = valid_values.value_counts()
    total = int(counts.sum())
    return (
        {str(category): float(count / total) for category, count in counts.sort_index().items()},
        {str(category): int(count) for category, count in counts.sort_index().items()},
    )


def jensen_shannon_distance(
    baseline_frequencies: Mapping[str, float], current_frequencies: Mapping[str, float]
) -> float:
    """Calculate a symmetric Jensen-Shannon distance for two category distributions."""
    categories = sorted(set(baseline_frequencies).union(current_frequencies))
    if not categories:
        return 0.0
    divergence = 0.0
    for category in categories:
        baseline = baseline_frequencies.get(category, 0.0)
        current = current_frequencies.get(category, 0.0)
        midpoint = (baseline + current) / 2
        if baseline:
            divergence += 0.5 * baseline * math.log2(baseline / midpoint)
        if current:
            divergence += 0.5 * current * math.log2(current / midpoint)
    return math.sqrt(max(divergence, 0.0))


def category_drift_issue(
    field: str,
    distance: float,
    shifted_categories: list[dict[str, Any]],
    disappearing_categories: list[dict[str, Any]],
    baseline_frequencies: Mapping[str, float],
    current_frequencies: Mapping[str, float],
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an incident-shaped issue for material distribution drift."""
    return {
        "issue_type": "category_drift",
        "severity": "warning",
        "affected_field": field,
        "description": "Category distribution exceeds the configured drift threshold.",
        "observed_value": {
            "jensen_shannon_distance": distance,
            "materially_shifted_categories": shifted_categories,
            "disappearing_categories": disappearing_categories,
            "baseline_frequencies": dict(baseline_frequencies),
            "current_frequencies": dict(current_frequencies),
        },
        "expected_value": {
            "maximum_jensen_shannon_distance": configuration["overall_drift_threshold"],
            "minimum_absolute_category_frequency_change": configuration[
                "minimum_absolute_category_frequency_change"
            ],
        },
        "suggested_action": "Investigate material category mix changes before relying on current data.",
    }


def unexpected_category_issue(
    field: str, new_categories: list[dict[str, Any]], configuration: Mapping[str, Any]
) -> dict[str, Any]:
    """Build an incident-shaped issue for qualifying newly appearing complaint categories."""
    return {
        "issue_type": "unexpected_category",
        "severity": "warning",
        "affected_field": field,
        "description": "New complaint categories exceed the configured appearance threshold.",
        "observed_value": {"new_categories": new_categories},
        "expected_value": {
            "minimum_new_category_count": configuration["minimum_new_category_count"],
            "minimum_new_category_frequency": configuration["minimum_new_category_frequency"],
        },
        "suggested_action": "Confirm new category values are valid source changes rather than coding errors.",
    }


def compare_categorical_distribution(
    baseline_dataframe: pd.DataFrame,
    current_dataframe: pd.DataFrame,
    field: str,
    overall_drift_threshold: float = DEFAULT_OVERALL_DRIFT_THRESHOLD,
    minimum_absolute_category_frequency_change: float = DEFAULT_MINIMUM_FREQUENCY_CHANGE,
    minimum_new_category_count: int = DEFAULT_MINIMUM_NEW_CATEGORY_COUNT,
    minimum_new_category_frequency: float = DEFAULT_MINIMUM_NEW_CATEGORY_FREQUENCY,
    minimum_disappearing_category_count: int = DEFAULT_MINIMUM_DISAPPEARING_CATEGORY_COUNT,
) -> dict[str, Any]:
    """Compare one categorical field and return deterministic drift findings."""
    configuration = {
        "overall_drift_threshold": overall_drift_threshold,
        "minimum_absolute_category_frequency_change": minimum_absolute_category_frequency_change,
        "minimum_new_category_count": minimum_new_category_count,
        "minimum_new_category_frequency": minimum_new_category_frequency,
        "minimum_disappearing_category_count": minimum_disappearing_category_count,
    }
    if field not in baseline_dataframe.columns or field not in current_dataframe.columns:
        return {
            "detector_name": "categorical_distribution_drift",
            "configuration": configuration,
            "compared_field": field,
            "field_available": False,
            "baseline_frequencies": {},
            "current_frequencies": {},
            "jensen_shannon_distance": 0.0,
            "shifted_categories": [],
            "new_categories": [],
            "disappearing_categories": [],
            "drift_detected": False,
            "issues": [],
        }

    baseline_frequencies, baseline_counts = normalized_frequencies(baseline_dataframe, field)
    current_frequencies, current_counts = normalized_frequencies(current_dataframe, field)
    distance = jensen_shannon_distance(baseline_frequencies, current_frequencies)
    shared_categories = sorted(set(baseline_frequencies).intersection(current_frequencies))
    shifted_categories = [
        {
            "category": category,
            "baseline_frequency": baseline_frequencies[category],
            "current_frequency": current_frequencies[category],
            "absolute_frequency_change": abs(current_frequencies[category] - baseline_frequencies[category]),
        }
        for category in shared_categories
        if abs(current_frequencies[category] - baseline_frequencies[category])
        >= minimum_absolute_category_frequency_change
    ]
    new_categories = [
        {
            "category": category,
            "count": current_counts[category],
            "frequency": current_frequencies[category],
        }
        for category in sorted(set(current_frequencies).difference(baseline_frequencies))
        if current_counts[category] >= minimum_new_category_count
        or current_frequencies[category] >= minimum_new_category_frequency
    ]
    disappearing_categories = [
        {
            "category": category,
            "baseline_count": baseline_counts[category],
            "baseline_frequency": baseline_frequencies[category],
        }
        for category in sorted(set(baseline_frequencies).difference(current_frequencies))
        if baseline_counts[category] >= minimum_disappearing_category_count
    ]
    material_existing_drift = bool(shifted_categories or disappearing_categories)
    category_drift_detected = distance >= overall_drift_threshold and material_existing_drift
    issues: list[dict[str, Any]] = []
    if category_drift_detected:
        issues.append(
            category_drift_issue(
                field,
                distance,
                shifted_categories,
                disappearing_categories,
                baseline_frequencies,
                current_frequencies,
                configuration,
            )
        )
    if field == "complaint_type" and new_categories:
        issues.append(unexpected_category_issue(field, new_categories, configuration))
    return {
        "detector_name": "categorical_distribution_drift",
        "configuration": configuration,
        "compared_field": field,
        "field_available": True,
        "baseline_frequencies": baseline_frequencies,
        "current_frequencies": current_frequencies,
        "jensen_shannon_distance": distance,
        "shifted_categories": shifted_categories,
        "new_categories": new_categories,
        "disappearing_categories": disappearing_categories,
        "drift_detected": bool(issues),
        "issues": issues,
    }


def detect_distribution_drift(
    baseline_dataframe: pd.DataFrame, current_dataframe: pd.DataFrame
) -> dict[str, Any]:
    """Detect complaint-type and borough distribution drift without changing inputs."""
    complaint_result = compare_categorical_distribution(
        baseline_dataframe, current_dataframe, "complaint_type"
    )
    borough_result = compare_categorical_distribution(baseline_dataframe, current_dataframe, "borough")
    issues = [*complaint_result["issues"], *borough_result["issues"]]
    return {
        "detector_name": "complaint_and_borough_distribution_drift",
        "configuration": {
            "overall_drift_threshold": DEFAULT_OVERALL_DRIFT_THRESHOLD,
            "minimum_absolute_category_frequency_change": DEFAULT_MINIMUM_FREQUENCY_CHANGE,
            "minimum_new_category_count": DEFAULT_MINIMUM_NEW_CATEGORY_COUNT,
            "minimum_new_category_frequency": DEFAULT_MINIMUM_NEW_CATEGORY_FREQUENCY,
        },
        "field_results": {
            "complaint_type": complaint_result,
            "borough": borough_result,
        },
        "issues": issues,
    }


def print_comparison(label: str, result: Mapping[str, Any]) -> None:
    """Print the requested compact comparison details for one scenario."""
    complaint_result = result["field_results"]["complaint_type"]
    print(label)
    print(f"Jensen-Shannon distance: {complaint_result['jensen_shannon_distance']}")
    print(f"Materially shifted categories: {complaint_result['shifted_categories']}")
    print(f"New categories: {complaint_result['new_categories']}")
    print(f"Disappearing categories: {complaint_result['disappearing_categories']}")
    print(f"Detected issue types: {[issue['issue_type'] for issue in result['issues']]}" )


def main() -> None:
    """Compare clean data with the deterministic frequency-shift and unseen-category scenarios."""
    if not FREQUENCY_SHIFT_PATH.exists() or not UNSEEN_CATEGORY_PATH.exists():
        run_fault_injection()
    clean_dataframe = load_clean_dataset()
    frequency_shift_dataframe = pd.read_csv(FREQUENCY_SHIFT_PATH)
    unseen_category_dataframe = pd.read_csv(UNSEEN_CATEGORY_PATH)
    print_comparison(
        "Complaint-type frequency shift",
        detect_distribution_drift(clean_dataframe, frequency_shift_dataframe),
    )
    print_comparison(
        "Unseen complaint category",
        detect_distribution_drift(clean_dataframe, unseen_category_dataframe),
    )


if __name__ == "__main__":
    main()
