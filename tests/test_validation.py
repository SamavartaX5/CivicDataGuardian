"""Schema-driven validation tests using deterministic in-memory data."""

from __future__ import annotations

import pandas as pd
import pytest

from src.validation import validate_dataframe


ISSUE_FIELDS = {
    "issue_type",
    "severity",
    "affected_field",
    "description",
    "observed_value",
    "expected_value",
    "suggested_action",
}


def issue_types(result: dict) -> set[str]:
    return {issue["issue_type"] for issue in result["issues"]}


def test_clean_dataframe_passes_without_mutation(valid_dataframe, schema) -> None:
    original = valid_dataframe.copy(deep=True)
    result = validate_dataframe(valid_dataframe, schema)

    assert result["validation_passed"] is True
    pd.testing.assert_frame_equal(valid_dataframe, original)


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        (lambda frame: frame.drop(columns=["agency"]), "missing_required_column"),
        (lambda frame: frame.assign(unexpected_field="value"), "unexpected_column"),
        (lambda frame: frame.assign(unique_key=[1] * len(frame)), "duplicate_unique_key"),
        (lambda frame: frame.assign(created_date="bad-date"), "invalid_timestamp"),
        (lambda frame: frame.assign(latitude="not-a-number"), "invalid_numeric_value"),
        (lambda frame: frame.assign(longitude=999.0), "numeric_out_of_range"),
        (lambda frame: frame.assign(agency="   "), "missing_required_value"),
        (lambda frame: frame.assign(descriptor=pd.NA), "missingness_threshold_exceeded"),
    ],
)
def test_validation_detects_core_issues(valid_dataframe, schema, transform, expected) -> None:
    result = validate_dataframe(transform(valid_dataframe.copy(deep=True)), schema)
    assert expected in issue_types(result)


def test_missing_key_date_order_and_unique_rejected_indices(valid_dataframe, schema) -> None:
    corrupted = valid_dataframe.copy(deep=True)
    corrupted.loc[0, "unique_key"] = pd.NA
    corrupted.loc[0, "created_date"] = "bad-date"
    corrupted.loc[1, "closed_date"] = "2025-12-31T00:00:00.000"
    result = validate_dataframe(corrupted, schema)

    assert {"missing_unique_key", "invalid_timestamp", "invalid_date_order"}.issubset(issue_types(result))
    assert result["rejected_row_count"] == len(set(result["rejected_row_indices"]))
    assert 0 in result["rejected_row_indices"]


def test_duplicate_row_metrics_remain_separate_from_duplicate_keys(valid_dataframe, schema) -> None:
    duplicated = pd.concat([valid_dataframe, valid_dataframe.iloc[[0]]], ignore_index=True)
    result = validate_dataframe(duplicated, schema)
    metrics = result["quality_metrics"]

    assert metrics["duplicate_row_count"] == 1
    assert metrics["duplicate_unique_key_count"] == 2
    assert metrics["duplicate_unique_key_percentage"] > metrics["duplicate_row_percentage"]


def test_missing_dependent_columns_do_not_crash_and_issues_are_complete(valid_dataframe, schema) -> None:
    result = validate_dataframe(valid_dataframe.drop(columns=["created_date", "latitude"]), schema)

    assert "missing_required_column" in issue_types(result)
    assert all(ISSUE_FIELDS.issubset(issue) for issue in result["issues"])
