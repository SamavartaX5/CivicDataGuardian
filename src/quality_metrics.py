"""Dataset-level quality metrics for schema-driven validation."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def empty_or_blank_mask(series: pd.Series) -> pd.Series:
    """Return a mask for null and whitespace-only values without altering the source."""
    text_values = series.astype("string").str.strip()
    return series.isna() | text_values.eq("")


def calculate_missing_percentages(dataframe: pd.DataFrame) -> dict[str, float]:
    """Calculate missing percentages for every returned column."""
    total_rows = len(dataframe)
    if total_rows == 0:
        return {column: 0.0 for column in dataframe.columns}
    return {
        column: round(float(empty_or_blank_mask(dataframe[column]).mean() * 100), 2)
        for column in dataframe.columns
    }


def find_missingness_threshold_exceedances(
    missing_percentages: Mapping[str, float], schema: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    """Return configured columns whose observed missingness exceeds their threshold."""
    exceedances: dict[str, dict[str, float]] = {}
    for column, rules in schema.get("columns", {}).items():
        if column not in missing_percentages:
            continue
        threshold = rules.get("max_missing_percentage")
        if threshold is not None and missing_percentages[column] > float(threshold):
            exceedances[column] = {
                "observed_missing_percentage": missing_percentages[column],
                "maximum_missing_percentage": float(threshold),
            }
    return exceedances


def calculate_base_quality_metrics(
    dataframe: pd.DataFrame, schema: Mapping[str, Any]
) -> dict[str, Any]:
    """Calculate dataset statistics shared by validation checks and final reporting."""
    total_rows = len(dataframe)
    missing_percentages = calculate_missing_percentages(dataframe)
    duplicate_row_count = int(dataframe.duplicated().sum())
    duplicate_row_percentage = (
        round(duplicate_row_count / total_rows * 100, 2) if total_rows else 0.0
    )
    return {
        "total_rows_processed": total_rows,
        "missing_value_percentages": missing_percentages,
        "columns_exceeding_missingness_thresholds": find_missingness_threshold_exceedances(
            missing_percentages, schema
        ),
        "duplicate_row_count": duplicate_row_count,
        "duplicate_row_percentage": duplicate_row_percentage,
    }


def assemble_quality_metrics(
    base_metrics: Mapping[str, Any], validation_counts: Mapping[str, int]
) -> dict[str, Any]:
    """Combine shared dataset statistics with counts produced by record-level checks."""
    metrics = dict(base_metrics)
    total_rows = int(base_metrics["total_rows_processed"])
    duplicate_unique_key_count = int(validation_counts["duplicate_unique_key_count"])
    duplicate_unique_key_percentage = (
        round(duplicate_unique_key_count / total_rows * 100, 2) if total_rows else 0.0
    )
    metrics.update(
        {
            "duplicate_unique_key_count": duplicate_unique_key_count,
            "duplicate_unique_key_percentage": duplicate_unique_key_percentage,
            "invalid_timestamp_row_count": int(
                validation_counts["invalid_timestamp_row_count"]
            ),
            "invalid_date_order_count": int(validation_counts["invalid_date_order_count"]),
            "invalid_coordinate_row_count": int(
                validation_counts["invalid_coordinate_row_count"]
            ),
            "missing_unique_key_count": int(validation_counts["missing_unique_key_count"]),
            "rejected_row_count": int(validation_counts["rejected_row_count"]),
        }
    )
    return metrics
