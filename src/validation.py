"""Schema-driven validation for NYC 311 service request data."""

from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import Any, Mapping, TypedDict

import pandas as pd
import yaml

from src.quality_metrics import (
    assemble_quality_metrics,
    calculate_base_quality_metrics,
    empty_or_blank_mask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "config" / "schema.yaml"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nyc_311_recent.csv"
ISSUE_SAMPLE_LIMIT = 10


class ValidationIssue(TypedDict):
    """A single aggregated validation finding."""

    issue_type: str
    severity: str
    affected_field: str
    description: str
    observed_value: object
    expected_value: object
    suggested_action: str


class ValidationResult(TypedDict):
    """Structured result returned by validate_dataframe."""

    validation_passed: bool
    schema_passed: bool
    total_rows: int
    issues: list[ValidationIssue]
    quality_metrics: dict[str, Any]
    rejected_row_indices: list[object]
    rejected_row_count: int


def load_schema(schema_path: Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    """Load the YAML data contract from disk."""
    with schema_path.open(encoding="utf-8") as schema_file:
        schema = yaml.safe_load(schema_file)
    if not isinstance(schema, dict):
        raise ValueError("Schema configuration must contain a YAML mapping.")
    return schema


def load_raw_data(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw NYC 311 CSV without transforming its values."""
    return pd.read_csv(data_path)


def make_issue(
    issue_type: str,
    affected_field: str,
    description: str,
    observed_value: object,
    expected_value: object,
    suggested_action: str,
    severity: str = "error",
) -> ValidationIssue:
    """Build a consistently shaped validation issue."""
    return {
        "issue_type": issue_type,
        "severity": severity,
        "affected_field": affected_field,
        "description": description,
        "observed_value": observed_value,
        "expected_value": expected_value,
        "suggested_action": suggested_action,
    }


def sample_indices(mask: pd.Series) -> list[object]:
    """Return a short list of affected row indices for readable issue output."""
    return list(mask[mask].index[:ISSUE_SAMPLE_LIMIT])


def check_missing_required_columns(
    dataframe: pd.DataFrame, schema: Mapping[str, Any]
) -> list[ValidationIssue]:
    """Report required schema columns absent from the dataset."""
    issues: list[ValidationIssue] = []
    for column in schema.get("required_columns", []):
        if column not in dataframe.columns:
            issues.append(
                make_issue(
                    "missing_required_column",
                    column,
                    "A required schema column is absent from the dataset.",
                    "absent",
                    "column present",
                    "Restore the column in the source query or dataset.",
                )
            )
    return issues


def check_unexpected_columns(
    dataframe: pd.DataFrame, schema: Mapping[str, Any]
) -> list[ValidationIssue]:
    """Report returned columns that are not defined by the schema when prohibited."""
    if schema.get("allow_unexpected_columns", True):
        return []
    defined_columns = set(schema.get("columns", {}))
    unexpected_columns = [column for column in dataframe.columns if column not in defined_columns]
    return [
        make_issue(
            "unexpected_column",
            column,
            "The dataset contains a column that is not defined by the schema.",
            "present",
            "column absent or explicitly allowed",
            "Remove the column from the input or add it to the schema deliberately.",
        )
        for column in unexpected_columns
    ]


def check_datatype_compatibility(
    dataframe: pd.DataFrame, schema: Mapping[str, Any]
) -> tuple[list[ValidationIssue], set[object], dict[str, pd.Series], int, int]:
    """Check datetime and numeric compatibility using copied Series only."""
    issues: list[ValidationIssue] = []
    rejected_indices: set[object] = set()
    invalid_timestamp_indices: set[object] = set()
    invalid_coordinate_indices: set[object] = set()
    parsed_timestamps: dict[str, pd.Series] = {}

    for column, rules in schema.get("columns", {}).items():
        if column not in dataframe.columns:
            continue
        source_values = dataframe[column].copy()
        present_values = ~empty_or_blank_mask(source_values)
        datatype = rules.get("datatype")

        if datatype == "datetime":
            parsed_values = pd.to_datetime(source_values.copy(), errors="coerce", utc=True)
            parsed_timestamps[column] = parsed_values
            invalid_mask = present_values & parsed_values.isna()
            invalid_count = int(invalid_mask.sum())
            if invalid_count:
                issues.append(
                    make_issue(
                        "invalid_timestamp",
                        column,
                        "Non-empty values could not be parsed as datetimes.",
                        {"count": invalid_count, "sample_indices": sample_indices(invalid_mask)},
                        "datetime-compatible value",
                        "Correct the timestamp format or remove invalid records.",
                    )
                )
                rejected_indices.update(invalid_mask[invalid_mask].index)
                invalid_timestamp_indices.update(invalid_mask[invalid_mask].index)

        if datatype == "numeric":
            numeric_values = pd.to_numeric(source_values.copy(), errors="coerce")
            invalid_mask = present_values & numeric_values.isna()
            invalid_count = int(invalid_mask.sum())
            if invalid_count:
                issues.append(
                    make_issue(
                        "invalid_numeric_value",
                        column,
                        "Non-empty values could not be converted to numbers.",
                        {"count": invalid_count, "sample_indices": sample_indices(invalid_mask)},
                        "numeric-compatible value",
                        "Correct the numeric values or remove invalid records.",
                    )
                )
                rejected_indices.update(invalid_mask[invalid_mask].index)
                invalid_coordinate_indices.update(invalid_mask[invalid_mask].index)

            minimum = rules.get("minimum")
            maximum = rules.get("maximum")
            range_mask = pd.Series(False, index=dataframe.index)
            if minimum is not None:
                range_mask |= numeric_values.notna() & (numeric_values < float(minimum))
            if maximum is not None:
                range_mask |= numeric_values.notna() & (numeric_values > float(maximum))
            range_count = int(range_mask.sum())
            if range_count:
                issues.append(
                    make_issue(
                        "numeric_out_of_range",
                        column,
                        "Numeric values fall outside the configured inclusive range.",
                        {"count": range_count, "sample_indices": sample_indices(range_mask)},
                        {"minimum": minimum, "maximum": maximum},
                        "Correct coordinates or remove out-of-range records.",
                    )
                )
                rejected_indices.update(range_mask[range_mask].index)
                invalid_coordinate_indices.update(range_mask[range_mask].index)

    return (
        issues,
        rejected_indices,
        parsed_timestamps,
        len(invalid_timestamp_indices),
        len(invalid_coordinate_indices),
    )


def check_missing_unique_keys(
    dataframe: pd.DataFrame, schema: Mapping[str, Any]
) -> tuple[list[ValidationIssue], set[object], int]:
    """Report null and whitespace-only values in the configured unique-key column."""
    column = schema.get("unique_key_column")
    if not column or column not in dataframe.columns:
        return [], set(), 0
    missing_mask = empty_or_blank_mask(dataframe[column])
    missing_count = int(missing_mask.sum())
    if not missing_count:
        return [], set(), 0
    return (
        [
            make_issue(
                "missing_unique_key",
                column,
                "Unique-key values must not be null, empty, or whitespace-only.",
                {"count": missing_count, "sample_indices": sample_indices(missing_mask)},
                "non-empty unique key",
                "Populate the key or exclude affected records.",
            )
        ],
        set(missing_mask[missing_mask].index),
        missing_count,
    )


def check_duplicate_unique_keys(
    dataframe: pd.DataFrame, schema: Mapping[str, Any]
) -> tuple[list[ValidationIssue], set[object], int]:
    """Report duplicate non-empty values in the configured unique-key column."""
    column = schema.get("unique_key_column")
    if not column or column not in dataframe.columns:
        return [], set(), 0
    non_empty = ~empty_or_blank_mask(dataframe[column])
    duplicate_mask = pd.Series(False, index=dataframe.index)
    duplicate_mask.loc[non_empty] = dataframe.loc[non_empty, column].duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if not duplicate_count:
        return [], set(), 0
    return (
        [
            make_issue(
                "duplicate_unique_key",
                column,
                "Non-empty unique-key values must be unique.",
                {"count": duplicate_count, "sample_indices": sample_indices(duplicate_mask)},
                "no duplicate values",
                "Deduplicate records or correct repeated unique keys.",
            )
        ],
        set(duplicate_mask[duplicate_mask].index),
        duplicate_count,
    )


def check_date_order(
    parsed_timestamps: Mapping[str, pd.Series], schema: Mapping[str, Any]
) -> tuple[list[ValidationIssue], set[object], int]:
    """Report closure timestamps that occur before creation timestamps."""
    rule = schema.get("rules", {}).get("date_order", {})
    earlier_column = rule.get("earlier_column")
    later_column = rule.get("later_column")
    if earlier_column not in parsed_timestamps or later_column not in parsed_timestamps:
        return [], set(), 0
    earlier_values = parsed_timestamps[earlier_column]
    later_values = parsed_timestamps[later_column]
    invalid_mask = earlier_values.notna() & later_values.notna() & (later_values < earlier_values)
    invalid_count = int(invalid_mask.sum())
    if not invalid_count:
        return [], set(), 0
    return (
        [
            make_issue(
                "invalid_date_order",
                f"{later_column}/{earlier_column}",
                "The closure timestamp occurs before the creation timestamp.",
                {"count": invalid_count, "sample_indices": sample_indices(invalid_mask)},
                rule.get("condition", "closed_date must not be earlier than created_date"),
                "Correct the timestamps or remove affected records.",
            )
        ],
        set(invalid_mask[invalid_mask].index),
        invalid_count,
    )


def check_non_nullable_values(
    dataframe: pd.DataFrame, schema: Mapping[str, Any]
) -> tuple[list[ValidationIssue], set[object]]:
    """Report blank values in schema-declared non-nullable fields."""
    issues: list[ValidationIssue] = []
    rejected_indices: set[object] = set()
    unique_key_column = schema.get("unique_key_column")
    for column, rules in schema.get("columns", {}).items():
        if column == unique_key_column or rules.get("nullable", True) or column not in dataframe.columns:
            continue
        missing_mask = empty_or_blank_mask(dataframe[column])
        missing_count = int(missing_mask.sum())
        if missing_count:
            issues.append(
                make_issue(
                    "missing_required_value",
                    column,
                    "A non-nullable field contains null, empty, or whitespace-only values.",
                    {"count": missing_count, "sample_indices": sample_indices(missing_mask)},
                    "non-empty value",
                    "Populate the field or remove affected records.",
                )
            )
            rejected_indices.update(missing_mask[missing_mask].index)
    return issues, rejected_indices


def check_missingness_thresholds(base_metrics: Mapping[str, Any]) -> list[ValidationIssue]:
    """Turn pre-calculated missingness threshold exceedances into issues."""
    return [
        make_issue(
            "missingness_threshold_exceeded",
            column,
            "Observed missingness exceeds the configured maximum percentage.",
            details["observed_missing_percentage"],
            details["maximum_missing_percentage"],
            "Investigate source completeness or adjust the threshold deliberately.",
        )
        for column, details in base_metrics["columns_exceeding_missingness_thresholds"].items()
    ]


def validate_dataframe(dataframe: pd.DataFrame, schema: Mapping[str, Any]) -> ValidationResult:
    """Validate a DataFrame against the supplied YAML-derived schema mapping."""
    base_metrics = calculate_base_quality_metrics(dataframe, schema)
    issues = check_missing_required_columns(dataframe, schema)
    issues.extend(check_unexpected_columns(dataframe, schema))

    datatype_issues, datatype_rejections, parsed_timestamps, invalid_timestamp_row_count, invalid_coordinate_row_count = (
        check_datatype_compatibility(dataframe, schema)
    )
    issues.extend(datatype_issues)

    missing_key_issues, missing_key_rejections, missing_unique_key_count = check_missing_unique_keys(
        dataframe, schema
    )
    duplicate_key_issues, duplicate_key_rejections, duplicate_unique_key_count = check_duplicate_unique_keys(
        dataframe, schema
    )
    date_order_issues, date_order_rejections, invalid_date_order_count = check_date_order(
        parsed_timestamps, schema
    )
    non_nullable_issues, non_nullable_rejections = check_non_nullable_values(
        dataframe, schema
    )
    issues.extend(missing_key_issues)
    issues.extend(duplicate_key_issues)
    issues.extend(date_order_issues)
    issues.extend(non_nullable_issues)
    issues.extend(check_missingness_thresholds(base_metrics))

    rejected_indices = (
        datatype_rejections
        | missing_key_rejections
        | duplicate_key_rejections
        | date_order_rejections
        | non_nullable_rejections
    )
    validation_counts = {
        "duplicate_unique_key_count": duplicate_unique_key_count,
        "invalid_timestamp_row_count": invalid_timestamp_row_count,
        "invalid_date_order_count": invalid_date_order_count,
        "invalid_coordinate_row_count": invalid_coordinate_row_count,
        "missing_unique_key_count": missing_unique_key_count,
        "rejected_row_count": len(rejected_indices),
    }
    schema_issue_types = {"missing_required_column", "unexpected_column", "invalid_timestamp", "invalid_numeric_value"}
    schema_passed = not any(issue["issue_type"] in schema_issue_types for issue in issues)
    return {
        "validation_passed": not issues,
        "schema_passed": schema_passed,
        "total_rows": len(dataframe),
        "issues": issues,
        "quality_metrics": assemble_quality_metrics(base_metrics, validation_counts),
        "rejected_row_indices": sorted(rejected_indices),
        "rejected_row_count": len(rejected_indices),
    }


def main() -> None:
    """Run validation for the configured NYC 311 raw-data sample."""
    schema = load_schema()
    dataframe = load_raw_data()
    result = validate_dataframe(dataframe, schema)
    print("NYC 311 validation summary")
    print(f"Validation passed: {result['validation_passed']}")
    print(f"Schema passed: {result['schema_passed']}")
    print(f"Total rows: {result['total_rows']}")
    print(f"Rejected rows: {result['rejected_row_count']}")
    print("Issues:")
    pprint(result["issues"])
    print("Quality metrics:")
    pprint(result["quality_metrics"])


if __name__ == "__main__":
    main()
