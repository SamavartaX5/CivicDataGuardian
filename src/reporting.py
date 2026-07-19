"""Reliability scoring and structured incident reporting for NYC 311 data."""

from __future__ import annotations

import json
import math
import time
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.ingestion import DATASET_ID, SOURCE_URL, run_ingestion
from src.validation import load_schema, validate_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
REPORT_VERSION = "1.0"
ISSUE_FIELDS = (
    "issue_type",
    "severity",
    "affected_field",
    "description",
    "observed_value",
    "expected_value",
    "suggested_action",
)

# Configurable product rules for this application; these weights are not a universal scientific score.
ISSUE_PENALTIES = {
    "missing_required_column": 30,
    "incompatible_datatype": 15,
    "invalid_numeric_value": 15,
    "duplicate_unique_key": 10,
    "missingness_threshold_exceeded": 10,
    "invalid_timestamp": 10,
    "invalid_date_order": 10,
    "numeric_out_of_range": 10,
    "missing_unique_key": 10,
    "missing_required_value": 10,
    "unexpected_column": 5,
}


def calculate_reliability_score(issues: Sequence[Mapping[str, Any]]) -> int:
    """Apply configurable per-issue product penalties to a score from 0 to 100."""
    total_penalty = sum(ISSUE_PENALTIES.get(issue.get("issue_type"), 0) for issue in issues)
    return max(0, min(100, 100 - total_penalty))


def determine_health_status(reliability_score: int | float) -> str:
    """Convert a bounded reliability score into a product health label."""
    if reliability_score >= 90:
        return "Healthy"
    if reliability_score >= 75:
        return "Warning"
    return "Critical"


def to_json_safe(value: Any) -> Any:
    """Recursively convert common pandas, NumPy, datetime, and path values for JSON."""
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
        return [to_json_safe(record) for record in value.to_dict(orient="records")]
    if isinstance(value, Mapping):
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


def build_structured_report(
    validation_result: Mapping[str, Any],
    ingestion_metadata: Mapping[str, Any],
    dataset_id: str,
    source_url: str,
    pipeline_runtime_seconds: float,
    records_validated_per_second: float,
    report_generation_time_seconds: float,
) -> dict[str, Any]:
    """Build a JSON-safe incident report from existing ingestion and validation output."""
    issues = validation_result["issues"]
    reliability_score = calculate_reliability_score(issues)
    incidents = [
        {field: issue.get(field) for field in ISSUE_FIELDS}
        for issue in issues
    ]
    report = {
        "report_version": REPORT_VERSION,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_id": dataset_id,
        "source_url": source_url,
        "latest_source_data_timestamp": ingestion_metadata.get("latest_created_date"),
        "rows_processed": validation_result["total_rows"],
        "reliability_score": reliability_score,
        "health_status": determine_health_status(reliability_score),
        "validation_passed": validation_result["validation_passed"],
        "schema_passed": validation_result["schema_passed"],
        "issue_count": len(issues),
        "incidents": incidents,
        "quality_metrics": validation_result["quality_metrics"],
        "ingestion_metadata": dict(ingestion_metadata),
        "pipeline_runtime_seconds": round(pipeline_runtime_seconds, 3),
        "records_validated_per_second": round(records_validated_per_second, 3),
        "report_generation_time_seconds": round(report_generation_time_seconds, 6),
    }
    return to_json_safe(report)


def save_report(
    report: Mapping[str, Any], output_directory: Path = REPORTS_DIRECTORY
) -> tuple[Path, Path]:
    """Save stable and timestamped copies of an incident report as indented JSON."""
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    latest_path = output_directory / "latest_report.json"
    snapshot_path = output_directory / f"incident_report_{timestamp}.json"
    json_report = to_json_safe(report)

    for output_path in (latest_path, snapshot_path):
        with output_path.open("w", encoding="utf-8") as report_file:
            json.dump(json_report, report_file, indent=2, ensure_ascii=False)
            report_file.write("\n")
    return latest_path, snapshot_path


def run_reporting() -> tuple[dict[str, Any], Path, Path]:
    """Run ingestion and validation, then build and save both incident report files."""
    pipeline_started_at = time.perf_counter()
    dataframe, ingestion_metadata = run_ingestion()
    schema = load_schema()
    validation_result = validate_dataframe(dataframe, schema)
    pipeline_runtime_seconds = time.perf_counter() - pipeline_started_at
    records_validated_per_second = (
        len(dataframe) / pipeline_runtime_seconds if pipeline_runtime_seconds > 0 else 0.0
    )

    report_started_at = time.perf_counter()
    report = build_structured_report(
        validation_result=validation_result,
        ingestion_metadata=ingestion_metadata,
        dataset_id=schema.get("dataset", {}).get("id", DATASET_ID),
        source_url=ingestion_metadata.get("source_url", SOURCE_URL),
        pipeline_runtime_seconds=pipeline_runtime_seconds,
        records_validated_per_second=records_validated_per_second,
        report_generation_time_seconds=0.0,
    )
    report["report_generation_time_seconds"] = round(
        time.perf_counter() - report_started_at, 6
    )
    latest_path, snapshot_path = save_report(report)
    return report, latest_path, snapshot_path


def main() -> None:
    """Run the reporting pipeline and print its concise execution summary."""
    report, latest_path, snapshot_path = run_reporting()
    print(f"Reliability score: {report['reliability_score']}")
    print(f"Health status: {report['health_status']}")
    print(f"Issue count: {report['issue_count']}")
    print(f"Rows processed: {report['rows_processed']}")
    print(f"Records validated per second: {report['records_validated_per_second']}")
    print(f"Latest report: {latest_path}")
    print(f"Incident report: {snapshot_path}")
    print(f"Report-generation time: {report['report_generation_time_seconds']} seconds")


if __name__ == "__main__":
    main()
