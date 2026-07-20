"""Pipeline orchestration and fault-detection evaluation for CivicData Guardian."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.anomaly_detection import compare_baseline_current_volumes
from src.drift_detection import detect_distribution_drift
from src.fault_injection import RAW_DATA_PATH, load_clean_dataset, run_fault_injection
from src.ingestion import run_ingestion
from src.reporting import REPORTS_DIRECTORY, to_json_safe
from src.validation import load_schema, validate_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATEST_EVALUATION_PATH = REPORTS_DIRECTORY / "latest_evaluation.json"
SUPPORTED_ISSUE_TYPES = (
    "missing_required_column",
    "unexpected_column",
    "invalid_numeric_value",
    "duplicate_unique_key",
    "volume_anomaly",
    "category_drift",
    "unexpected_category",
    "invalid_timestamp",
    "missingness_threshold_exceeded",
    "invalid_date_order",
)
PENDING_DETECTOR_TYPES: tuple[str, ...] = ()


def normalize_issue_type(issue_type: object) -> str:
    """Normalize issue-type labels before deterministic matching."""
    return "_".join(str(issue_type).strip().lower().replace("-", " ").split())


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded ratio without raising for a zero denominator."""
    return round(numerator / denominator, 6) if denominator else 0.0


def extract_detected_issue_types(validation_result: Mapping[str, Any]) -> list[str]:
    """Return sorted, normalized issue types from an existing validation result."""
    return sorted(
        {
            normalize_issue_type(issue["issue_type"])
            for issue in validation_result["issues"]
        }
    )


def evaluate_scenario(
    scenario_name: str,
    expected_issue_types: Iterable[str],
    validation_result: Mapping[str, Any],
    row_count: int,
    pending: bool = False,
    volume_detection: Mapping[str, Any] | None = None,
    drift_detection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one scenario with a supported-issue-type multi-label matrix."""
    expected_types = sorted({normalize_issue_type(issue_type) for issue_type in expected_issue_types})
    detected_types = set(extract_detected_issue_types(validation_result))
    if volume_detection is not None:
        detected_types.update(
            normalize_issue_type(issue["issue_type"])
            for issue in volume_detection["issues"]
        )
    if drift_detection is not None:
        detected_types.update(
            normalize_issue_type(issue["issue_type"])
            for issue in drift_detection["issues"]
        )
    detected_types = sorted(detected_types)
    expected_supported = set(expected_types).intersection(SUPPORTED_ISSUE_TYPES)
    detected_supported = set(detected_types).intersection(SUPPORTED_ISSUE_TYPES)
    matrix: dict[str, str] = {}
    for issue_type in SUPPORTED_ISSUE_TYPES:
        expected = issue_type in expected_supported
        detected = issue_type in detected_supported
        if expected and detected:
            matrix[issue_type] = "true_positive"
        elif expected:
            matrix[issue_type] = "false_negative"
        elif detected:
            matrix[issue_type] = "false_positive"
        else:
            matrix[issue_type] = "true_negative"

    return {
        "scenario_name": scenario_name,
        "evaluation_status": "pending" if pending else "evaluated",
        "expected_issue_types": expected_types,
        "detected_issue_types": detected_types,
        "expected_supported_issue_types": sorted(expected_supported),
        "detected_supported_issue_types": sorted(detected_supported),
        "scenario_passed": None if pending else expected_supported == detected_supported,
        "row_count": row_count,
        "validation_passed": validation_result["validation_passed"],
        "schema_passed": validation_result["schema_passed"],
        "validation_result": validation_result,
        "issue_type_matrix": matrix,
        "volume_detection": volume_detection,
        "drift_detection": drift_detection,
    }


def aggregate_evaluation_metrics(
    scenario_results: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Aggregate per-scenario matrices into reproducible detection metrics."""
    counts = {key: 0 for key in ("true_positives", "false_positives", "false_negatives", "true_negatives")}
    by_fault_type = {
        issue_type: {"expected_incidents": 0, "detected_expected_incidents": 0}
        for issue_type in SUPPORTED_ISSUE_TYPES
    }
    pending_scenarios = 0

    for result in scenario_results:
        if result["evaluation_status"] == "pending":
            pending_scenarios += 1
            continue
        for issue_type, outcome in result["issue_type_matrix"].items():
            counts[f"{outcome}s"] += 1
            if outcome in {"true_positive", "false_negative"}:
                by_fault_type[issue_type]["expected_incidents"] += 1
            if outcome == "true_positive":
                by_fault_type[issue_type]["detected_expected_incidents"] += 1

    true_positives = counts["true_positives"]
    false_positives = counts["false_positives"]
    false_negatives = counts["false_negatives"]
    true_negatives = counts["true_negatives"]
    detection_rate_by_fault_type = {
        issue_type: {
            **values,
            "detection_rate": safe_divide(
                values["detected_expected_incidents"], values["expected_incidents"]
            ),
            "status": "evaluated",
        }
        for issue_type, values in by_fault_type.items()
    }
    return {
        **counts,
        "precision": safe_divide(true_positives, true_positives + false_positives),
        "recall": safe_divide(true_positives, true_positives + false_negatives),
        "f1_score": safe_divide(
            2 * true_positives, 2 * true_positives + false_positives + false_negatives
        ),
        "false_positive_rate": safe_divide(false_positives, false_positives + true_negatives),
        "detected_expected_incidents": true_positives,
        "total_expected_incidents": true_positives + false_negatives,
        "detection_rate_by_fault_type": detection_rate_by_fault_type,
        "pending_scenario_count": pending_scenarios,
    }


def load_or_ingest_clean_data() -> pd.DataFrame:
    """Load the existing raw CSV, ingesting only when it is unavailable."""
    if RAW_DATA_PATH.exists():
        return load_clean_dataset()
    dataframe, _ = run_ingestion()
    return dataframe


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load the generated fault manifest without changing its metadata."""
    with manifest_path.open(encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def save_evaluation_report(report: Mapping[str, Any], timestamp: str) -> tuple[Path, Path]:
    """Save stable and timestamped JSON evaluation reports."""
    REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    snapshot_path = REPORTS_DIRECTORY / f"evaluation_report_{timestamp}.json"
    safe_report = to_json_safe(report)
    for output_path in (LATEST_EVALUATION_PATH, snapshot_path):
        with output_path.open("w", encoding="utf-8") as report_file:
            json.dump(safe_report, report_file, indent=2, ensure_ascii=False)
            report_file.write("\n")
    return LATEST_EVALUATION_PATH, snapshot_path


def run_pipeline() -> dict[str, Any]:
    """Run clean validation, deterministic fault evaluation, and JSON reporting."""
    pipeline_started_at = time.perf_counter()
    clean_dataframe = load_or_ingest_clean_data()
    schema = load_schema()
    clean_validation_result = validate_dataframe(clean_dataframe, schema)
    scenario_metadata, manifest_path = run_fault_injection()
    manifest = load_manifest(manifest_path)

    scenario_results: list[dict[str, Any]] = []
    total_rows_evaluated = len(clean_dataframe)
    peak_scenario_row_count = len(clean_dataframe)
    clean_volume_detection = compare_baseline_current_volumes(clean_dataframe, clean_dataframe)
    clean_drift_detection = detect_distribution_drift(clean_dataframe, clean_dataframe)
    clean_result = evaluate_scenario(
        "clean_baseline",
        [],
        clean_validation_result,
        len(clean_dataframe),
        volume_detection=clean_volume_detection,
        drift_detection=clean_drift_detection,
    )

    for metadata in manifest["scenarios"]:
        scenario_path = PROJECT_ROOT / Path(metadata["output_path"])
        scenario_dataframe = pd.read_csv(scenario_path)
        validation_result = validate_dataframe(scenario_dataframe, schema)
        volume_detection = compare_baseline_current_volumes(clean_dataframe, scenario_dataframe)
        drift_detection = detect_distribution_drift(clean_dataframe, scenario_dataframe)
        expected_issue_types = metadata["expected_issue_types"]
        pending = any(
            normalize_issue_type(issue_type) in PENDING_DETECTOR_TYPES
            for issue_type in expected_issue_types
        )
        scenario_results.append(
            evaluate_scenario(
                metadata["scenario_name"],
                expected_issue_types,
                validation_result,
                len(scenario_dataframe),
                pending=pending,
                volume_detection=volume_detection,
                drift_detection=drift_detection,
            )
        )
        total_rows_evaluated += len(scenario_dataframe)
        peak_scenario_row_count = max(peak_scenario_row_count, len(scenario_dataframe))

    all_evaluation_results = [clean_result, *scenario_results]
    metrics = aggregate_evaluation_metrics(all_evaluation_results)
    for pending_type in PENDING_DETECTOR_TYPES:
        metrics["detection_rate_by_fault_type"][pending_type] = {
            "expected_incidents": sum(
                pending_type in result["expected_issue_types"] for result in scenario_results
            ),
            "detected_expected_incidents": 0,
            "detection_rate": None,
            "status": "pending",
        }

    pipeline_runtime_seconds = time.perf_counter() - pipeline_started_at
    records_validated_per_second = safe_divide(total_rows_evaluated, pipeline_runtime_seconds)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_paths = {
        "latest_evaluation_path": str(LATEST_EVALUATION_PATH),
        "snapshot_evaluation_path": str(REPORTS_DIRECTORY / f"evaluation_report_{timestamp}.json"),
    }
    report = {
        "report_version": "1.0",
        "run_timestamp_utc": datetime.now(timezone.utc),
        "clean_baseline_result": {
            **clean_result,
            "validation_result": clean_validation_result,
        },
        "supported_issue_types": list(SUPPORTED_ISSUE_TYPES),
        "pending_detector_types": list(PENDING_DETECTOR_TYPES),
        "per_scenario_results": scenario_results,
        "true_positives": metrics["true_positives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "true_negatives": metrics["true_negatives"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "false_positive_rate": metrics["false_positive_rate"],
        "detected_expected_incidents": metrics["detected_expected_incidents"],
        "total_expected_incidents": metrics["total_expected_incidents"],
        "detection_rate_by_fault_type": metrics["detection_rate_by_fault_type"],
        "pending_scenario_count": metrics["pending_scenario_count"],
        "total_rows_evaluated": total_rows_evaluated,
        "peak_scenario_row_count": peak_scenario_row_count,
        "total_pipeline_runtime_seconds": round(pipeline_runtime_seconds, 6),
        "records_validated_per_second": records_validated_per_second,
        "report_paths": report_paths,
        "fault_manifest_path": str(manifest_path),
        "manifest_scenario_count": manifest["scenario_count"],
        "generated_scenario_metadata": scenario_metadata,
    }
    latest_path, snapshot_path = save_evaluation_report(report, timestamp)
    report["report_paths"] = {
        "latest_evaluation_path": str(latest_path),
        "snapshot_evaluation_path": str(snapshot_path),
    }
    return to_json_safe(report)


def print_final_summary(report: Mapping[str, Any]) -> None:
    """Print a compact, readable evaluation summary for the project entry point."""
    print("CivicData Guardian pipeline evaluation")
    print(
        "Precision: {precision} | Recall: {recall} | F1: {f1_score} | FPR: {false_positive_rate}".format(
            **report
        )
    )
    print(
        "Rows evaluated: {total_rows_evaluated} | Peak scenario rows: {peak_scenario_row_count} | "
        "Records/sec: {records_validated_per_second}".format(**report)
    )
    print(f"Pending scenarios: {report['pending_scenario_count']}")
    print("Scenario results:")
    for result in report["per_scenario_results"]:
        print(
            "- {scenario_name}: expected={expected_issue_types}, detected={detected_issue_types}, "
            "status={evaluation_status}, passed={scenario_passed}".format(**result)
        )
    print(f"Latest evaluation report: {report['report_paths']['latest_evaluation_path']}")
    print(f"Snapshot evaluation report: {report['report_paths']['snapshot_evaluation_path']}")


def main() -> None:
    """Run the pipeline directly as a module."""
    print_final_summary(run_pipeline())


if __name__ == "__main__":
    main()
