"""Reliability scoring and JSON-report tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.reporting import (
    build_structured_report,
    calculate_reliability_score,
    determine_health_status,
    save_report,
    to_json_safe,
)


def test_reliability_scoring_boundaries_and_per_issue_penalties(issue_factory) -> None:
    assert calculate_reliability_score([]) == 100
    assert calculate_reliability_score([issue_factory("unexpected_column")]) == 95
    assert calculate_reliability_score([issue_factory("duplicate_unique_key"), issue_factory("missingness_threshold_exceeded")]) == 80
    assert calculate_reliability_score([issue_factory("missing_required_column")]) == 70
    assert calculate_reliability_score([issue_factory("missing_required_column") for _ in range(4)]) == 0
    assert calculate_reliability_score([issue_factory("duplicate_unique_key", observed_value={"count": 100})]) == 90
    assert calculate_reliability_score([issue_factory("unknown_type")]) == 100
    assert determine_health_status(90) == "Healthy"
    assert determine_health_status(89) == determine_health_status(75) == "Warning"
    assert determine_health_status(74) == "Critical"


def test_json_safe_and_saved_reports_round_trip(issue_factory, tmp_path: Path) -> None:
    values = {
        "path": Path("example.json"),
        "datetime": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
        "integer": np.int64(1),
        "float": np.float64(1.5),
        "boolean": np.bool_(True),
        "missing": pd.NA,
        "nan": float("nan"),
    }
    safe_values = to_json_safe(values)
    assert safe_values["path"] == "example.json"
    assert safe_values["missing"] is None and safe_values["nan"] is None

    validation_result = {
        "validation_passed": False,
        "schema_passed": True,
        "total_rows": 1,
        "issues": [issue_factory("unexpected_column", observed_value=values)],
        "quality_metrics": values,
    }
    report = build_structured_report(
        validation_result,
        {"latest_created_date": pd.Timestamp("2026-01-01T00:00:00Z")},
        "erm2-nwe9",
        "https://example.test",
        1.0,
        1.0,
        0.1,
    )
    latest, snapshot = save_report(report, tmp_path)
    with latest.open(encoding="utf-8") as latest_file, snapshot.open(encoding="utf-8") as snapshot_file:
        assert json.load(latest_file) == json.load(snapshot_file) == report
