"""Offline smoke coverage for pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import pipeline


def test_pipeline_offline_smoke_uses_tmp_artifacts(monkeypatch, tmp_path: Path, valid_dataframe, schema) -> None:
    injected_root = tmp_path / "data" / "injected"
    scenarios = []

    def add_scenario(name: str, expected: list[str], dataframe: pd.DataFrame) -> None:
        relative_path = Path("data") / "injected" / f"{name}.csv"
        output_path = tmp_path / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
        scenarios.append(
            {
                "scenario_name": name,
                "expected_issue_types": expected,
                "output_path": str(relative_path),
            }
        )

    add_scenario("missing_column", ["missing_required_column"], valid_dataframe.drop(columns=["agency"]))
    add_scenario("unexpected", ["unexpected_column"], valid_dataframe.assign(extra="x"))
    add_scenario("numeric", ["invalid_numeric_value"], valid_dataframe.assign(latitude="bad"))
    add_scenario("duplicate", ["duplicate_unique_key"], pd.concat([valid_dataframe, valid_dataframe.iloc[[0]]], ignore_index=True))
    add_scenario("timestamp", ["invalid_timestamp"], valid_dataframe.assign(created_date="bad"))
    add_scenario("missingness", ["missingness_threshold_exceeded"], valid_dataframe.assign(descriptor=pd.NA))
    reversed_dates = valid_dataframe.copy(deep=True)
    reversed_dates.loc[0, "closed_date"] = "2025-12-31T00:00:00.000"
    add_scenario("date_order", ["invalid_date_order"], reversed_dates)
    volume = pd.concat([valid_dataframe, valid_dataframe.iloc[[0] * 200]], ignore_index=True)
    volume.loc[len(valid_dataframe):, "unique_key"] = range(90_000, 90_200)
    add_scenario("volume", ["volume_anomaly"], volume)
    shifted = valid_dataframe.copy(deep=True)
    shifted.loc[:9, "complaint_type"] = "Noise - Street/Sidewalk"
    add_scenario("drift", ["category_drift"], shifted)
    unseen = valid_dataframe.copy(deep=True)
    unseen.loc[:9, "complaint_type"] = "Synthetic Unseen Complaint Type"
    add_scenario("unseen", ["unexpected_category"], unseen)

    manifest_path = injected_root / "fault_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"scenario_count": len(scenarios), "scenarios": scenarios}), encoding="utf-8")
    reports_directory = tmp_path / "reports"
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "REPORTS_DIRECTORY", reports_directory)
    monkeypatch.setattr(pipeline, "LATEST_EVALUATION_PATH", reports_directory / "latest_evaluation.json")
    monkeypatch.setattr(
        pipeline,
        "load_or_ingest_clean_data",
        lambda: (
            valid_dataframe.copy(deep=True),
            {
                "source_url": "https://example.test",
                "latest_created_date": valid_dataframe.loc[0, "created_date"],
            },
        ),
    )
    monkeypatch.setattr(pipeline, "load_schema", lambda: schema)
    monkeypatch.setattr(pipeline, "run_fault_injection", lambda: (scenarios, manifest_path))

    report = pipeline.run_pipeline()
    saved_report = reports_directory / "latest_evaluation.json"
    saved_clean_report = reports_directory / "latest_report.json"
    assert saved_report.exists()
    assert saved_clean_report.exists()
    assert json.loads(saved_report.read_text(encoding="utf-8"))["report_version"] == "1.0"
    assert json.loads(saved_clean_report.read_text(encoding="utf-8"))["source_url"] == "https://example.test"
    assert report["pending_scenario_count"] == 0
    assert set(pipeline.SUPPORTED_ISSUE_TYPES).issubset(report["detection_rate_by_fault_type"])
    assert all(0 <= report[name] <= 1 for name in ("precision", "recall", "f1_score", "false_positive_rate"))


def test_dashboard_module_imports_without_running_ui() -> None:
    import app.streamlit_app  # noqa: F401
