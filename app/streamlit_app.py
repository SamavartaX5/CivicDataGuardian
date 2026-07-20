"""Streamlit monitoring dashboard for CivicData Guardian."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection import aggregate_daily_request_counts, compare_baseline_current_volumes
from src.drift_detection import detect_distribution_drift
from src.pipeline import run_pipeline


REPORT_PATH = PROJECT_ROOT / "reports" / "latest_report.json"
EVALUATION_PATH = PROJECT_ROOT / "reports" / "latest_evaluation.json"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nyc_311_recent.csv"
SCHEMA_PATH = PROJECT_ROOT / "config" / "schema.yaml"
INJECTED_DIRECTORY = PROJECT_ROOT / "data" / "injected"


def file_signature(path: Path) -> int:
    """Return a cache-busting file modification signature, or zero when absent."""
    return path.stat().st_mtime_ns if path.exists() else 0


@st.cache_data(show_spinner=False)
def load_json_file(path_string: str, signature: int) -> dict[str, Any] | None:
    """Load a JSON artifact only when its file signature changes."""
    path = Path(path_string)
    if not signature:
        return None
    with path.open(encoding="utf-8") as source_file:
        return json.load(source_file)


@st.cache_data(show_spinner=False)
def load_csv_file(path_string: str, signature: int) -> pd.DataFrame | None:
    """Load a CSV artifact only when its file signature changes."""
    path = Path(path_string)
    return pd.read_csv(path) if signature else None


@st.cache_data(show_spinner=False)
def load_schema_file(path_string: str, signature: int) -> dict[str, Any] | None:
    """Load the schema configuration used for dashboard thresholds."""
    path = Path(path_string)
    if not signature:
        return None
    with path.open(encoding="utf-8") as schema_file:
        return yaml.safe_load(schema_file)


def format_value(value: Any) -> str:
    """Render nested incident fields as readable table text."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return "" if value is None else str(value)


def incident_dataframe(incidents: list[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert incident objects into a display-ready table."""
    columns = [
        "severity",
        "issue_type",
        "affected_field",
        "description",
        "observed_value",
        "expected_value",
        "suggested_action",
    ]
    return pd.DataFrame(
        [{column: format_value(incident.get(column)) for column in columns} for incident in incidents],
        columns=columns,
    )


def scenario_lookup(evaluation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index persisted scenario results by their stable scenario name."""
    return {item["scenario_name"]: item for item in evaluation.get("per_scenario_results", [])}


def render_overview(report: Mapping[str, Any], evaluation: Mapping[str, Any]) -> None:
    """Render real reliability and evaluation metrics from generated JSON artifacts."""
    st.subheader("Overview")
    first_row = st.columns(4)
    first_row[0].metric("Reliability score", report.get("reliability_score", "—"))
    first_row[1].metric("Health status", report.get("health_status", "—"))
    first_row[2].metric("Rows processed", report.get("rows_processed", "—"))
    first_row[3].metric("Validation issues", report.get("issue_count", "—"))
    second_row = st.columns(4)
    second_row[0].metric("Precision", evaluation.get("precision", "—"))
    second_row[1].metric("Recall", evaluation.get("recall", "—"))
    second_row[2].metric("F1 score", evaluation.get("f1_score", "—"))
    second_row[3].metric("False-positive rate", evaluation.get("false_positive_rate", "—"))

    details = {
        "Latest source-data timestamp": report.get("latest_source_data_timestamp", "Unavailable"),
        "Total pipeline runtime (seconds)": evaluation.get("total_pipeline_runtime_seconds", "Unavailable"),
        "Records validated per second": evaluation.get("records_validated_per_second", "Unavailable"),
        "Total rows evaluated": evaluation.get("total_rows_evaluated", "Unavailable"),
        "Peak scenario row count": evaluation.get("peak_scenario_row_count", "Unavailable"),
        "Scenarios evaluated": len(evaluation.get("per_scenario_results", [])),
        "Pending scenario count": evaluation.get("pending_scenario_count", "Unavailable"),
    }
    details_frame = pd.DataFrame(details.items(), columns=["Metric", "Value"])
    details_frame["Value"] = details_frame["Value"].map(format_value)
    st.dataframe(details_frame, hide_index=True, width="stretch")


def render_data_quality(
    evaluation: Mapping[str, Any], raw_dataframe: pd.DataFrame | None, schema: Mapping[str, Any] | None
) -> None:
    """Render clean baseline quality metrics, missingness, and limited rejected-row context."""
    st.subheader("Data Quality")
    clean_validation = evaluation.get("clean_baseline_result", {}).get("validation_result", {})
    quality_metrics = clean_validation.get("quality_metrics", {})
    status_columns = st.columns(4)
    status_columns[0].metric("Schema passed", clean_validation.get("schema_passed", "—"))
    status_columns[1].metric("Validation passed", clean_validation.get("validation_passed", "—"))
    status_columns[2].metric("Rejected rows", clean_validation.get("rejected_row_count", "—"))
    status_columns[3].metric("Missing unique keys", quality_metrics.get("missing_unique_key_count", "—"))
    metric_columns = st.columns(4)
    metric_columns[0].metric("Duplicate unique keys", quality_metrics.get("duplicate_unique_key_count", "—"))
    metric_columns[1].metric("Duplicate unique-key %", quality_metrics.get("duplicate_unique_key_percentage", "—"))
    metric_columns[2].metric("Exact duplicate rows", quality_metrics.get("duplicate_row_count", "—"))
    metric_columns[3].metric("Exact duplicate-row %", quality_metrics.get("duplicate_row_percentage", "—"))
    metric_columns = st.columns(3)
    metric_columns[0].metric("Invalid timestamp rows", quality_metrics.get("invalid_timestamp_row_count", "—"))
    metric_columns[1].metric("Invalid date-order rows", quality_metrics.get("invalid_date_order_count", "—"))
    metric_columns[2].metric("Invalid coordinate rows", quality_metrics.get("invalid_coordinate_row_count", "—"))

    missing_percentages = quality_metrics.get("missing_value_percentages", {})
    if missing_percentages:
        st.markdown("#### Missing-value percentage")
        missing_chart = pd.DataFrame(
            {"column": list(missing_percentages), "missing_percentage": list(missing_percentages.values())}
        ).set_index("column")
        st.bar_chart(missing_chart)
        threshold_rows = []
        for column, percentage in missing_percentages.items():
            threshold = (schema or {}).get("columns", {}).get(column, {}).get("max_missing_percentage")
            threshold_rows.append(
                {
                    "column": column,
                    "missing percentage": percentage,
                    "configured threshold": threshold,
                    "threshold exceeded": threshold is not None and percentage > threshold,
                }
            )
        st.dataframe(pd.DataFrame(threshold_rows), hide_index=True, width="stretch")

    rejected_indices = clean_validation.get("rejected_row_indices", [])
    if rejected_indices and raw_dataframe is not None:
        preview_indices = [index for index in rejected_indices[:25] if index in raw_dataframe.index]
        st.markdown("#### Rejected-row preview")
        st.dataframe(raw_dataframe.loc[preview_indices], hide_index=True, width="stretch")
    elif not rejected_indices:
        st.success("No clean-baseline rows were rejected.")


def render_anomalies_and_drift(
    evaluation: Mapping[str, Any], raw_dataframe: pd.DataFrame | None, selected_scenario: str
) -> None:
    """Render scenario status plus volume and drift context with existing helpers."""
    st.subheader("Anomalies and Drift")
    scenarios = scenario_lookup(evaluation)
    focus_names = [
        "daily_request_volume_spike",
        "complaint_type_frequency_shift",
        "unseen_complaint_type_category",
    ]
    status_rows = [
        {
            "scenario": name,
            "evaluation status": scenarios.get(name, {}).get("evaluation_status", "Unavailable"),
            "passed": scenarios.get(name, {}).get("scenario_passed", "—"),
            "detected types": ", ".join(scenarios.get(name, {}).get("detected_issue_types", [])),
        }
        for name in focus_names
    ]
    st.dataframe(pd.DataFrame(status_rows), hide_index=True, width="stretch")

    scenario_rows = [
        {
            "scenario name": item["scenario_name"],
            "expected issue types": ", ".join(item.get("expected_issue_types", [])),
            "detected issue types": ", ".join(item.get("detected_issue_types", [])),
            "evaluation status": item.get("evaluation_status"),
            "passed": item.get("scenario_passed"),
            "row count": item.get("row_count"),
        }
        for item in evaluation.get("per_scenario_results", [])
    ]
    st.markdown("#### Scenario results")
    st.dataframe(pd.DataFrame(scenario_rows), hide_index=True, width="stretch")

    if raw_dataframe is None:
        st.info("Raw data is unavailable; run `python run_monitor.py` to restore dashboard inputs.")
        return
    st.markdown("#### Daily request volume")
    daily_counts = aggregate_daily_request_counts(raw_dataframe)
    if not daily_counts.empty:
        st.line_chart(daily_counts.set_index("calendar_date")["request_count"])
    if selected_scenario == "daily_request_volume_spike":
        scenario_path = INJECTED_DIRECTORY / "daily_request_volume_spike.csv"
        injected = load_csv_file(str(scenario_path), file_signature(scenario_path))
        if injected is not None:
            comparison = compare_baseline_current_volumes(raw_dataframe, injected)
            anomalies = [record for record in comparison["daily_volume_records"] if record["is_anomaly"]]
            if anomalies:
                st.warning("Detected daily volume anomaly dates")
                st.dataframe(pd.DataFrame(anomalies), hide_index=True, width="stretch")
            else:
                st.info("No daily volume anomaly was detected for this scenario.")

    if selected_scenario in {"complaint_type_frequency_shift", "unseen_complaint_type_category"}:
        scenario_path = INJECTED_DIRECTORY / f"{selected_scenario}.csv"
        injected = load_csv_file(str(scenario_path), file_signature(scenario_path))
        if injected is None:
            st.warning("The selected injected CSV is missing. Run `python run_monitor.py` to regenerate it.")
            return
        drift = detect_distribution_drift(raw_dataframe, injected)
        complaint = drift["field_results"]["complaint_type"]
        st.markdown("#### Complaint-type drift")
        drift_columns = st.columns(2)
        drift_columns[0].metric("Jensen-Shannon distance", round(complaint["jensen_shannon_distance"], 4))
        drift_columns[1].metric("Detected issue types", ", ".join(issue["issue_type"] for issue in drift["issues"]) or "None")
        st.write("Materially shifted categories", complaint["shifted_categories"])
        st.write("New categories", complaint["new_categories"])
        st.write("Disappearing categories", complaint["disappearing_categories"])
        frequency_frame = pd.DataFrame(
            {
                "baseline": pd.Series(complaint["baseline_frequencies"]),
                "current": pd.Series(complaint["current_frequencies"]),
            }
        ).fillna(0)
        top_categories = frequency_frame.max(axis=1).nlargest(12).index
        st.bar_chart(frequency_frame.loc[top_categories])


def render_incident_report(report: Mapping[str, Any], evaluation: Mapping[str, Any], selected_scenario: str) -> None:
    """Render clean and selected-scenario incidents with readable nested values."""
    st.subheader("Incident Report")
    st.markdown("#### Latest clean validation report")
    clean_incidents = report.get("incidents", [])
    if clean_incidents:
        st.dataframe(incident_dataframe(clean_incidents), hide_index=True, width="stretch")
    else:
        st.success("Healthy: the latest clean validation report contains no incidents.")

    selected = scenario_lookup(evaluation).get(selected_scenario)
    if selected:
        st.markdown(f"#### Injected scenario incidents: {selected_scenario}")
        incidents = selected.get("validation_result", {}).get("issues", [])
        if incidents:
            st.dataframe(incident_dataframe(incidents), hide_index=True, width="stretch")
        else:
            st.info("This scenario has no validation-engine incidents; detector findings appear in Anomalies and Drift.")

    st.markdown("#### Download generated JSON")
    report_columns = st.columns(2)
    if REPORT_PATH.exists():
        report_columns[0].download_button(
            "Download latest_report.json",
            REPORT_PATH.read_bytes(),
            file_name="latest_report.json",
            mime="application/json",
        )
    if EVALUATION_PATH.exists():
        report_columns[1].download_button(
            "Download latest_evaluation.json",
            EVALUATION_PATH.read_bytes(),
            file_name="latest_evaluation.json",
            mime="application/json",
        )


def main() -> None:
    """Run the Streamlit dashboard from the project root."""
    st.set_page_config(page_title="CivicData Guardian", page_icon="🛡️", layout="wide")
    report = load_json_file(str(REPORT_PATH), file_signature(REPORT_PATH))
    evaluation = load_json_file(str(EVALUATION_PATH), file_signature(EVALUATION_PATH))
    raw_dataframe = load_csv_file(str(RAW_DATA_PATH), file_signature(RAW_DATA_PATH))
    schema = load_schema_file(str(SCHEMA_PATH), file_signature(SCHEMA_PATH))

    st.sidebar.title("CivicData Guardian")
    st.sidebar.caption(f"Latest report: {(report or {}).get('run_timestamp_utc', 'Unavailable')}")
    if st.sidebar.button("Run monitoring pipeline", width="stretch"):
        try:
            with st.spinner("Running the monitoring pipeline..."):
                run_pipeline()
            st.cache_data.clear()
            st.success("Monitoring pipeline completed. Reloading generated artifacts.")
            st.rerun()
        except Exception as error:
            st.error(f"Monitoring pipeline failed: {error}")

    scenario_names = [item["scenario_name"] for item in (evaluation or {}).get("per_scenario_results", [])]
    selected_scenario = st.sidebar.selectbox("Scenario detail", scenario_names or ["No scenarios available"])

    st.title("CivicData Guardian")
    st.caption("Operational monitoring for NYC 311 data quality, anomalies, drift, and incidents.")
    if report is None or evaluation is None or raw_dataframe is None:
        st.warning("Generated monitoring artifacts are missing. Run `python run_monitor.py` or use the sidebar button.")
        return

    overview_tab, quality_tab, anomaly_tab, incident_tab = st.tabs(
        ["Overview", "Data Quality", "Anomalies and Drift", "Incident Report"]
    )
    with overview_tab:
        render_overview(report, evaluation)
    with quality_tab:
        render_data_quality(evaluation, raw_dataframe, schema)
    with anomaly_tab:
        render_anomalies_and_drift(evaluation, raw_dataframe, selected_scenario)
    with incident_tab:
        render_incident_report(report, evaluation, selected_scenario)


if __name__ == "__main__":
    main()
