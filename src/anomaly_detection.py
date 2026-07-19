"""Deterministic daily request-volume anomaly detection for NYC 311 data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.fault_injection import RAW_DATA_PATH, load_clean_dataset, run_fault_injection
from src.reporting import to_json_safe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOLUME_SPIKE_PATH = PROJECT_ROOT / "data" / "injected" / "daily_request_volume_spike.csv"
DEFAULT_ROLLING_WINDOW = 7
DEFAULT_MIN_PERIODS = 3
DEFAULT_STANDARDIZED_THRESHOLD = 3.0
DEFAULT_MIN_ABSOLUTE_INCREASE = 200


def aggregate_daily_request_counts(
    dataframe: pd.DataFrame, date_column: str = "created_date"
) -> pd.DataFrame:
    """Safely parse a date column and return sorted calendar-day request counts."""
    if dataframe.empty or date_column not in dataframe.columns:
        return pd.DataFrame(columns=["calendar_date", "request_count"])
    parsed_dates = pd.to_datetime(dataframe[date_column].copy(), errors="coerce", utc=True)
    valid_dates = parsed_dates.dropna()
    if valid_dates.empty:
        return pd.DataFrame(columns=["calendar_date", "request_count"])
    counts = valid_dates.dt.strftime("%Y-%m-%d").value_counts().sort_index()
    return counts.rename_axis("calendar_date").reset_index(name="request_count")


def build_volume_issue(record: Mapping[str, Any], configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Build an incident-shaped issue for one anomalous daily count comparison."""
    return {
        "issue_type": "volume_anomaly",
        "severity": "warning",
        "affected_field": "created_date",
        "description": "Daily request volume exceeds the configured clean-baseline threshold.",
        "observed_value": {
            "calendar_date": record["calendar_date"],
            "baseline_count": record["baseline_count"],
            "current_count": record["current_count"],
            "absolute_difference": record["absolute_difference"],
            "percentage_change": record["percentage_change"],
            "standardized_count_difference": record["standardized_count_difference"],
        },
        "expected_value": {
            "minimum_absolute_increase": configuration["minimum_absolute_increase"],
            "standardized_threshold": configuration["standardized_threshold"],
        },
        "suggested_action": "Investigate the source or operational cause of the daily volume increase.",
    }


def detect_rolling_volume_anomalies(
    dataframe: pd.DataFrame,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    minimum_periods: int = DEFAULT_MIN_PERIODS,
    standardized_threshold: float = DEFAULT_STANDARDIZED_THRESHOLD,
    minimum_absolute_increase: int = DEFAULT_MIN_ABSOLUTE_INCREASE,
    detect_increases: bool = True,
    detect_decreases: bool = False,
) -> dict[str, Any]:
    """Detect daily anomalies against prior rolling history without mutating input data."""
    configuration = {
        "rolling_window": rolling_window,
        "minimum_periods": minimum_periods,
        "standardized_threshold": standardized_threshold,
        "minimum_absolute_increase": minimum_absolute_increase,
        "detect_increases": detect_increases,
        "detect_decreases": detect_decreases,
    }
    daily_counts = aggregate_daily_request_counts(dataframe)
    if daily_counts.empty:
        return {
            "detector_name": "rolling_daily_volume",
            "configuration": configuration,
            "daily_volume_records": [],
            "anomaly_dates": [],
            "anomaly_count": 0,
            "issues": [],
        }

    counts = daily_counts["request_count"].astype(float)
    history_mean = counts.shift(1).rolling(rolling_window, min_periods=minimum_periods).mean()
    history_std = counts.shift(1).rolling(rolling_window, min_periods=minimum_periods).std(ddof=0)
    standardized = pd.Series(np.nan, index=daily_counts.index, dtype=float)
    usable_history = history_mean.notna() & history_std.notna() & history_std.gt(0)
    standardized.loc[usable_history] = (
        (counts.loc[usable_history] - history_mean.loc[usable_history])
        / history_std.loc[usable_history]
    )
    increase = counts - history_mean
    anomaly_mask = pd.Series(False, index=daily_counts.index)
    if detect_increases:
        anomaly_mask |= usable_history & increase.ge(minimum_absolute_increase) & standardized.ge(
            standardized_threshold
        )
    if detect_decreases:
        anomaly_mask |= usable_history & increase.le(-minimum_absolute_increase) & standardized.le(
            -standardized_threshold
        )

    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, row in daily_counts.iterrows():
        record = {
            "calendar_date": row["calendar_date"],
            "request_count": int(row["request_count"]),
            "rolling_mean": None if pd.isna(history_mean.loc[index]) else float(history_mean.loc[index]),
            "rolling_standard_deviation": None if pd.isna(history_std.loc[index]) else float(history_std.loc[index]),
            "rolling_z_score": None if pd.isna(standardized.loc[index]) else float(standardized.loc[index]),
            "history_sufficient": bool(usable_history.loc[index]),
            "is_anomaly": bool(anomaly_mask.loc[index]),
        }
        records.append(record)
        if record["is_anomaly"]:
            issues.append(
                {
                    **build_volume_issue(
                        {
                            "calendar_date": record["calendar_date"],
                            "baseline_count": record["rolling_mean"],
                            "current_count": record["request_count"],
                            "absolute_difference": abs(record["request_count"] - record["rolling_mean"]),
                            "percentage_change": None,
                            "standardized_count_difference": record["rolling_z_score"],
                        },
                        configuration,
                    ),
                    "description": "Daily request volume exceeds the configured rolling-history threshold.",
                }
            )
    anomaly_dates = [record["calendar_date"] for record in records if record["is_anomaly"]]
    return {
        "detector_name": "rolling_daily_volume",
        "configuration": configuration,
        "daily_volume_records": records,
        "anomaly_dates": anomaly_dates,
        "anomaly_count": len(anomaly_dates),
        "issues": issues,
    }


def compare_baseline_current_volumes(
    baseline_dataframe: pd.DataFrame,
    current_dataframe: pd.DataFrame,
    standardized_threshold: float = DEFAULT_STANDARDIZED_THRESHOLD,
    minimum_absolute_increase: int = DEFAULT_MIN_ABSOLUTE_INCREASE,
    detect_increases: bool = True,
    detect_decreases: bool = False,
) -> dict[str, Any]:
    """Compare baseline and current daily counts using deterministic standardized differences."""
    configuration = {
        "standardized_threshold": standardized_threshold,
        "minimum_absolute_increase": minimum_absolute_increase,
        "detect_increases": detect_increases,
        "detect_decreases": detect_decreases,
        "standardization": "difference divided by square root of baseline_count",
    }
    baseline_counts = aggregate_daily_request_counts(baseline_dataframe).rename(
        columns={"request_count": "baseline_count"}
    )
    current_counts = aggregate_daily_request_counts(current_dataframe).rename(
        columns={"request_count": "current_count"}
    )
    comparison = baseline_counts.merge(current_counts, on="calendar_date", how="outer").fillna(0)
    if comparison.empty:
        return {
            "detector_name": "baseline_current_daily_volume",
            "configuration": configuration,
            "daily_volume_records": [],
            "anomaly_dates": [],
            "anomaly_count": 0,
            "issues": [],
        }

    comparison = comparison.sort_values("calendar_date").reset_index(drop=True)
    comparison["baseline_count"] = comparison["baseline_count"].astype(int)
    comparison["current_count"] = comparison["current_count"].astype(int)
    difference = comparison["current_count"] - comparison["baseline_count"]
    comparison["absolute_difference"] = difference.abs()
    comparison["percentage_change"] = np.where(
        comparison["baseline_count"].gt(0),
        difference / comparison["baseline_count"] * 100,
        np.nan,
    )
    comparison["standardized_count_difference"] = difference / np.sqrt(
        comparison["baseline_count"].clip(lower=1)
    )
    anomaly_mask = pd.Series(False, index=comparison.index)
    if detect_increases:
        anomaly_mask |= difference.ge(minimum_absolute_increase) & comparison[
            "standardized_count_difference"
        ].ge(standardized_threshold)
    if detect_decreases:
        anomaly_mask |= difference.le(-minimum_absolute_increase) & comparison[
            "standardized_count_difference"
        ].le(-standardized_threshold)

    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, row in comparison.iterrows():
        record = {
            "calendar_date": row["calendar_date"],
            "baseline_count": int(row["baseline_count"]),
            "current_count": int(row["current_count"]),
            "absolute_difference": int(row["absolute_difference"]),
            "percentage_change": None if pd.isna(row["percentage_change"]) else float(row["percentage_change"]),
            "standardized_count_difference": float(row["standardized_count_difference"]),
            "is_anomaly": bool(anomaly_mask.loc[index]),
        }
        records.append(record)
        if record["is_anomaly"]:
            issues.append(build_volume_issue(record, configuration))
    anomaly_dates = [record["calendar_date"] for record in records if record["is_anomaly"]]
    return {
        "detector_name": "baseline_current_daily_volume",
        "configuration": configuration,
        "daily_volume_records": records,
        "anomaly_dates": anomaly_dates,
        "anomaly_count": len(anomaly_dates),
        "issues": issues,
    }


def main() -> None:
    """Compare the clean raw dataset with the deterministic daily volume spike scenario."""
    if not VOLUME_SPIKE_PATH.exists():
        run_fault_injection()
    clean_dataframe = load_clean_dataset()
    spike_dataframe = pd.read_csv(VOLUME_SPIKE_PATH)
    comparison = compare_baseline_current_volumes(clean_dataframe, spike_dataframe)
    rolling = detect_rolling_volume_anomalies(spike_dataframe)
    print("Daily volume comparison")
    for record in comparison["daily_volume_records"]:
        if record["absolute_difference"] or record["is_anomaly"]:
            print(record)
    print(f"Anomaly dates: {comparison['anomaly_dates']}")
    print("Rolling volume output:")
    for record in rolling["daily_volume_records"]:
        print(record)
    print(f"Rolling anomaly dates: {rolling['anomaly_dates']}")


if __name__ == "__main__":
    main()
