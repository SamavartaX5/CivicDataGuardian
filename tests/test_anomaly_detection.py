"""Daily-volume anomaly detection tests."""

from __future__ import annotations

import pandas as pd

from src.anomaly_detection import compare_baseline_current_volumes, detect_rolling_volume_anomalies
from src.fault_injection import inject_daily_volume_spike, inject_duplicate_rows


def test_volume_comparisons_detect_only_large_spike(valid_dataframe) -> None:
    clean_result = compare_baseline_current_volumes(valid_dataframe, valid_dataframe)
    duplicate, _ = inject_duplicate_rows(valid_dataframe, affected_count=4)
    duplicate_result = compare_baseline_current_volumes(valid_dataframe, duplicate)
    spike, _ = inject_daily_volume_spike(valid_dataframe, added_count=200)
    spike_result = compare_baseline_current_volumes(valid_dataframe, spike)

    assert clean_result["anomaly_count"] == 0
    assert duplicate_result["anomaly_count"] == 0
    assert [issue["issue_type"] for issue in spike_result["issues"]] == ["volume_anomaly"]


def test_rolling_sparse_and_missing_dates_are_safe_and_inputs_unchanged(valid_dataframe) -> None:
    original = valid_dataframe.copy(deep=True)
    rolling = detect_rolling_volume_anomalies(valid_dataframe)
    missing_column = detect_rolling_volume_anomalies(valid_dataframe.drop(columns=["created_date"]))

    assert rolling["anomaly_count"] == 0
    assert all(not record["history_sufficient"] for record in rolling["daily_volume_records"])
    assert missing_column["daily_volume_records"] == []
    pd.testing.assert_frame_equal(valid_dataframe, original)
