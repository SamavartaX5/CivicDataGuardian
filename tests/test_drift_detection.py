"""Complaint-category and borough distribution drift tests."""

from __future__ import annotations

import pandas as pd

from src.drift_detection import detect_distribution_drift
from src.fault_injection import (
    inject_complaint_type_frequency_shift,
    inject_daily_volume_spike,
    inject_duplicate_rows,
    inject_unseen_complaint_category,
)


def detected_types(result: dict) -> set[str]:
    return {issue["issue_type"] for issue in result["issues"]}


def test_drift_detector_finds_shift_and_unseen_categories(valid_dataframe) -> None:
    shifted, _ = inject_complaint_type_frequency_shift(valid_dataframe, affected_count=5)
    unseen, _ = inject_unseen_complaint_category(valid_dataframe, affected_count=10)

    assert "category_drift" in detected_types(detect_distribution_drift(valid_dataframe, shifted))
    assert "unexpected_category" in detected_types(detect_distribution_drift(valid_dataframe, unseen))


def test_clean_duplicate_volume_and_missing_column_are_safe(valid_dataframe) -> None:
    original = valid_dataframe.copy(deep=True)
    duplicate, _ = inject_duplicate_rows(valid_dataframe, affected_count=4)
    spike, _ = inject_daily_volume_spike(valid_dataframe, added_count=200)
    clean = detect_distribution_drift(valid_dataframe, valid_dataframe)
    duplicate_result = detect_distribution_drift(valid_dataframe, duplicate)
    spike_result = detect_distribution_drift(valid_dataframe, spike)
    missing_result = detect_distribution_drift(valid_dataframe, valid_dataframe.drop(columns=["complaint_type"]))

    assert not detected_types(clean)
    assert not detected_types(duplicate_result)
    assert not detected_types(spike_result)
    assert missing_result["field_results"]["complaint_type"]["field_available"] is False
    pd.testing.assert_frame_equal(valid_dataframe, original)
