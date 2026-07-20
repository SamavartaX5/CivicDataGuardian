"""Deterministic fault-injector tests."""

from __future__ import annotations

import pandas as pd

from src import fault_injection as faults


REQUIRED_METADATA = {
    "scenario_name", "fault_type", "expected_issue_types", "affected_columns", "affected_row_indices",
    "affected_row_count", "parameters", "random_seed", "description", "output_path",
}


def test_injectors_return_new_frames_and_preserve_clean_input(valid_dataframe) -> None:
    original = valid_dataframe.copy(deep=True)
    outputs = [function(valid_dataframe)[0] for function in faults.SCENARIO_FUNCTIONS]

    pd.testing.assert_frame_equal(valid_dataframe, original)
    assert all(output is not valid_dataframe for output in outputs)
    assert len({id(output) for output in outputs}) == len(outputs)


def test_schema_and_value_faults_modify_expected_data(valid_dataframe) -> None:
    deleted, deleted_metadata = faults.inject_deleted_agency(valid_dataframe)
    renamed, _ = faults.inject_renamed_complaint_type(valid_dataframe)
    latitude, latitude_metadata = faults.inject_invalid_latitude_text(valid_dataframe, affected_count=3)
    timestamp, timestamp_metadata = faults.inject_invalid_created_date(valid_dataframe, affected_count=3)

    assert "agency" not in deleted and REQUIRED_METADATA.issubset(deleted_metadata)
    assert "complaint_type" not in renamed and "complaint_category" in renamed
    assert latitude.loc[latitude_metadata["affected_row_indices"], "latitude"].eq("invalid-latitude").all()
    assert timestamp.loc[timestamp_metadata["affected_row_indices"], "created_date"].eq("invalid-created-date").all()


def test_duplicates_missingness_and_dates(valid_dataframe) -> None:
    duplicated, duplicate_metadata = faults.inject_duplicate_rows(valid_dataframe, affected_count=4)
    missing_10, _ = faults.inject_descriptor_missingness_plus_10(valid_dataframe)
    missing_25, _ = faults.inject_descriptor_missingness_plus_25(valid_dataframe)
    missing_50, _ = faults.inject_descriptor_missingness_plus_50(valid_dataframe)
    reversed_dates, reversed_metadata = faults.inject_reversed_closed_dates(valid_dataframe, affected_count=3)

    assert len(duplicated) == len(valid_dataframe) + duplicate_metadata["affected_row_count"]
    assert duplicated["unique_key"].duplicated(keep=False).any()
    percentages = [frame["descriptor"].isna().mean() for frame in (missing_10, missing_25, missing_50)]
    assert percentages[0] < percentages[1] < percentages[2]
    created = pd.to_datetime(reversed_dates.loc[reversed_metadata["affected_row_indices"], "created_date"])
    closed = pd.to_datetime(reversed_dates.loc[reversed_metadata["affected_row_indices"], "closed_date"])
    assert (closed < created).all()


def test_volume_and_category_faults(valid_dataframe) -> None:
    spike, spike_metadata = faults.inject_daily_volume_spike(valid_dataframe, added_count=200)
    shifted, shifted_metadata = faults.inject_complaint_type_frequency_shift(valid_dataframe, affected_count=5)
    unseen, unseen_metadata = faults.inject_unseen_complaint_category(valid_dataframe, affected_count=5)

    appended_keys = spike["unique_key"].iloc[-spike_metadata["affected_row_count"]:]
    assert appended_keys.is_unique and not appended_keys.isin(valid_dataframe["unique_key"]).any()
    target = shifted_metadata["parameters"]["target_category"]
    assert shifted["complaint_type"].eq(target).mean() > valid_dataframe["complaint_type"].eq(target).mean()
    unseen_category = unseen_metadata["parameters"]["unseen_category"]
    assert not valid_dataframe["complaint_type"].eq(unseen_category).any()
    assert unseen["complaint_type"].eq(unseen_category).sum() == unseen_metadata["affected_row_count"]
