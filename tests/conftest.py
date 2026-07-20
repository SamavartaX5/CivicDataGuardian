"""Shared deterministic fixtures for CivicData Guardian tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def schema() -> dict:
    """Load the real NYC 311 schema without modifying it."""
    with (PROJECT_ROOT / "config" / "schema.yaml").open(encoding="utf-8") as schema_file:
        return yaml.safe_load(schema_file)


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    """Return a small valid NYC 311 DataFrame with repeatable distributions."""
    categories = [
        "Noise - Street/Sidewalk",
        "Noise - Street/Sidewalk",
        "Noise - Street/Sidewalk",
        "Noise - Street/Sidewalk",
        "Noise - Street/Sidewalk",
        "Illegal Parking",
        "Illegal Parking",
        "Illegal Parking",
        "HEAT/HOT WATER",
        "HEAT/HOT WATER",
    ]
    boroughs = [
        "MANHATTAN",
        "MANHATTAN",
        "MANHATTAN",
        "MANHATTAN",
        "BROOKLYN",
        "BROOKLYN",
        "BROOKLYN",
        "QUEENS",
        "QUEENS",
        "QUEENS",
    ]
    rows = []
    for day_offset, day in enumerate(("2026-01-01", "2026-01-02")):
        for position, (category, borough) in enumerate(zip(categories, boroughs, strict=True)):
            row_number = day_offset * len(categories) + position
            rows.append(
                {
                    "unique_key": 10_000 + row_number,
                    "created_date": f"{day}T10:{position:02d}:00.000",
                    "closed_date": f"{day}T12:{position:02d}:00.000",
                    "agency": "NYPD",
                    "complaint_type": category,
                    "descriptor": "Fixture descriptor",
                    "status": "Closed",
                    "borough": borough,
                    "incident_zip": "10001",
                    "latitude": 40.70 + position / 1000,
                    "longitude": -73.99 - position / 1000,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def issue_factory():
    """Return a factory for incident-shaped validation issues."""
    def make(issue_type: str, **overrides: object) -> dict:
        issue = {
            "issue_type": issue_type,
            "severity": "error",
            "affected_field": "fixture_field",
            "description": "Fixture issue.",
            "observed_value": {"count": 1},
            "expected_value": "expected value",
            "suggested_action": "Correct the fixture.",
        }
        issue.update(overrides)
        return issue

    return make
