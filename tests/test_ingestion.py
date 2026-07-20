"""Offline tests for NYC 311 ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from src import ingestion


class FakeResponse:
    """Minimal requests response double."""

    def __init__(self, payload=None, json_error: Exception | None = None) -> None:
        self.payload = payload
        self.json_error = json_error

    def raise_for_status(self) -> None:
        return None

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload


def test_fetch_uses_socrata_params_and_returns_payload_dataframe(monkeypatch) -> None:
    captured = {}
    payload = [{"unique_key": "1", "created_date": "2026-01-01T00:00:00.000", "agency": "NYPD"}]

    def fake_get(url, params, timeout):
        captured.update({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(payload)

    monkeypatch.setattr(ingestion.requests, "get", fake_get)
    dataframe = ingestion.fetch_nyc_311_data(limit=7, timeout=11)

    assert dataframe.to_dict("records") == payload
    assert captured["url"] == ingestion.SOURCE_URL
    assert captured["params"]["$select"] == ",".join(ingestion.REQUESTED_COLUMNS)
    assert captured["params"]["$order"] == "created_date DESC"
    assert captured["params"]["$limit"] == 7
    assert captured["timeout"] == 11


def test_fetch_raises_for_malformed_empty_and_failed_requests(monkeypatch) -> None:
    monkeypatch.setattr(ingestion.requests, "get", lambda *_, **__: FakeResponse(json_error=ValueError()))
    with pytest.raises(ingestion.IngestionError, match="malformed JSON"):
        ingestion.fetch_nyc_311_data()

    monkeypatch.setattr(ingestion.requests, "get", lambda *_, **__: FakeResponse([]))
    with pytest.raises(ingestion.IngestionError, match="no records"):
        ingestion.fetch_nyc_311_data()

    def failed_get(*_, **__):
        raise requests.RequestException("offline")

    monkeypatch.setattr(ingestion.requests, "get", failed_get)
    with pytest.raises(ingestion.IngestionError, match="request failed"):
        ingestion.fetch_nyc_311_data()


def test_save_snapshots_writes_both_csvs(valid_dataframe, tmp_path: Path) -> None:
    stable_path, snapshot_path = ingestion.save_snapshots(valid_dataframe, tmp_path)

    assert stable_path == tmp_path / "nyc_311_recent.csv"
    assert stable_path.exists() and snapshot_path.exists()
    assert snapshot_path.name.startswith("nyc_311_")
