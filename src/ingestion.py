"""Fetch recent NYC 311 service requests from the Socrata API."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import pandas as pd
import requests


LOGGER = logging.getLogger(__name__)
DATASET_ID = "erm2-nwe9"
SOURCE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
DEFAULT_LIMIT = 5_000
DEFAULT_TIMEOUT_SECONDS = 30
REQUESTED_COLUMNS = (
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "complaint_type",
    "descriptor",
    "status",
    "borough",
    "incident_zip",
    "latitude",
    "longitude",
)
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


class IngestionError(RuntimeError):
    """Raised when NYC 311 data cannot be downloaded or decoded."""


class IngestionMetadata(TypedDict):
    """Metadata produced by a successful ingestion run."""

    source_url: str
    rows_downloaded: int
    columns_downloaded: int
    column_names: list[str]
    latest_created_date: str | None
    stable_output_path: str
    snapshot_output_path: str
    download_runtime_seconds: float


def fetch_nyc_311_data(
    limit: int = DEFAULT_LIMIT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Fetch recent NYC 311 records as a DataFrame without changing datatypes."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer.")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout must be a positive integer.")

    params = {
        "$select": ",".join(REQUESTED_COLUMNS),
        "$order": "created_date DESC",
        "$limit": limit,
    }

    LOGGER.info("Fetching up to %s recent records from NYC 311.", limit)
    try:
        response = requests.get(SOURCE_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as error:
        raise IngestionError(f"NYC 311 request failed: {error}") from error

    try:
        payload = response.json()
    except ValueError as error:
        raise IngestionError("NYC 311 API returned malformed JSON.") from error

    if not isinstance(payload, list):
        raise IngestionError("NYC 311 API returned JSON in an unexpected format.")
    if not payload:
        raise IngestionError("NYC 311 API returned no records.")

    dataframe = pd.DataFrame.from_records(payload)
    available_columns = [column for column in REQUESTED_COLUMNS if column in dataframe.columns]
    dataframe = dataframe.loc[:, available_columns]
    LOGGER.info("Downloaded %s NYC 311 records.", len(dataframe))
    return dataframe


def save_snapshots(
    dataframe: pd.DataFrame,
    output_directory: Path = RAW_DATA_DIR,
) -> tuple[Path, Path]:
    """Save stable and timestamped CSV copies of downloaded records."""
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stable_path = output_directory / "nyc_311_recent.csv"
    snapshot_path = output_directory / f"nyc_311_{timestamp}.csv"

    dataframe.to_csv(stable_path, index=False)
    dataframe.to_csv(snapshot_path, index=False)
    LOGGER.info("Saved stable CSV to %s.", stable_path)
    LOGGER.info("Saved timestamped CSV to %s.", snapshot_path)
    return stable_path, snapshot_path


def run_ingestion(
    limit: int = DEFAULT_LIMIT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    output_directory: Path = RAW_DATA_DIR,
) -> tuple[pd.DataFrame, IngestionMetadata]:
    """Download NYC 311 data, write snapshots, and return data with metadata."""
    started_at = time.perf_counter()
    dataframe = fetch_nyc_311_data(limit=limit, timeout=timeout)
    download_runtime_seconds = round(time.perf_counter() - started_at, 3)
    stable_path, snapshot_path = save_snapshots(dataframe, output_directory)

    latest_created_date = None
    if "created_date" in dataframe.columns:
        created_dates = dataframe["created_date"].dropna()
        if not created_dates.empty:
            latest_created_date = str(created_dates.iloc[0])

    metadata: IngestionMetadata = {
        "source_url": SOURCE_URL,
        "rows_downloaded": len(dataframe),
        "columns_downloaded": len(dataframe.columns),
        "column_names": list(dataframe.columns),
        "latest_created_date": latest_created_date,
        "stable_output_path": str(stable_path),
        "snapshot_output_path": str(snapshot_path),
        "download_runtime_seconds": download_runtime_seconds,
    }
    return dataframe, metadata


def main() -> None:
    """Run ingestion and print its requested execution summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        dataframe, metadata = run_ingestion()
    except IngestionError as error:
        LOGGER.error("Ingestion failed: %s", error)
        raise SystemExit(1) from error

    print(f"DataFrame shape: {dataframe.shape}")
    print(f"Column names: {list(dataframe.columns)}")
    print(f"Ingestion metadata: {metadata}")


if __name__ == "__main__":
    main()
