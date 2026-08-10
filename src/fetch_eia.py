"""Downloads hourly NYISO electricity demand from the EIA API v2 and cached it to CSV.
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import BA_CODE, EIA_BASE_URL, END_DATE, LOCAL_TZ, RAW_DIR, START_DATE

PAGE_SIZE = 5000  # EIA's maximum rows per request
OUT_PATH = RAW_DIR / f"eia_demand_{BA_CODE}.csv"


def build_params(api_key: str, offset: int) -> dict:
    """Assembles the query parameters for one page of results."""
    return {
        "api_key": api_key,
        "frequency": "hourly",              # hourly returns UTC timestamps
        "data[0]": "value",                 # which measurement column we want
        "facets[respondent][]": BA_CODE,    # filter to NYISO
        "facets[type][]": "D",              # D = Demand (NG = net generation, TI = interchange)
        "start": f"{START_DATE}T00",
        "end": f"{END_DATE}T23",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": offset,
        "length": PAGE_SIZE,
    }


def fetch_all(api_key: str) -> list[dict]:
    """Pages through the API until we have every row in the date range."""
    rows: list[dict] = []
    offset = 0

    while True:
        resp = requests.get(EIA_BASE_URL, params=build_params(api_key, offset), timeout=60)
        resp.raise_for_status()         
        payload = resp.json()

        if "response" not in payload:    
            sys.exit(f"Unexpected API response:\n{payload}")

        body = payload["response"]
        batch = body["data"]
        total = int(body.get("total", 0))
        rows.extend(batch)

        print(f"  fetched {len(rows):>6,} / {total:,} rows")

        if len(batch) < PAGE_SIZE or len(rows) >= total:
            break

        offset += PAGE_SIZE
        time.sleep(0.3)   

    return rows


def to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Turns the raw JSON records into a two-column cleaned DataFrame."""
    df = pd.DataFrame(rows)

    df["timestamp_utc"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True)
    
    df["demand_mw"] = pd.to_numeric(df["value"], errors="coerce")

    return (
        df[["timestamp_utc", "demand_mw"]]
        .drop_duplicates(subset="timestamp_utc")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )


def summarise(df: pd.DataFrame) -> None:
    """Prints sanity checks, including an empirical timezone verification."""
    print(f"\nrows:      {len(df):,}")
    print(f"range:     {df['timestamp_utc'].min()}  ->  {df['timestamp_utc'].max()}")
    print(f"missing:   {df['demand_mw'].isna().sum():,}")

    expected = int((df["timestamp_utc"].max() - df["timestamp_utc"].min())
                   .total_seconds() // 3600) + 1
    print(f"expected hours: {expected:,}   gaps: {expected - len(df):,}")

    # Timezone check: convert to New York local time and find average demand
    # per local hour. Electricity demand peaks in the late afternoon/evening.
    local_hour = df["timestamp_utc"].dt.tz_convert(LOCAL_TZ).dt.hour
    profile = df.groupby(local_hour)["demand_mw"].mean().sort_values(ascending=False)
    print(f"\nhighest-demand local hours: {list(profile.index[:3])}")
    print(f"lowest-demand local hours:  {list(profile.index[-3:])}")


def main() -> None:
    load_dotenv()  
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        sys.exit("EIA_API_KEY not found. Create a .env file at the repo root.")

    print(f"Fetching {BA_CODE} demand, {START_DATE} to {END_DATE} ...")
    df = to_dataframe(fetch_all(api_key))
    summarise(df)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()