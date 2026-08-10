"""Downloads hourly historical weather for New York sites from Open-Meteo.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from src.config import END_DATE, RAW_DIR, START_DATE, WEATHER_SITES

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OUT_PATH = RAW_DIR / "weather_sites.csv"

# Maps Open-Meteo variable names to our column names
VAR_MAP = {
    "temperature_2m": "temp_f",
    "relative_humidity_2m": "humidity_pct",
    "wind_speed_10m": "wind_mph",
    "cloud_cover": "cloud_pct",
}


def fetch_site(name: str, lat: float, lon: float) -> pd.DataFrame:
    """Fetches the full date range of hourly weather for one location."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(VAR_MAP.keys()),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",          # matches the EIA data so the join works
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=180)
    resp.raise_for_status()

    hourly = resp.json()["hourly"]   # dict of parallel lists: time, temperature_2m, etc
    df = pd.DataFrame(hourly)

    df["timestamp_utc"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop(columns=["time"]).rename(columns=VAR_MAP)

    # Changes values to numeric and sets null values to NAN
    for col in VAR_MAP.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.insert(0, "site", name)
    return df


def main() -> None:
    frames = []
    for name, meta in WEATHER_SITES.items():
        print(f"fetching {name} ...")
        frames.append(fetch_site(name, meta["lat"], meta["lon"]))
        time.sleep(1.0)   

    out = pd.concat(frames, ignore_index=True).sort_values(["site", "timestamp_utc"])

    print(f"\nrows: {len(out):,}   sites: {out['site'].nunique()}")
    print(out.groupby("site")["temp_f"].describe()[["min", "mean", "max"]])

    out.to_csv(OUT_PATH, index=False)
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()