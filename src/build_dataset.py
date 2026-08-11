"""Joins EIA demand with population-weighted NY weather into a table of hourly observations, filling short gaps with interpolation.
"""
from __future__ import annotations

import pandas as pd

from src.config import BA_CODE, LOCAL_TZ, PROCESSED_DIR, RAW_DIR, WEATHER_SITES

DEMAND_PATH = RAW_DIR / f"eia_demand_{BA_CODE}.csv"
WEATHER_PATH = RAW_DIR / "weather_sites.csv"
OUT_PATH = PROCESSED_DIR / "hourly.parquet"

WEATHER_COLS = ["temp_f", "humidity_pct", "wind_mph", "cloud_pct"]

MAX_INTERPOLATE_HOURS = 3   # fills gaps up to three hours long, leave longer ones as NaN


def load_demand() -> pd.Series:
    """Reads the raw demand CSV and flags physically impossible values."""
    df = pd.read_csv(DEMAND_PATH)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    n_bad = int((df["demand_mw"] <= 0).sum())
    if n_bad:
        print(f"flagged {n_bad} non-positive demand values as missing")
    df.loc[df["demand_mw"] <= 0, "demand_mw"] = pd.NA

    return df.set_index("timestamp_utc")["demand_mw"].astype("float64")


def load_weather() -> pd.DataFrame:
    """Weighted statewide series PLUS per-city temperature, so the model
    can learn its own effective weighting instead of trusting our guess."""
    w = pd.read_csv(WEATHER_PATH)
    w["timestamp_utc"] = pd.to_datetime(w["timestamp_utc"], utc=True)

    weights = pd.Series({k: v["weight"] for k, v in WEATHER_SITES.items()})

    frames = {}
    for col in WEATHER_COLS:
        wide = w.pivot(index="timestamp_utc", columns="site", values=col)
        wide = wide[weights.index]

        # statewide weighted average (interpretability, charts)
        frames[col] = wide.mul(weights, axis=1).sum(axis=1, min_count=1)

        # per-city temperature (the model learns the real weighting)
        if col == "temp_f":
            for site in weights.index:
                frames[f"temp_f_{site}"] = wide[site]

    return pd.DataFrame(frames)


def main() -> None:
    demand = load_demand()
    weather = load_weather()

    # Builds a complete hourly index. Any hour missing from the source data shows up as a NaN row
    full_index = pd.date_range(demand.index.min(), demand.index.max(),
                               freq="h", tz="UTC")
    df = pd.DataFrame(index=full_index)
    df.index.name = "timestamp_utc"

    df["demand_mw"] = demand.reindex(full_index)
    df = df.join(weather)   # joins on the index, which is the timestamp

    print(f"\nrows: {len(df):,}")
    print("missing before interpolation:")
    print(df.isna().sum())

    # Interpolate short gaps only.
    df = df.interpolate(method="time", limit=MAX_INTERPOLATE_HOURS,
                        limit_direction="forward")

    print("\nmissing after interpolation:")
    print(df.isna().sum())

    # Local calendar time
    df["timestamp_local"] = df.index.tz_convert(LOCAL_TZ)

    profile = df.groupby(df["timestamp_local"].dt.hour)["demand_mw"].mean()
    peak_hour = int(profile.idxmax())
    trough_hour = int(profile.idxmin())
    print(f"\npeak local hour: {peak_hour}   trough local hour: {trough_hour}")
    if not (14 <= peak_hour <= 21):
        print("WARNING: peak hour is not in the expected afternoon/evening window. "
              "Check timezone handling before continuing.")

    df.to_parquet(OUT_PATH)
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()