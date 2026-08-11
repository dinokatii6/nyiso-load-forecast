"""
Builds the model-ready feature matrix.

"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from src.config import HORIZON_HOURS, LOCAL_TZ, PROCESSED_DIR

IN_PATH = PROCESSED_DIR / "hourly.parquet"
OUT_PATH = PROCESSED_DIR / "features.parquet"

TARGET = "demand_mw"
BASE_TEMP_F = 65.0


LAGS = [24, 25, 26, 27, 48, 72, 96, 168, 169, 192, 336]

ROLLING = [(24, 24), (168, 24)]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Clock and calendar features, derived from LOCAL time."""
    t = df["timestamp_local"]

    df["hour"] = t.dt.hour
    df["dayofweek"] = t.dt.dayofweek # Monday starts at 0, Sunday = 6
    df["month"] = t.dt.month
    df["dayofyear"] = t.dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Captures multi-year drift from efficiency gains,
    # rooftop solar, electrification. Measured in years since the start.
    df["year_frac"] = (df.index - df.index.min()).total_seconds() / (365.25 * 24 * 3600)

    # Puts each periodic variable on a circle.
    for col, period in [("hour", 24), ("dayofweek", 7), ("dayofyear", 365.25)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    return df


def add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """US + New York State holidays, plus the days around them."""
    years = sorted(df["timestamp_local"].dt.year.unique())
    try:
        cal = holidays.country_holidays("US", subdiv="NY", years=years)
    except TypeError:                    
        cal = holidays.US(state="NY", years=years)

    local_date = df["timestamp_local"].dt.date
    df["is_holiday"] = local_date.map(lambda d: d in cal).astype(int)

    # The day before Thanksgiving and the days between
    # Christmas and New Year behave unlike normal weekdays.
    daily = df.groupby(local_date)["is_holiday"].max()
    df["is_day_before_holiday"] = local_date.map(daily.shift(-1).fillna(0)).astype(int)
    df["is_day_after_holiday"] = local_date.map(daily.shift(1).fillna(0)).astype(int)

    return df


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Degree days and thermal-inertia features."""
    df["heating_degrees"] = np.maximum(BASE_TEMP_F - df["temp_f"], 0.0)
    df["cooling_degrees"] = np.maximum(df["temp_f"] - BASE_TEMP_F, 0.0)


    # AC load accelerates as it gets hotter (more units switch on, all run longer).
    df["cooling_degrees_sq"] = df["cooling_degrees"] ** 2
    df["heating_degrees_sq"] = df["heating_degrees"] ** 2

    #Buildings have mass. A single hot hour after a cool
    # week loads the grid less than the fifth day of a heatwave.
    df["temp_24h_mean"] = df["temp_f"].rolling(24, min_periods=12).mean()
    df["temp_72h_mean"] = df["temp_f"].rolling(72, min_periods=36).mean()

    # Cooling load is concentrated in daylight/afternoon hours.
    df["cdd_x_hour_sin"] = df["cooling_degrees"] * df["hour_sin"]
    df["cdd_x_hour_cos"] = df["cooling_degrees"] * df["hour_cos"]

    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Past values of the target."""
    s = df[TARGET]

    for lag in LAGS:
        assert lag >= HORIZON_HOURS, f"lag_{lag} violates the {HORIZON_HOURS}h cutoff"
        df[f"lag_{lag}"] = s.shift(lag)

    for window, shift in ROLLING:
        assert shift >= HORIZON_HOURS
        base = s.shift(shift)
        df[f"roll_mean_{window}h"] = base.rolling(window, min_periods=window // 2).mean()
        df[f"roll_std_{window}h"] = base.rolling(window, min_periods=window // 2).std()

    # Recent week-over-week momentum from legal history
    df["lag_24_minus_lag_168"] = df["lag_24"] - df["lag_168"]

    return df


def build() -> pd.DataFrame:
    df = pd.read_parquet(IN_PATH).copy()

    df = add_calendar(df)
    df = add_holidays(df)
    df = add_weather(df)
    df = add_lags(df)

    before = len(df)
    df = df.dropna()  # drops the warm-up period and any residual gaps
    print(f"dropped {before - len(df):,} rows with missing values "
          f"({before:,} -> {len(df):,})")

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except the target and the raw timestamp."""
    exclude = {TARGET, "timestamp_local"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    out = build()
    out.to_parquet(OUT_PATH)
    print(f"\nshape: {out.shape}")
    print(f"features: {len(feature_columns(out))}")
    print(f"range: {out.index.min()} -> {out.index.max()}")
    print(f"\nsaved -> {OUT_PATH}")