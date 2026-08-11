"""Temporal splitting, metrics, and naive baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TEST_MONTHS

VAL_MONTHS = 12


def temporal_split(df: pd.DataFrame,
                   val_months: int = VAL_MONTHS,
                   test_months: int = TEST_MONTHS):
    """Chronological three-way split."""
    end = df.index.max()
    test_start = end - pd.DateOffset(months=test_months)
    val_start = test_start - pd.DateOffset(months=val_months)

    train = df.loc[df.index < val_start]
    val = df.loc[(df.index >= val_start) & (df.index < test_start)]
    test = df.loc[df.index >= test_start]

    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:>6}: {len(part):>7,} rows   "
              f"{part.index.min().date()} -> {part.index.max().date()}")

    return train, val, test


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mape(y, yhat):
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def score(name: str, y, yhat) -> dict:
    return {"model": name, "MAE_MW": mae(y, yhat),
            "RMSE_MW": rmse(y, yhat), "MAPE_pct": mape(y, yhat)}


def results_table(rows: list[dict], baseline_name: str | None = None) -> pd.DataFrame:
    """Sorted results, with improvement over the baseline if one is named."""
    out = pd.DataFrame(rows).sort_values("MAE_MW").reset_index(drop=True)
    if baseline_name is not None:
        base = out.loc[out["model"] == baseline_name, "MAE_MW"].iloc[0]
        out["vs_baseline_pct"] = (1 - out["MAE_MW"] / base) * 100
    return out.round(3)