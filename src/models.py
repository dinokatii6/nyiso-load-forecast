"""Trains Ridge and gradient boosting and evaluates."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR
from src.evaluate import results_table, score, temporal_split
from src.features import TARGET, feature_columns


def build_ridge(alpha: float = 10.0) -> Pipeline:
    """Scaling before Ridge"""
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def build_gbm() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,   # see note: sklearn's internal split is random
        random_state=42,
    )


def train_all() -> dict:
    """Fit both models and print validation results."""
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    train, val, test = temporal_split(df)

    feats = feature_columns(df)
    X_tr, y_tr = train[feats], train[TARGET]
    X_va, y_va = val[feats], val[TARGET]

    rows = [
        score("Naive-24", y_va, val["lag_24"]),
        score("Naive-168", y_va, val["lag_168"]),
    ]

    ridge = build_ridge().fit(X_tr, y_tr)
    rows.append(score("Ridge", y_va, ridge.predict(X_va)))

    gbm = build_gbm().fit(X_tr, y_tr)
    rows.append(score("GradientBoosting", y_va, gbm.predict(X_va)))

    # Whichever naive baseline is harder becomes the bar to beat.
    baseline_name = min(rows[:2], key=lambda r: r["MAE_MW"])["model"]

    table = results_table(rows, baseline_name=baseline_name)
    print(f"\nValidation results (baseline: {baseline_name}):\n")
    print(table.to_string(index=False))

    return {
        "ridge": ridge,
        "gbm": gbm,
        "train": train,
        "val": val,
        "test": test,
        "feats": feats,
        "results": table,
    }


def main() -> None:
    train_all()


if __name__ == "__main__":
    main()