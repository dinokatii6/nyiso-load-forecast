"""
Trains the PyTorch MLP, records learning curves, and checks for overfitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import FIG_DIR, PROCESSED_DIR
from src.evaluate import mae, temporal_split
from src.features import TARGET, feature_columns

torch.manual_seed(42)
np.random.seed(42)


def build_mlp(n_features: int, n_hidden: int = 128, dropout: float = 0.1) -> nn.Module:
    """Same 2-layer shape as the original mlp with dropout added.
    """
    return nn.Sequential(
        nn.Linear(n_features, n_hidden),
        nn.Tanh(),
        nn.Dropout(dropout),
        nn.Linear(n_hidden, 1),
    )


def prepare(train, val, test, feats):
    """Scales features and target using train statistics."""
    x_scaler = StandardScaler().fit(train[feats])
    y_scaler = StandardScaler().fit(train[[TARGET]])

    def to_tensor(part):
        X = torch.tensor(x_scaler.transform(part[feats]), dtype=torch.float32)
        y = torch.tensor(y_scaler.transform(part[[TARGET]]), dtype=torch.float32)
        return X, y

    # y_scaler converts predictions back to megawatts
    return to_tensor(train), to_tensor(val), to_tensor(test), x_scaler, y_scaler


def train_model(model, train_t, val_t, y_scaler,
                epochs=60, lr=1e-3, batch_size=512, weight_decay=1e-4):
    X_tr, y_tr = train_t
    X_va, y_va = val_t

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_mae_mw": []}

    for epoch in range(epochs):
        model.train()                       
        running, n_seen = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()                 
                                            
            loss = loss_fn(model(xb), yb)
            loss.backward()                
            opt.step()                      
            running += loss.item() * len(xb)
            n_seen += len(xb)

        model.eval()                        
        with torch.no_grad():               
            train_loss = running / n_seen
            val_pred = model(X_va)
            val_loss = loss_fn(val_pred, y_va).item()
            pred_mw = y_scaler.inverse_transform(val_pred.numpy())
            true_mw = y_scaler.inverse_transform(y_va.numpy())
            val_mae = mae(true_mw, pred_mw)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_mw"].append(val_mae)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:>3}  train {train_loss:.4f}  "
                  f"val {val_loss:.4f}  val MAE {val_mae:,.0f} MW")

    return history


def main():
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    train, val, test = temporal_split(df)
    feats = feature_columns(df)

    train_t, val_t, test_t, x_scaler, y_scaler = prepare(train, val, test, feats)

    model = build_mlp(len(feats))
    history = train_model(model, train_t, val_t, y_scaler)

    best = int(np.argmin(history["val_mae_mw"]))
    print(f"\nbest epoch: {best}   val MAE {history['val_mae_mw'][best]:,.0f} MW")


    return {
        "model": model,
        "history": history,
        "train": train, "val": val, "test": test,
        "train_t": train_t, "val_t": val_t, "test_t": test_t,
        "feats": feats,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }


if __name__ == "__main__":
    main()