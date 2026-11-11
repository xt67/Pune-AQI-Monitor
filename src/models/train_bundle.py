"""Reproducible station training, identical held-out targets and inference export.

Input CSV: hourly timestamp, node_id, aqi and optional numeric feature columns.
Synthetic smoke data must use --data-source simulated; it is never served live.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

from src.models.inference import read_history, check_hourly
from src.models.lstm import LSTMForecaster


def train_bundle(data, station, output, features, window=72, horizon=24, epochs=20, seed=42, data_source="observed"):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    history = read_history(data, station)
    check_hourly(history)
    if "aqi" not in features or len(set(features)) != len(features):
        raise ValueError("Features must be unique and include aqi")
    values = history[features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Clean missing values causally before training")
    if not 0 <= history.aqi.min() <= history.aqi.max() <= 500:
        raise ValueError("AQI must be in [0,500]")
    a, b = int(len(history) * .7), int(len(history) * .85)
    if a < window + horizon or b-a < horizon or len(history)-b < horizon:
        raise ValueError("Insufficient history for disjoint train/validation/test targets")
    scaler = MinMaxScaler().fit(values[:a])
    normalized = scaler.transform(values).astype(np.float32)
    target = features.index("aqi")

    def windows(start, end):
        origins = list(range(max(window, start), end-horizon+1))
        x = np.stack([normalized[i-window:i] for i in origins])
        y = np.stack([normalized[i:i+horizon, target] for i in origins])
        return torch.from_numpy(x), torch.from_numpy(y), origins

    train_x, train_y, _ = windows(window, a)
    val_x, val_y, _ = windows(a, b)
    test_x, _, origins = windows(b, len(history))
    architecture = dict(input_size=len(features), hidden_sizes=[32,16], horizon=horizon, dropout=.1)
    model = LSTMForecaster(**architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    best, state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for ids in torch.randperm(len(train_x)).split(64):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(train_x[ids]), train_y[ids])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = torch.nn.functional.mse_loss(model(val_x), val_y).item()
        if score < best:
            best, state = score, {k:v.clone() for k,v in model.state_dict().items()}
    if state is None:
        raise ValueError("No finite training checkpoint produced")
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred = (model(test_x).numpy() - scaler.min_[target]) / scaler.scale_[target]
    actual = np.stack([values[i:i+horizon, target] for i in origins])
    arima = ARIMA(values[:a, target], order=(2,1,2)).fit()
    arima_pred = np.stack([arima.apply(values[i-window:i, target], refit=False).forecast(horizon) for i in origins])
    base = Path(output) / station
    base.mkdir(parents=True, exist_ok=True)
    history.to_csv(base / "history.csv", index=False)
    report = {"station_id": station, "data_source": data_source, "seed": seed,
              "horizon": horizon, "window_size": window, "test_windows": len(origins),
              "train_end": history.timestamp.iloc[a-1].isoformat(),
              "validation_end": history.timestamp.iloc[b-1].isoformat(),
              "test_start": history.timestamp.iloc[b].isoformat(), "models": {}}
    for kind, predictions in (("lstm", pred), ("arima", arima_pred)):
        metrics = {"mae": float(np.abs(predictions-actual).mean()),
                   "rmse": float(np.sqrt(((predictions-actual)**2).mean()))}
        report["models"][kind] = metrics
        root = base / kind
        root.mkdir(exist_ok=True)
        meta = {"version":1, "station_id":station, "model_type":kind, "data_source":data_source,
                "feature_cols":features, "window_size":window, "horizon":horizon,
                "scale":scaler.scale_.tolist(), "offset":scaler.min_.tolist(),
                "architecture":architecture, "train_end":report["train_end"], "test_mae":metrics["mae"]}
        (root / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if kind == "lstm":
            torch.save(state, root / "weights.pt")
        else:
            arima.save(root / "model.pkl")
    (base / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--station", required=True)
    parser.add_argument("--output", default="models/bundles")
    parser.add_argument("--features", default="aqi")
    parser.add_argument("--window", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-source", choices=["observed", "simulated"], required=True)
    args = vars(parser.parse_args())
    args["features"] = args["features"].split(",")
    print(json.dumps(train_bundle(**args), indent=2))
