"""Station-scoped inference from versioned training bundles and observed history."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_history(path, station):
    frame = pd.read_csv(path)
    if not {"timestamp", "node_id", "aqi"}.issubset(frame):
        raise ValueError("History needs timestamp, node_id and aqi")
    frame = frame.loc[frame.node_id.astype(str) == station].copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if frame.empty or frame.timestamp.duplicated().any():
        raise ValueError("Station history is empty or contains duplicate timestamps")
    return frame


def check_hourly(frame):
    if not frame.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("History must be contiguous hourly observations; resample explicitly before training")


class BundleForecaster:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.bundles = {}
        for path in self.directory.glob("*/*/metadata.json"):
            meta = json.loads(path.read_text(encoding="utf-8"))
            if meta.get("version") != 1 or meta.get("data_source") != "observed":
                continue
            station, kind = meta["station_id"], meta["model_type"]
            if kind not in ("lstm", "arima"):
                continue
            if path.parent.name != kind or path.parent.parent.name != station:
                raise ValueError("Bundle path does not match station/model metadata")
            self.bundles[(station, kind)] = (path.parent, meta)

    def predict(self, station, kind, horizon, as_of=None):
        if (station, kind) not in self.bundles:
            raise ValueError(f"No observed-data {kind} bundle available for station {station}")
        root, meta = self.bundles[(station, kind)]
        if not 1 <= horizon <= meta["horizon"]:
            raise ValueError(f"Requested horizon exceeds trained horizon {meta['horizon']}")
        history = read_history(root.parent / "history.csv", station)
        cutoff = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        history = history.loc[history.timestamp <= cutoff]
        window = history.tail(meta["window_size"])
        if len(window) != meta["window_size"]:
            raise ValueError("Insufficient historical observations for the trained window")
        check_hourly(window)
        latest = window.timestamp.iloc[-1]
        if cutoff - latest > pd.Timedelta(hours=2):
            raise ValueError("Station history is stale; supply refreshed observations or an explicit historical as_of")
        if latest < pd.Timestamp(meta["train_end"]):
            raise ValueError("Forecast origin precedes training cutoff")
        features = meta["feature_cols"]
        values = window[features].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("History contains missing or non-finite feature values")
        if kind == "lstm":
            import torch
            from src.models.lstm import LSTMForecaster
            scale = np.asarray(meta["scale"], dtype=float)
            offset = np.asarray(meta["offset"], dtype=float)
            if scale.shape != (len(features),) or offset.shape != scale.shape or not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError("Invalid training scaler metadata")
            model = LSTMForecaster(**meta["architecture"])
            model.load_state_dict(torch.load(root / "weights.pt", map_location="cpu", weights_only=True))
            model.eval()
            with torch.no_grad():
                result = model(torch.tensor(values * scale + offset, dtype=torch.float32)[None]).numpy()[0]
            target = features.index("aqi")
            result = (result - offset[target]) / scale[target]
        else:
            from statsmodels.tsa.arima.model import ARIMAResults
            # Only load trusted, locally trained pickle artifacts.
            model = ARIMAResults.load(root / "model.pkl")
            result = model.apply(window.aqi.to_numpy(dtype=float), refit=False).forecast(horizon)
        result = np.asarray(result, dtype=float)[:horizon]
        if len(result) != horizon or not np.isfinite(result).all():
            raise ValueError("Model returned invalid forecast values")
        return {"values": np.clip(result, 0, 500), "origin": latest.to_pydatetime(),
                "current_aqi": float(window.aqi.iloc[-1]), "mae": meta.get("test_mae"),
                "data_source": "model", "train_end": meta["train_end"]}
