import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.inference import BundleForecaster
from src.models.lstm import LSTMForecaster
from src.models.dataset import AQIDataset
from src.data.preprocess import create_train_val_test_split


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "MH020" / "lstm"
    root.mkdir(parents=True)
    architecture = dict(input_size=2, hidden_sizes=[4,4], horizon=3, dropout=0.)
    model = LSTMForecaster(**architecture)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.output[-1].bias.data.fill_(.5)
    torch.save(model.state_dict(), root / "weights.pt")
    metadata = {"version":1,"station_id":"MH020","model_type":"lstm","data_source":"observed",
                "feature_cols":["aqi","pm25"],"window_size":4,"horizon":3,
                "architecture":architecture,"scale":[.01,.02],"offset":[-1.,0.],
                "train_end":"2026-01-01T00:00:00+00:00","test_mae":12.3}
    (root / "metadata.json").write_text(json.dumps(metadata))
    history = pd.DataFrame({"timestamp":pd.date_range("2026-01-02",periods=6,freq="h",tz="UTC"),
                            "node_id":"MH020", "aqi":100.,"pm25":20.})
    history.to_csv(root.parent / "history.csv",index=False)
    return BundleForecaster(tmp_path), history.timestamp.iloc[-1], root


def test_inference_uses_saved_scaler_and_horizon(bundle):
    service, origin, _ = bundle
    result = service.predict("MH020","lstm",2,origin)
    np.testing.assert_allclose(result["values"],[150.,150.])
    assert result["origin"] == origin
    assert result["current_aqi"] == 100


def test_station_horizon_staleness_and_missing_history_fail(bundle):
    service, origin, root = bundle
    for station, horizon, when in [("MH021",2,origin),("MH020",4,origin),("MH020",2,origin+timedelta(days=2))]:
        with pytest.raises(ValueError):
            service.predict(station,"lstm",horizon,when)
    history = pd.read_csv(root.parent / "history.csv")
    history.loc[4,"aqi"] = np.nan
    history.to_csv(root.parent / "history.csv",index=False)
    with pytest.raises(ValueError,match="non-finite"):
        service.predict("MH020","lstm",2,origin)


def test_simulated_bundle_is_not_served_as_observed(bundle):
    _, _, root = bundle
    meta = json.loads((root / "metadata.json").read_text())
    meta["data_source"] = "simulated"
    (root / "metadata.json").write_text(json.dumps(meta))
    assert not BundleForecaster(root.parents[1]).bundles


def test_api_forecast_starts_after_observation(bundle, monkeypatch):
    from fastapi.testclient import TestClient
    from app.api.main import app
    from app.api import routes
    service, origin, _ = bundle
    monkeypatch.setattr(routes,"_forecaster",service)
    response = TestClient(app).post("/api/v1/predict",json={"station_id":"MH020","horizon":2,"as_of":origin.isoformat()})
    assert response.status_code == 200
    result = response.json()
    assert result["data_source"] == "model"
    assert result["predictions"][0]["hour"] == 1
    assert pd.Timestamp(result["predictions"][0]["timestamp"]) == origin + timedelta(hours=1)


def test_unsorted_multi_station_splits_do_not_leak_timestamps():
    frame = pd.DataFrame({"timestamp":list(pd.date_range("2026-01-01",periods=100,freq="h"))*2,
                          "node_id":["a"]*100+["b"]*100,"aqi":100.}).sample(frac=1,random_state=4)
    splits = create_train_val_test_split(frame)
    assert splits["train_df"].timestamp.max() < splits["val_df"].timestamp.min()
    assert splits["val_df"].timestamp.max() < splits["test_df"].timestamp.min()


def test_dataset_windows_do_not_bridge_missing_hours():
    frame = pd.DataFrame({"timestamp":pd.date_range("2026-01-01",periods=10,freq="h"),"node_id":"a","aqi":100.})
    frame = frame.drop(index=5)
    assert len(AQIDataset(frame,window_size=3,horizon=2)) == 1
