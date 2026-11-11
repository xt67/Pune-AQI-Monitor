from datetime import datetime, timedelta, timezone
import httpx
import pytest
from src.data.live_api import LiveForecastService, LiveDataUnavailable


@pytest.fixture
def provider():
    clock = [datetime(2026,11,11,9,5,tzinfo=timezone.utc)]
    calls = []
    state = {"fail":False, "missing":False}
    def handler(request):
        calls.append(request)
        if state["fail"]:
            return httpx.Response(503)
        fields = request.url.params["hourly"].split(",")
        start = int(clock[0].replace(minute=0).timestamp())
        hourly = {"time":[start+3600*i for i in range(74)]}
        hourly.update({field:[42.] * 74 for field in fields})
        if state["missing"] and "pm2_5" in hourly:
            hourly["pm2_5"][0] = None
        return httpx.Response(200,json={"hourly":hourly,"hourly_units":{field:"test" for field in fields}})
    return LiveForecastService(transport=httpx.MockTransport(handler),now=lambda:clock[0]),clock,calls,state


def test_live_join_cache_and_provenance(provider):
    service,clock,calls,_ = provider
    data = service.forecast("shivajinagar",24)
    assert len(data["hourly"]) == 24
    assert data["data_source"] == "open_meteo_cams_forecast"
    assert "not Indian CPCB" in data["aqi_standard"]
    assert "temperature_2m" in data["hourly"][0]
    assert len(calls) == 2
    data["hourly"][0]["us_aqi"] = -10
    assert service.forecast("shivajinagar",48)["hourly"][0]["us_aqi"] == 42
    assert len(calls) == 2
    clock[0] += timedelta(minutes=16)
    service.forecast("shivajinagar",24)
    assert len(calls) == 4


def test_outage_has_bounded_explicit_stale_fallback(provider):
    service,clock,_,state = provider
    service.forecast("shivajinagar",24)
    state["fail"] = True
    clock[0] += timedelta(minutes=20)
    assert service.forecast("shivajinagar",24)["stale"]
    clock[0] += timedelta(hours=2)
    with pytest.raises(LiveDataUnavailable):
        service.forecast("shivajinagar",24)


def test_missing_values_are_not_fabricated(provider):
    service,_,_,state = provider
    state["missing"] = True
    assert service.forecast("shivajinagar",24)["hourly"][0]["pm2_5"] is None


def test_first_request_outage_has_no_random_fallback(provider):
    service,_,_,state = provider
    state["fail"] = True
    with pytest.raises(LiveDataUnavailable):
        service.forecast("shivajinagar",24)


def test_live_endpoint_validation_and_response(provider, monkeypatch):
    from app.api.main import app
    from app.api import routes
    from fastapi.testclient import TestClient
    monkeypatch.setattr(routes,"live_service",provider[0])
    client = TestClient(app)
    assert client.get("/api/v1/forecast/shivajinagar?hours=24").status_code == 200
    assert client.get("/api/v1/forecast/unknown").status_code == 404
    assert client.get("/api/v1/forecast/shivajinagar?hours=1000").status_code == 422
