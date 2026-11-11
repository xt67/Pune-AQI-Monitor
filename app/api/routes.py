"""API routes for AQI prediction service."""
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.api.models import (
    ErrorResponse,
    HealthResponse,
    HourlyPrediction,
    PredictionRequest,
    PredictionResponse,
    StationInfo,
    StationsResponse,
    get_aqi_category,
)

router = APIRouter()
from src.data.live_api import live_service, LOCATIONS, LiveDataUnavailable


@router.get("/locations", tags=["Live forecasts"])
async def live_locations():
    return [{"id":key, "name":value[0], "latitude":value[1], "longitude":value[2]} for key,value in LOCATIONS.items()]


@router.get("/forecast/{location}", tags=["Live forecasts"])
async def live_forecast(location: str, hours: int = Query(default=24, ge=1, le=72)):
    from starlette.concurrency import run_in_threadpool
    if location not in LOCATIONS:
        raise HTTPException(404, "Unknown Pune location")
    try:
        return await run_in_threadpool(live_service.forecast, location, hours)
    except LiveDataUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


_forecaster = None

def set_forecaster(forecaster):
    global _forecaster
    _forecaster = forecaster


# Pune stations data
STATIONS = {
    "MH020": StationInfo(
        station_id="MH020",
        name="Pune - Karve Road",
        latitude=18.5018,
        longitude=73.8167,
        city="Pune",
        state="Maharashtra",
    ),
    "MH021": StationInfo(
        station_id="MH021",
        name="Pune - Shivajinagar",
        latitude=18.5308,
        longitude=73.8475,
        city="Pune",
        state="Maharashtra",
    ),
    "MH022": StationInfo(
        station_id="MH022",
        name="Pune - Hadapsar",
        latitude=18.5089,
        longitude=73.9260,
        city="Pune",
        state="Maharashtra",
    ),
}

# Model cache (populated on startup)
_models = {"lstm": None, "arima": None}
_model_metrics = {"lstm": {"mae": 59.93}, "arima": {"mae": 87.36}}


def set_models(lstm_model=None, arima_model=None):
    """Set models from startup event."""
    global _models
    _models["lstm"] = lstm_model
    _models["arima"] = arima_model


def get_models():
    """Get loaded models."""
    return _models


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    models_loaded = bool(_forecaster and _forecaster.bundles)
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=models_loaded,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/stations", response_model=StationsResponse, tags=["Stations"])
async def list_stations():
    """Get list of available monitoring stations."""
    return StationsResponse(
        stations=list(STATIONS.values()),
        count=len(STATIONS),
    )


@router.get("/stations/{station_id}", response_model=StationInfo, tags=["Stations"])
async def get_station(station_id: str):
    """Get details for a specific station."""
    if station_id not in STATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Station {station_id} not found",
        )
    return STATIONS[station_id]


@router.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_aqi(request: PredictionRequest):
    """Generate AQI predictions for a station."""
    # Validate station
    if request.station_id not in STATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Station {request.station_id} not found",
        )
    
    # Validate model type
    if request.model_type not in ["lstm", "arima"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type: {request.model_type}. Use 'lstm' or 'arima'",
        )
    
    if not request.demo:
        if _forecaster is None:
            raise HTTPException(503, "No inference bundles configured")
        try:
            from starlette.concurrency import run_in_threadpool
            result = await run_in_threadpool(_forecaster.predict, request.station_id,
                                            request.model_type, request.horizon, request.as_of)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            raise HTTPException(503, str(exc)) from exc
        predictions = []
        for step, value in enumerate(result["values"], start=1):
            value = round(float(value), 1)
            category, color = get_aqi_category(value)
            predictions.append(HourlyPrediction(hour=step,
                timestamp=result["origin"] + timedelta(hours=step), aqi=value, category=category, color=color))
        return PredictionResponse(station_id=request.station_id, model_type=request.model_type,
            horizon=request.horizon, predictions=predictions, data_source="model",
            warning="Forecast from observed station history; model estimates are not measurements.",
            forecast_origin=result["origin"], current_aqi=result["current_aqi"], mae=result["mae"])

    # Generate explicitly requested demonstration data.
    predictions = []
    base_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    
    import random
    base_aqi = random.uniform(80, 150)
    
    for hour in range(request.horizon):
        # Simulate daily pattern (higher AQI in morning/evening rush hours)
        hour_of_day = (base_time.hour + hour) % 24
        if 7 <= hour_of_day <= 10 or 17 <= hour_of_day <= 20:
            aqi = base_aqi * 1.2 + random.uniform(-10, 10)
        else:
            aqi = base_aqi + random.uniform(-15, 15)
        
        aqi = max(0, min(500, aqi))  # Clamp to valid range
        category, color = get_aqi_category(aqi)
        
        predictions.append(
            HourlyPrediction(
                hour=hour + 1,
                timestamp=base_time + timedelta(hours=hour + 1),
                aqi=round(aqi, 1),
                category=category,
                color=color,
            )
        )
    
    return PredictionResponse(
        station_id=request.station_id,
        model_type=request.model_type,
        horizon=request.horizon,
        generated_at=datetime.now(timezone.utc),
        predictions=predictions,
        current_aqi=None,
        mae=None,
    )


@router.get("/predict/{station_id}", response_model=PredictionResponse, tags=["Predictions"])
async def predict_aqi_get(
    station_id: str,
    model_type: str = Query(default="lstm", description="Model type: 'lstm' or 'arima'"),
    horizon: int = Query(default=24, ge=1, le=72, description="Forecast horizon in hours"),
    demo: bool = Query(default=False, description="Request simulated demonstration data"),
    as_of: Optional[datetime] = None,
):
    """Generate AQI predictions (GET endpoint for convenience)."""
    request = PredictionRequest(
        station_id=station_id,
        model_type=model_type,
        horizon=horizon,
        demo=demo,
        as_of=as_of,
    )
    return await predict_aqi(request)
