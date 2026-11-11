"""FastAPI main application."""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router, set_forecaster
from src.models.inference import BundleForecaster

logger = logging.getLogger(__name__)

# Application metadata
APP_TITLE = "Pune Air Quality Sentinel API"
APP_DESCRIPTION = """
Live Pune air-quality and weather forecasts from Open-Meteo / CAMS, with bounded
caching, freshness metadata and source attribution. Use `/api/v1/locations` and
`/api/v1/forecast/{location}`. No API key or old training dataset is needed.

Provider AQI uses the US EPA scale, not Indian CPCB AQI. Forecasts are regional
model estimates, not neighborhood sensor readings. The legacy `/predict` routes
remain available for explicit demos and locally trained research bundles.
"""
APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    directory = os.environ.get("PAQS_BUNDLE_DIR", str(Path(__file__).resolve().parents[2] / "models" / "bundles"))
    service = BundleForecaster(directory)
    set_forecaster(service)
    logger.info("Available station/model bundles: %s", list(service.bundles))
    try:
        yield
    finally:
        set_forecaster(None)


# Create FastAPI app
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
