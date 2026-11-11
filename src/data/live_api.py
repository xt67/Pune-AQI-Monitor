"""Keyless Open-Meteo weather + CAMS air-quality forecasts with bounded caching.

Public endpoints are for non-commercial use. Values are provider model estimates,
not CPCB station measurements or predictions from this repository's ML models.
"""
import copy
import math
import threading
from datetime import datetime, timedelta, timezone

import httpx

LOCATIONS = {
    "shivajinagar": ("Shivajinagar", 18.5308, 73.8475),
    "karve-road": ("Karve Road", 18.5018, 73.8167),
    "hadapsar": ("Hadapsar", 18.5089, 73.9260),
    "kothrud": ("Kothrud", 18.5074, 73.8077),
    "hinjewadi": ("Hinjewadi", 18.5912, 73.7390),
    "pimpri": ("Pimpri-Chinchwad", 18.6298, 73.7997),
    "katraj": ("Katraj", 18.4575, 73.8678),
    "viman-nagar": ("Viman Nagar", 18.5679, 73.9143),
    "aundh": ("Aundh", 18.5590, 73.8076),
    "mundhwa": ("Mundhwa", 18.5330, 73.9320),
}
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


class LiveDataUnavailable(RuntimeError):
    pass


def finite_or_none(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


class LiveForecastService:
    def __init__(self, transport=None, now=None):
        self.transport = transport
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.cache = {}
        self.lock = threading.Lock()

    def forecast(self, location, hours=24):
        if location not in LOCATIONS:
            raise ValueError("Unknown Pune location")
        if not 1 <= hours <= 72:
            raise ValueError("hours must be between 1 and 72")
        # One cached 72-hour forecast per location bounds cache size and avoids
        # refetching whenever a visitor changes the displayed horizon.
        with self.lock:
            now = self.now()
            cached = self.cache.get(location)
            if cached and now - cached[0] < timedelta(minutes=15):
                return self._slice(cached[1], hours, now, False)
            try:
                result = self._fetch(location, now)
                self.cache[location] = (now, result)
                return self._slice(result, hours, now, False)
            except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
                if cached and now - cached[0] < timedelta(hours=2):
                    return self._slice(cached[1], hours, now, True)
                raise LiveDataUnavailable("Live provider unavailable or incomplete; retry later. No simulated values were substituted.") from exc

    def _slice(self, result, hours, now, stale):
        result = copy.deepcopy(result)
        cutoff = now.replace(minute=0, second=0, microsecond=0)
        result["hourly"] = [row for row in result["hourly"] if datetime.fromisoformat(row["timestamp"]) >= cutoff][:hours]
        if len(result["hourly"]) < hours:
            raise LiveDataUnavailable("Cached provider forecast no longer covers the requested horizon")
        result["stale"] = stale
        result["cache_age_seconds"] = max(0, int((now - datetime.fromisoformat(result["fetched_at"])).total_seconds()))
        result["hours"] = hours
        return result

    def _fetch(self, location, now):
        name, latitude, longitude = LOCATIONS[location]
        params = {"latitude":latitude, "longitude":longitude, "timezone":"UTC", "timeformat":"unixtime", "forecast_hours":74}
        air_fields = ["us_aqi", "pm2_5", "pm10", "nitrogen_dioxide", "ozone"]
        weather_fields = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation_probability"]
        with httpx.Client(timeout=15, transport=self.transport, follow_redirects=False) as client:
            air = client.get(AIR_URL, params={**params, "hourly": ",".join(air_fields), "domains":"cams_global"})
            air.raise_for_status()
            weather = client.get(WEATHER_URL, params={**params, "hourly": ",".join(weather_fields), "wind_speed_unit":"kmh"})
            weather.raise_for_status()
        air, weather = air.json(), weather.json()
        aq, wx = air["hourly"], weather["hourly"]
        times = aq["time"]
        if len(times) != len(set(times)) or times != sorted(times):
            raise ValueError("Invalid provider timestamps")
        if any(b-a != 3600 for a,b in zip(times,times[1:])):
            raise ValueError("Provider forecast has missing hours")
        weather_times = {value:index for index,value in enumerate(wx["time"])}
        hourly = []
        for i, stamp in enumerate(times):
            if stamp not in weather_times:
                raise ValueError("Weather and air-quality hours do not align")
            row = {"timestamp":datetime.fromtimestamp(stamp, timezone.utc).isoformat()}
            row.update({field:finite_or_none(aq[field][i]) for field in air_fields})
            row.update({field:finite_or_none(wx[field][weather_times[stamp]]) for field in weather_fields})
            hourly.append(row)
        if not hourly or all(row["us_aqi"] is None for row in hourly):
            raise ValueError("No AQI forecast values available")
        return {"location_id":location,"location":name,"latitude":latitude,"longitude":longitude,
                "data_source":"open_meteo_cams_forecast", "aqi_standard":"US EPA AQI (not Indian CPCB AQI)",
                "fetched_at":now.isoformat(),"hourly":hourly,
                "units":{**air.get("hourly_units",{}), **weather.get("hourly_units",{})},
                "attribution":"Open-Meteo and Copernicus Atmosphere Monitoring Service (CAMS)",
                "source_urls":["https://open-meteo.com/", "https://atmosphere.copernicus.eu/"],
                "limitations":"Regional model estimates (~45 km CAMS global grid), not neighborhood sensor readings. Nearby Pune locations may share a grid cell."}


live_service = LiveForecastService()
