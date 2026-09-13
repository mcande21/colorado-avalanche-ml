from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any

import duckdb
import httpx

logger = logging.getLogger(__name__)

AWDB_BASE = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
MAX_RETRIES = 3
BACKOFF_BASE = 1.0
DEFAULT_RATE_LIMIT_WAIT = 60


class SnotelClient:
    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        self._db = db

    async def _request(self, url: str, params: dict | None = None) -> Any:
        async with httpx.AsyncClient(timeout=30.0) as http:
            for attempt in range(MAX_RETRIES + 1):
                response = await http.get(url, params=params)
                if response.status_code == 429:
                    wait = int(response.headers.get("Retry-After", DEFAULT_RATE_LIMIT_WAIT))
                    logger.warning("Rate limited, waiting %ds", wait)
                    await asyncio.sleep(wait)
                    continue
                if response.status_code >= 500:
                    if attempt < MAX_RETRIES:
                        delay = BACKOFF_BASE * (2 ** attempt)
                        logger.warning(
                            "Server error %d, retry %d/%d in %.1fs",
                            response.status_code, attempt + 1, MAX_RETRIES, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    response.raise_for_status()
                response.raise_for_status()
                return response.json()
        raise httpx.HTTPStatusError(
            "Max retries exceeded", request=httpx.Request("GET", url), response=response
        )

    async def discover(self) -> int:
        data = await self._request(
            f"{AWDB_BASE}/stations",
            params={"stateCode": "CO", "networkCode": "SNTL"},
        )
        now = datetime.datetime.now(tz=datetime.UTC)
        count = 0
        for station in data:
            self._db.execute("""
                INSERT INTO stations (station_id, name, latitude, longitude, elevation, huc,
                                      state_code, county, active, catalog_updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, TRUE, ?)
                ON CONFLICT (station_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    elevation = EXCLUDED.elevation,
                    huc = EXCLUDED.huc,
                    state_code = EXCLUDED.state_code,
                    county = EXCLUDED.county,
                    catalog_updated_at = EXCLUDED.catalog_updated_at
            """, [
                station["stationTriplet"],
                station["name"],
                station["latitude"],
                station["longitude"],
                station["elevation"],
                station.get("huc"),
                station.get("stateCode"),
                station.get("countyName"),
                now,
            ])
            count += 1
        return count

    async def ingest(
        self,
        station_id: str,
        start_date: datetime.date,
        end_date: datetime.date,
        resolution: str = "daily",
    ) -> int:
        params = {
            "stationTriplets": station_id,
            "beginDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
        }
        if resolution == "hourly":
            params["duration"] = "HOURLY"
        else:
            params["duration"] = "DAILY"

        data = await self._request(f"{AWDB_BASE}/data", params=params)
        if not data:
            return 0

        count = 0
        prev_values: dict[str, float | None] = {}

        for record in data:
            values = record.get("values", [])
            for val in values:
                if resolution == "hourly":
                    count += self._insert_hourly(station_id, val, prev_values)
                else:
                    count += self._insert_daily(station_id, val, prev_values)
                prev_values = {
                    "swe": val.get("swe"),
                    "snowDepth": val.get("snowDepth"),
                }

        if count > 0:
            self._update_watermark(station_id, resolution, end_date)

        return count

    def _validate_daily(
        self, val: dict, prev: dict[str, float | None]
    ) -> list[str]:
        flags = []
        swe = val.get("swe")
        snow_depth = val.get("snowDepth")
        temp_min = val.get("airTempMin")
        temp_max = val.get("airTempMax")
        temp_avg = val.get("airTempAvg")

        if swe is not None and swe < 0:
            flags.append("swe_negative")
        if snow_depth is not None and snow_depth < 0:
            flags.append("snow_depth_negative")

        for temp in [temp_min, temp_max, temp_avg]:
            if temp is not None and (temp < -60 or temp > 120):
                flags.append("temp_out_of_range")
                break

        if swe is not None and prev.get("swe") is not None and prev["swe"] > 0:
            change_ratio = abs(swe - prev["swe"]) / prev["swe"]
            if change_ratio > 2.0:
                flags.append("swe_discontinuity")

        if (snow_depth is not None and prev.get("snowDepth") is not None
                and prev["snowDepth"] > 0):
            change_ratio = abs(snow_depth - prev["snowDepth"]) / prev["snowDepth"]
            if change_ratio > 2.0:
                flags.append("snow_depth_discontinuity")

        return flags

    def _validate_hourly(
        self, val: dict, prev: dict[str, float | None]
    ) -> list[str]:
        flags = []
        swe = val.get("swe")
        snow_depth = val.get("snowDepth")
        temp = val.get("airTemp")

        if swe is not None and swe < 0:
            flags.append("swe_negative")
        if snow_depth is not None and snow_depth < 0:
            flags.append("snow_depth_negative")
        if temp is not None and (temp < -60 or temp > 120):
            flags.append("temp_out_of_range")

        if swe is not None and prev.get("swe") is not None and prev["swe"] > 0:
            change_ratio = abs(swe - prev["swe"]) / prev["swe"]
            if change_ratio > 2.0:
                flags.append("swe_discontinuity")

        return flags

    def _insert_daily(
        self, station_id: str, val: dict, prev: dict[str, float | None]
    ) -> int:
        flags = self._validate_daily(val, prev)
        quality_flag = ",".join(flags) if flags else None
        self._db.execute("""
            INSERT INTO snotel_daily
                (station_id, date, swe_inches, snow_depth_inches,
                 air_temp_min_f, air_temp_max_f, air_temp_mean_f,
                 precip_increment_inches, quality_flag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (station_id, date) DO UPDATE SET
                swe_inches = EXCLUDED.swe_inches,
                snow_depth_inches = EXCLUDED.snow_depth_inches,
                air_temp_min_f = EXCLUDED.air_temp_min_f,
                air_temp_max_f = EXCLUDED.air_temp_max_f,
                air_temp_mean_f = EXCLUDED.air_temp_mean_f,
                precip_increment_inches = EXCLUDED.precip_increment_inches,
                quality_flag = EXCLUDED.quality_flag
        """, [
            station_id,
            val["date"],
            val.get("swe"),
            val.get("snowDepth"),
            val.get("airTempMin"),
            val.get("airTempMax"),
            val.get("airTempAvg"),
            val.get("precipIncrement"),
            quality_flag,
        ])
        return 1

    def _insert_hourly(
        self, station_id: str, val: dict, prev: dict[str, float | None]
    ) -> int:
        flags = self._validate_hourly(val, prev)
        quality_flag = ",".join(flags) if flags else None
        self._db.execute("""
            INSERT INTO snotel_hourly
                (station_id, timestamp, swe_inches, snow_depth_inches,
                 air_temp_f, precip_accum_inches, quality_flag)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (station_id, timestamp) DO UPDATE SET
                swe_inches = EXCLUDED.swe_inches,
                snow_depth_inches = EXCLUDED.snow_depth_inches,
                air_temp_f = EXCLUDED.air_temp_f,
                precip_accum_inches = EXCLUDED.precip_accum_inches,
                quality_flag = EXCLUDED.quality_flag
        """, [
            station_id,
            val["dateTime"],
            val.get("swe"),
            val.get("snowDepth"),
            val.get("airTemp"),
            val.get("precipAccum"),
            quality_flag,
        ])
        return 1

    def _update_watermark(
        self, station_id: str, resolution: str, end_date: datetime.date
    ) -> None:
        ts = datetime.datetime.combine(end_date, datetime.time.max)
        now = datetime.datetime.now(tz=datetime.UTC)
        self._db.execute("""
            INSERT INTO ingestion_watermarks (source, entity_id, resolution,
                                              last_timestamp, updated_at)
            VALUES ('snotel', ?, ?, ?, ?)
            ON CONFLICT (source, entity_id, resolution) DO UPDATE SET
                last_timestamp = CASE
                    WHEN EXCLUDED.last_timestamp > ingestion_watermarks.last_timestamp
                    THEN EXCLUDED.last_timestamp
                    ELSE ingestion_watermarks.last_timestamp
                END,
                updated_at = EXCLUDED.updated_at
        """, [station_id, resolution, ts, now])

    def _get_watermark(
        self, station_id: str, resolution: str
    ) -> datetime.datetime | None:
        row = self._db.execute(
            "SELECT last_timestamp FROM ingestion_watermarks "
            "WHERE source = 'snotel' AND entity_id = ? AND resolution = ?",
            [station_id, resolution],
        ).fetchone()
        if row is None:
            return None
        return row[0]

    async def ingest_incremental(
        self, station_id: str, resolution: str = "daily"
    ) -> int:
        watermark = self._get_watermark(station_id, resolution)
        if watermark is None:
            logger.warning(
                "No watermark for %s/%s, skipping incremental", station_id, resolution
            )
            return 0
        start_date = watermark.date() + datetime.timedelta(days=1)
        end_date = datetime.datetime.now(tz=datetime.UTC).date()
        if start_date > end_date:
            return 0
        return await self.ingest(
            station_id=station_id,
            start_date=start_date,
            end_date=end_date,
            resolution=resolution,
        )
