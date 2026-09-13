from __future__ import annotations

import csv
import datetime
import io
import logging
from typing import Any

import duckdb
import httpx

logger = logging.getLogger(__name__)

CAIC_API_BASE = "https://avalanche.state.co.us/api-proxy/avid"
AVORG_API_BASE = "https://api.avalanche.org/v2/public"
OAP_BASE = "https://raw.githubusercontent.com/openavproject/data"
CAIC_FEATURE_SERVER = (
    "https://services5.arcgis.com/CAIC/ArcGIS/rest/services"
    "/CAIC_Zones/FeatureServer/0/query"
)
AVORG_BAND_MAP = {
    "upper": "above_treeline",
    "middle": "near_treeline",
    "lower": "below_treeline",
}

ELEVATION_BANDS = ("above_treeline", "near_treeline", "below_treeline")

VALID_PROBLEM_TYPES = {
    "persistent slab": "Persistent Slab",
    "storm slab": "Storm Slab",
    "loose wet": "Loose Wet",
    "loose dry": "Loose Dry",
    "wind slab": "Wind Slab",
    "wet slab": "Wet Slab",
    "deep slab": "Deep Slab",
    "cornice fall": "Cornice Fall",
    "glide": "Glide",
}


class CaicClient:
    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        self._db = db

    async def _request(self, url: str, params: dict | None = None) -> httpx.Response:
        async with httpx.AsyncClient(timeout=30.0) as http:
            response = await http.get(url, params=params)
            response.raise_for_status()
            return response

    async def ingest_danger_ratings(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> int:
        response = await self._request(
            f"{CAIC_API_BASE}/products/all",
            params={
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
            },
        )
        data = response.json()
        return self._store_danger_ratings(data, source="caic")

    def _store_danger_ratings(self, data: list[dict], source: str) -> int:
        count = 0
        now = datetime.datetime.now(tz=datetime.UTC)
        for record in data:
            zone_id = record["zone_id"]
            date = record["date"]
            for band in ELEVATION_BANDS:
                danger_level = record.get(band)
                self._db.execute("""
                    INSERT INTO danger_ratings
                        (zone_id, date, elevation_band, danger_level, source, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (zone_id, date, elevation_band, source) DO UPDATE SET
                        danger_level = EXCLUDED.danger_level,
                        ingested_at = EXCLUDED.ingested_at
                """, [zone_id, date, band, danger_level, source, now])
                count += 1
        return count

    async def ingest_problem_types(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> int:
        response = await self._request(
            f"{CAIC_API_BASE}/products/all",
            params={
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
            },
        )
        data = response.json()
        return self._store_problem_types(data, source="caic")

    def _store_problem_types(self, data: list[dict], source: str) -> int:
        count = 0
        now = datetime.datetime.now(tz=datetime.UTC)
        for record in data:
            zone_id = record["zone_id"]
            date = record["date"]
            for idx, problem in enumerate(record.get("problems", [])):
                raw_type = problem.get("type", "")
                normalized = VALID_PROBLEM_TYPES.get(raw_type.lower().strip(), raw_type)
                self._db.execute("""
                    INSERT INTO problem_types
                        (zone_id, date, problem_type, likelihood, size_min, size_max,
                         aspects, elevation_bands, ordering, source, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (zone_id, date, problem_type, source) DO UPDATE SET
                        likelihood = EXCLUDED.likelihood,
                        size_min = EXCLUDED.size_min,
                        size_max = EXCLUDED.size_max,
                        aspects = EXCLUDED.aspects,
                        elevation_bands = EXCLUDED.elevation_bands,
                        ordering = EXCLUDED.ordering,
                        ingested_at = EXCLUDED.ingested_at
                """, [
                    zone_id,
                    date,
                    normalized,
                    problem.get("likelihood"),
                    problem.get("size_min"),
                    problem.get("size_max"),
                    problem.get("aspects"),
                    problem.get("elevation_bands"),
                    idx,
                    source,
                    now,
                ])
                count += 1
        return count

    async def ingest_oap_danger_ratings(self) -> int:
        response = await self._request(f"{OAP_BASE}/main/labels/danger_ratings.csv")
        reader = csv.DictReader(io.StringIO(response.text))
        records = []
        for row in reader:
            records.append({
                "zone_id": row["zone_id"],
                "date": row["date"],
                "above_treeline": _parse_int_or_none(row.get("danger_above")),
                "near_treeline": _parse_int_or_none(row.get("danger_near")),
                "below_treeline": _parse_int_or_none(row.get("danger_below")),
            })
        return self._store_danger_ratings(records, source="oap")

    async def ingest_oap_problem_types(self) -> int:
        response = await self._request(f"{OAP_BASE}/main/labels/problem_types.csv")
        reader = csv.DictReader(io.StringIO(response.text))
        records: dict[tuple[str, str], list[dict]] = {}
        for row in reader:
            key = (row["zone_id"], row["date"])
            if key not in records:
                records[key] = []
            records[key].append({
                "type": row["problem_type"],
                "likelihood": row.get("likelihood"),
                "size_min": None,
                "size_max": None,
                "aspects": None,
                "elevation_bands": None,
            })
        data = [
            {"zone_id": k[0], "date": k[1], "problems": probs}
            for k, probs in records.items()
        ]
        return self._store_problem_types(data, source="oap")

    async def ingest_avorg_danger_ratings(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> int:
        count = 0
        chunk_start = start_date
        while chunk_start < end_date:
            chunk_end = min(chunk_start + datetime.timedelta(days=30), end_date)
            try:
                response = await self._request(
                    f"{AVORG_API_BASE}/products",
                    params={
                        "type": "forecast",
                        "center_id": "CAIC",
                        "date_start": chunk_start.isoformat(),
                        "date_end": chunk_end.isoformat(),
                    },
                )
                data = response.json()
            except Exception as exc:
                logger.warning("Chunk %s to %s failed: %s", chunk_start, chunk_end, exc)
                chunk_start = chunk_end + datetime.timedelta(days=1)
                continue
            caic_items = [
                d for d in data
                if d.get("avalanche_center", {}).get("name", "").startswith("Colorado")
            ]
            now = datetime.datetime.now(tz=datetime.UTC)
            for item in caic_items:
                zones = item.get("forecast_zone", [])
                zone_names = list({z["name"] for z in zones})
                if not zone_names:
                    continue
                zone_id = zone_names[0]
                forecast_date = item["start_date"][:10]
                for danger in item.get("danger", []):
                    if danger.get("valid_day") != "current":
                        continue
                    for api_key, band in AVORG_BAND_MAP.items():
                        level = danger.get(api_key)
                        if level is None:
                            continue
                        self._db.execute("""
                            INSERT INTO danger_ratings
                                (zone_id, date, elevation_band, danger_level, source, ingested_at)
                            VALUES (?, ?, ?, ?, 'avorg', ?)
                            ON CONFLICT (zone_id, date, elevation_band, source) DO UPDATE SET
                                danger_level = EXCLUDED.danger_level,
                                ingested_at = EXCLUDED.ingested_at
                        """, [zone_id, forecast_date, band, int(level), now])
                        count += 1
            logger.info(
                "Chunk %s to %s: %d items, %d total ratings so far",
                chunk_start, chunk_end, len(caic_items), count,
            )
            chunk_start = chunk_end + datetime.timedelta(days=1)
        return count

    async def ingest_danger_ratings_with_fallback(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> int:
        try:
            return await self.ingest_avorg_danger_ratings(start_date, end_date)
        except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
            logger.warning("avalanche.org unavailable (%s), trying CAIC direct", exc)
            try:
                return await self.ingest_danger_ratings(start_date, end_date)
            except httpx.HTTPStatusError:
                logger.warning("CAIC unavailable, falling back to OAP")
                return await self.ingest_oap_danger_ratings()

    async def update_zone_station_mapping(self) -> int:
        response = await self._request(
            CAIC_FEATURE_SERVER,
            params={"where": "1=1", "outFields": "*", "f": "json"},
        )
        zones = response.json()
        stations = self._db.execute(
            "SELECT station_id, latitude, longitude FROM stations"
        ).fetchall()

        count = 0
        for feature in zones.get("features", []):
            zone_id = feature["attributes"]["zone_id"]
            rings = feature["geometry"]["rings"]
            for station_id, lat, lon in stations:
                if _point_in_polygon(lon, lat, rings[0]):
                    self._db.execute("""
                        INSERT INTO zone_station_mapping (zone_id, station_id)
                        VALUES (?, ?)
                        ON CONFLICT (zone_id, station_id) DO NOTHING
                    """, [zone_id, station_id])
                    count += 1
        return count

    def report_class_distribution(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> dict[str, Any]:
        danger_rows = self._db.execute(
            "SELECT danger_level, COUNT(*) FROM danger_ratings "
            "WHERE date >= ? AND date <= ? AND danger_level IS NOT NULL "
            "GROUP BY danger_level ORDER BY danger_level",
            [start_date.isoformat(), end_date.isoformat()],
        ).fetchall()
        danger_counts: dict[int, int] = {}
        total_danger = 0
        for level, cnt in danger_rows:
            danger_counts[level] = cnt
            total_danger += cnt

        danger_pct: dict[int, float] = {}
        for level, cnt in danger_counts.items():
            danger_pct[level] = round(cnt / total_danger * 100, 1) if total_danger else 0.0

        problem_rows = self._db.execute(
            "SELECT problem_type, COUNT(*) FROM problem_types "
            "WHERE date >= ? AND date <= ? "
            "GROUP BY problem_type ORDER BY problem_type",
            [start_date.isoformat(), end_date.isoformat()],
        ).fetchall()
        problem_counts: dict[str, int] = {}
        total_problems = 0
        for ptype, cnt in problem_rows:
            problem_counts[ptype] = cnt
            total_problems += cnt

        problem_pct: dict[str, float] = {}
        for ptype, cnt in problem_counts.items():
            problem_pct[ptype] = round(cnt / total_problems * 100, 1) if total_problems else 0.0

        logger.info(
            "Class distribution — Danger: %s | Problems: %s",
            {k: f"{v}%" for k, v in danger_pct.items()},
            {k: f"{v}%" for k, v in problem_pct.items()},
        )

        return {
            "danger_counts": danger_counts,
            "danger_pct": danger_pct,
            "problem_counts": problem_counts,
            "problem_pct": problem_pct,
        }


def _parse_int_or_none(val: str | None) -> int | None:
    if val is None or val.strip() == "":
        return None
    return int(val)


def _point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside
