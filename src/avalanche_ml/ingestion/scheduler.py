from __future__ import annotations

import asyncio
import logging
import time

import duckdb

logger = logging.getLogger(__name__)


async def _run_snotel_ingest(db: duckdb.DuckDBPyConnection) -> None:
    from avalanche_ml.ingestion.snotel import SnotelClient

    client = SnotelClient(db)
    count = await client.ingest_incremental()
    logger.info("SNOTEL ingest completed: %d records", count)


async def _run_caic_ingest(db: duckdb.DuckDBPyConnection) -> None:
    import datetime

    from avalanche_ml.ingestion.caic import CaicClient

    client = CaicClient(db)
    end = datetime.datetime.now(tz=datetime.UTC).date()
    start = end - datetime.timedelta(days=2)
    count = await client.ingest_danger_ratings_with_fallback(start, end)
    logger.info("CAIC ingest completed: %d records", count)


def _run_feature_computation(db: duckdb.DuckDBPyConnection) -> None:
    from avalanche_ml.features.physics import compute_physics_features
    from avalanche_ml.features.weather import compute_weather_features

    w = compute_weather_features(db)
    p = compute_physics_features(db)
    logger.info("Feature computation completed: %d weather, %d physics rows", w, p)


def _run_alignment(db: duckdb.DuckDBPyConnection) -> None:
    from avalanche_ml.features.alignment import assemble_training_matrix

    rows = assemble_training_matrix(db)
    logger.info("Training matrix alignment completed: %d rows", rows)


_STEPS: list[tuple[str, bool]] = [
    ("SNOTEL ingest", True),
    ("CAIC ingest", True),
    ("Feature computation", False),
    ("Training matrix alignment", False),
]


def run_daily_ingestion(db: duckdb.DuckDBPyConnection) -> dict[str, str]:
    results: dict[str, str] = {}
    runners = [
        _run_snotel_ingest,
        _run_caic_ingest,
        _run_feature_computation,
        _run_alignment,
    ]

    for (step_name, is_async), runner in zip(_STEPS, runners):
        t0 = time.monotonic()
        try:
            if is_async:
                asyncio.run(runner(db))
            else:
                runner(db)
            elapsed = time.monotonic() - t0
            logger.info("%s completed successfully in %.1fs", step_name, elapsed)
            results[step_name] = "success"
        except Exception:
            elapsed = time.monotonic() - t0
            logger.exception("%s failed after %.1fs", step_name, elapsed)
            results[step_name] = "failed"

    return results
