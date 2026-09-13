from __future__ import annotations

import datetime
from typing import Annotated

import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from avalanche_ml.api.dependencies import get_db, get_model
from avalanche_ml.api.schemas import (
    HealthResponse,
    HistoryRecord,
    PredictionRequest,
    PredictionResponse,
    StationResponse,
)

router = APIRouter()

DbDep = Annotated[duckdb.DuckDBPyConnection, Depends(get_db)]
ModelDep = Annotated[object, Depends(get_model)]


@router.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest, db: DbDep, model: ModelDep):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    zones = db.execute(
        "SELECT DISTINCT zone_id FROM zone_station_mapping WHERE zone_id = ?",
        [req.zone],
    ).fetchall()
    if not zones:
        raise HTTPException(status_code=404, detail=f"Zone '{req.zone}' not found")

    feature_names = model.stage2.feature_names_
    n_features = len(feature_names)
    X = pd.DataFrame(
        np.zeros((1, n_features)),
        columns=feature_names,
    )

    result = model.predict(X)

    distribution = result["danger_distribution"][0].tolist()
    binary_label = "elevated" if result["danger_binary"][0] == 1 else "not_elevated"
    confidence = float(result["danger_proba"][0])

    problem_probs = result["problem_types"]
    problem_dict = {col: float(problem_probs[col].iloc[0]) for col in problem_probs.columns}

    shap_vals = result["shap_values"]["binary"][0]
    top_indices = np.argsort(np.abs(shap_vals))[-10:][::-1]
    explanations = {}
    for idx in top_indices:
        if idx < len(feature_names):
            explanations[feature_names[idx]] = float(shap_vals[idx])

    model_version = model.stage2.metadata_.get("training_date", "unknown")

    last_ingestion = db.execute(
        "SELECT MAX(last_timestamp) FROM ingestion_watermarks"
    ).fetchone()
    data_freshness = (
        last_ingestion[0]
        if last_ingestion and last_ingestion[0]
        else datetime.datetime.now(datetime.UTC)
    )

    return PredictionResponse(
        danger_distribution=distribution,
        danger_binary=binary_label,
        confidence=confidence,
        problem_types=problem_dict,
        explanations=explanations,
        model_version=model_version,
        data_freshness=data_freshness,
    )


@router.get("/history", response_model=list[HistoryRecord])
def history(
    zone: str,
    elevation_band: str,
    start_date: datetime.date,
    end_date: datetime.date,
    db: DbDep,
):
    rows = db.execute(
        """
        SELECT date, danger_level
        FROM danger_ratings
        WHERE zone_id = ? AND elevation_band = ? AND date BETWEEN ? AND ?
        ORDER BY date
        """,
        [zone, elevation_band, start_date, end_date],
    ).fetchall()

    return [
        HistoryRecord(date=row[0], actual_danger=row[1])
        for row in rows
    ]


@router.get("/stations", response_model=list[StationResponse])
def stations(zone: str | None = None, *, db: DbDep):
    if zone:
        rows = db.execute(
            """
            SELECT s.station_id, s.name, s.elevation, s.active,
                   (SELECT MAX(date) FROM snotel_daily d WHERE d.station_id = s.station_id)
                     AS latest
            FROM stations s
            JOIN zone_station_mapping m ON s.station_id = m.station_id
            WHERE m.zone_id = ?
            ORDER BY s.station_id
            """,
            [zone],
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT s.station_id, s.name, s.elevation, s.active,
                   (SELECT MAX(date) FROM snotel_daily d WHERE d.station_id = s.station_id)
                     AS latest
            FROM stations s
            ORDER BY s.station_id
            """,
        ).fetchall()

    now = datetime.datetime.now(datetime.UTC)
    results = []
    for row in rows:
        station_id, name, elevation, active, latest = row
        if not active or latest is None:
            status = "offline"
        else:
            latest_dt = (
                datetime.datetime.combine(latest, datetime.time())
                if isinstance(latest, datetime.date) and not isinstance(latest, datetime.datetime)
                else latest
            )
            if (now - latest_dt.replace(tzinfo=datetime.UTC)).total_seconds() > 86400:
                status = "stale"
            else:
                status = "active"
        latest_reading = None
        if latest is not None:
            latest_reading = (
                datetime.datetime.combine(latest, datetime.time())
                if isinstance(latest, datetime.date) and not isinstance(latest, datetime.datetime)
                else latest
            )
        results.append(StationResponse(
            station_id=station_id,
            name=name,
            elevation=elevation,
            latest_reading=latest_reading,
            status=status,
        ))
    return results


@router.get("/health", response_model=HealthResponse)
def health(db: DbDep, model: ModelDep):
    station_counts = db.execute(
        "SELECT COUNT(*), SUM(CASE WHEN active THEN 1 ELSE 0 END) FROM stations"
    ).fetchone()
    total = station_counts[0] if station_counts else 0
    online = station_counts[1] if station_counts else 0

    last_ingestion_row = db.execute(
        "SELECT MAX(last_timestamp) FROM ingestion_watermarks"
    ).fetchone()
    last_ingestion = last_ingestion_row[0] if last_ingestion_row and last_ingestion_row[0] else None

    data_gaps: list[str] = []
    if last_ingestion:
        now = datetime.datetime.now(datetime.UTC)
        li = last_ingestion
        if isinstance(li, datetime.date) and not isinstance(li, datetime.datetime):
            li = datetime.datetime.combine(li, datetime.time())
        hours_since = (now - li.replace(tzinfo=datetime.UTC)).total_seconds() / 3600
        if hours_since > 36:
            data_gaps.append(f"stale_data: {hours_since:.0f}h since last ingestion")

    if model is None:
        return HealthResponse(
            status="degraded",
            model_version="none",
            last_ingestion=last_ingestion,
            data_gaps=["no_model_loaded", *data_gaps],
            stations_online=int(online),
            stations_total=int(total),
        )

    model_version = model.stage2.metadata_.get("training_date", "unknown")
    status = "degraded" if data_gaps else "healthy"

    return HealthResponse(
        status=status,
        model_version=model_version,
        last_ingestion=last_ingestion,
        data_gaps=data_gaps,
        stations_online=int(online),
        stations_total=int(total),
    )
