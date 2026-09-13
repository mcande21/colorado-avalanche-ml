from __future__ import annotations

import datetime
from typing import Literal

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    zone: str
    elevation_band: Literal["above_treeline", "near_treeline", "below_treeline"]
    date: datetime.date | None = None


class PredictionResponse(BaseModel):
    danger_distribution: list[float]
    danger_binary: str
    confidence: float
    problem_types: dict[str, float]
    explanations: dict
    model_version: str
    data_freshness: datetime.datetime


class HistoryRecord(BaseModel):
    date: datetime.date
    actual_danger: int | None
    predicted_danger: str | None = None
    danger_distribution: list[float] | None = None


class StationResponse(BaseModel):
    station_id: str
    name: str
    elevation: float
    latest_reading: datetime.datetime | None
    status: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    last_ingestion: datetime.datetime | None
    data_gaps: list[str]
    stations_online: int
    stations_total: int
