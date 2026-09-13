from __future__ import annotations

import datetime
from unittest.mock import MagicMock

import duckdb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from avalanche_ml.api.app import create_app
from avalanche_ml.api.dependencies import get_db, get_model
from avalanche_ml.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    StationResponse,
)
from avalanche_ml.db.schema import create_tables
from avalanche_ml.features.alignment import PROBLEM_TYPE_FLAGS


def _make_mock_model(feature_names: list[str] | None = None):
    model = MagicMock()
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(10)]
    model.stage2.feature_names_ = feature_names
    model.stage2.metadata_ = {
        "training_date": "2024-01-01T00:00:00",
        "feature_names": feature_names,
    }

    def mock_predict(X):
        n = len(X)
        distribution = np.zeros((n, 5))
        distribution[:, 2] = 0.6
        distribution[:, 1] = 0.25
        distribution[:, 3] = 0.1
        distribution[:, 0] = 0.03
        distribution[:, 4] = 0.02
        problem_probs = pd.DataFrame(
            {pt: [0.5] * n for pt in PROBLEM_TYPE_FLAGS},
            index=X.index,
        )
        shap_vals = np.random.default_rng(42).standard_normal((n, X.shape[1]))
        return {
            "danger_binary": np.array([1] * n),
            "danger_proba": np.array([0.85] * n),
            "danger_distribution": distribution,
            "danger_3class": np.array([2] * n),
            "problem_types": problem_probs,
            "shap_values": {"binary": shap_vals},
        }

    model.predict.side_effect = mock_predict
    return model


def _seed_db(conn: duckdb.DuckDBPyConnection) -> None:
    create_tables(conn)
    conn.execute("""
        INSERT INTO stations VALUES
        ('335:CO:SNTL', 'Berthoud Summit', 39.80, -105.78, 11300.0,
         '10190005', 'CO', 'Clear Creek', TRUE, '2024-01-01 00:00:00'),
        ('412:CO:SNTL', 'Copper Mountain', 39.50, -106.16, 10520.0,
         '10190004', 'CO', 'Summit', TRUE, '2024-01-01 00:00:00'),
        ('999:CO:SNTL', 'Offline Station', 39.00, -106.00, 9500.0,
         '10190005', 'CO', 'Clear Creek', FALSE, '2023-06-01 00:00:00')
    """)
    conn.execute("""
        INSERT INTO zone_station_mapping VALUES
        ('front-range', '335:CO:SNTL'),
        ('front-range', '999:CO:SNTL'),
        ('vail-summit', '412:CO:SNTL')
    """)
    now = datetime.datetime.now(datetime.UTC)
    conn.execute("""
        INSERT INTO snotel_daily VALUES
        ('335:CO:SNTL', '2024-01-15', 12.5, 38.0, 5.0, 22.0, 13.5, 0.3, 'V'),
        ('412:CO:SNTL', '2024-01-15', 10.0, 30.0, 0.0, 20.0, 10.0, 0.2, 'V')
    """)
    conn.execute(f"""
        INSERT INTO ingestion_watermarks VALUES
        ('snotel', '335:CO:SNTL', 'daily', '{now.isoformat()}', '{now.isoformat()}'),
        ('snotel', '412:CO:SNTL', 'daily', '{now.isoformat()}', '{now.isoformat()}')
    """)
    conn.execute("""
        INSERT INTO danger_ratings VALUES
        ('front-range', '2024-01-15', 'above_treeline', 3, 'caic', '2024-01-15 12:00:00'),
        ('front-range', '2024-01-14', 'above_treeline', 2, 'caic', '2024-01-14 12:00:00'),
        ('front-range', '2024-01-13', 'above_treeline', 4, 'caic', '2024-01-13 12:00:00')
    """)


@pytest.fixture
def test_db():
    conn = duckdb.connect(":memory:")
    _seed_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def mock_model():
    return _make_mock_model()


@pytest.fixture
def client(test_db, mock_model):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: test_db
    app.dependency_overrides[get_model] = lambda: mock_model
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# --- Schema tests ---

class TestPredictionRequest:
    def test_valid_request(self):
        req = PredictionRequest(
            zone="front-range",
            elevation_band="above_treeline",
        )
        assert req.zone == "front-range"
        assert req.date is None

    def test_with_date(self):
        req = PredictionRequest(
            zone="front-range",
            elevation_band="above_treeline",
            date=datetime.date(2024, 1, 15),
        )
        assert req.date == datetime.date(2024, 1, 15)

    def test_invalid_elevation_band(self):
        with pytest.raises(ValueError):
            PredictionRequest(
                zone="front-range",
                elevation_band="invalid_band",
            )


class TestPredictionResponse:
    def test_valid_response(self):
        resp = PredictionResponse(
            danger_distribution=[0.03, 0.25, 0.60, 0.10, 0.02],
            danger_binary="elevated",
            confidence=0.85,
            problem_types={"persistent_slab": 0.65, "storm_slab": 0.30},
            explanations={"temp_mean_24h": 0.15},
            model_version="2024-01-01T00:00:00",
            data_freshness=datetime.datetime(2024, 1, 15, 12, 0, 0, tzinfo=datetime.UTC),
        )
        assert resp.danger_binary == "elevated"
        assert len(resp.danger_distribution) == 5


class TestStationResponse:
    def test_valid_station(self):
        station = StationResponse(
            station_id="335:CO:SNTL",
            name="Berthoud Summit",
            elevation=11300.0,
            latest_reading=datetime.datetime(2024, 1, 15, 0, 0, 0, tzinfo=datetime.UTC),
            status="active",
        )
        assert station.status == "active"


class TestHealthResponse:
    def test_valid_health(self):
        health = HealthResponse(
            status="healthy",
            model_version="2024-01-01T00:00:00",
            last_ingestion=datetime.datetime(2024, 1, 15, 12, 0, 0, tzinfo=datetime.UTC),
            data_gaps=[],
            stations_online=2,
            stations_total=3,
        )
        assert health.status == "healthy"


# --- POST /predict ---

class TestPredictEndpoint:
    def test_predict_valid(self, client):
        resp = client.post("/predict", json={
            "zone": "front-range",
            "elevation_band": "above_treeline",
            "date": "2024-01-15",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["danger_distribution"]) == 5
        assert body["danger_binary"] in ("elevated", "not_elevated")
        assert "confidence" in body
        assert "problem_types" in body
        assert "explanations" in body
        assert "model_version" in body
        assert "data_freshness" in body

    def test_predict_invalid_zone(self, client):
        resp = client.post("/predict", json={
            "zone": "nonexistent-zone",
            "elevation_band": "above_treeline",
            "date": "2024-01-15",
        })
        assert resp.status_code == 404

    def test_predict_no_model(self, test_db):
        app = create_app()
        app.dependency_overrides[get_db] = lambda: test_db
        app.dependency_overrides[get_model] = lambda: None
        with TestClient(app) as c:
            resp = c.post("/predict", json={
                "zone": "front-range",
                "elevation_band": "above_treeline",
            })
        assert resp.status_code == 503
        app.dependency_overrides.clear()

    def test_predict_missing_fields(self, client):
        resp = client.post("/predict", json={"zone": "front-range"})
        assert resp.status_code == 422

    def test_predict_invalid_elevation_band(self, client):
        resp = client.post("/predict", json={
            "zone": "front-range",
            "elevation_band": "in_the_clouds",
        })
        assert resp.status_code == 422

    def test_predict_defaults_date_to_today(self, client):
        resp = client.post("/predict", json={
            "zone": "front-range",
            "elevation_band": "above_treeline",
        })
        # Should not 422 — date is optional
        assert resp.status_code in (200, 404)


# --- GET /history ---

class TestHistoryEndpoint:
    def test_history_valid(self, client):
        resp = client.get("/history", params={
            "zone": "front-range",
            "elevation_band": "above_treeline",
            "start_date": "2024-01-13",
            "end_date": "2024-01-15",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) > 0
        assert "date" in body[0]
        assert "actual_danger" in body[0]

    def test_history_empty_range(self, client):
        resp = client.get("/history", params={
            "zone": "front-range",
            "elevation_band": "above_treeline",
            "start_date": "2020-01-01",
            "end_date": "2020-01-02",
        })
        assert resp.status_code == 200
        assert resp.json() == []


# --- GET /stations ---

class TestStationsEndpoint:
    def test_stations_all(self, client):
        resp = client.get("/stations")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 3

    def test_stations_filter_by_zone(self, client):
        resp = client.get("/stations", params={"zone": "front-range"})
        assert resp.status_code == 200
        body = resp.json()
        ids = {s["station_id"] for s in body}
        assert "335:CO:SNTL" in ids
        assert "412:CO:SNTL" not in ids

    def test_station_response_shape(self, client):
        resp = client.get("/stations")
        assert resp.status_code == 200
        station = resp.json()[0]
        assert "station_id" in station
        assert "name" in station
        assert "elevation" in station
        assert "status" in station


# --- GET /health ---

class TestHealthEndpoint:
    def test_health_healthy(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("healthy", "degraded")
        assert "model_version" in body
        assert "stations_online" in body
        assert "stations_total" in body

    def test_health_degraded_no_model(self, test_db):
        app = create_app()
        app.dependency_overrides[get_db] = lambda: test_db
        app.dependency_overrides[get_model] = lambda: None
        with TestClient(app) as c:
            resp = c.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("degraded", "unhealthy")
        app.dependency_overrides.clear()
