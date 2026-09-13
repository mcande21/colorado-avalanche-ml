from __future__ import annotations

import duckdb
import pytest
from fastapi.testclient import TestClient

from avalanche_ml.api.app import create_app
from avalanche_ml.api.dependencies import get_db, get_model
from avalanche_ml.api.middleware import add_middleware
from avalanche_ml.api.schemas import PredictionResponse
from tests.test_api import _make_mock_model, _seed_db


@pytest.fixture
def db():
    conn = duckdb.connect(":memory:")
    _seed_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def mock_model():
    return _make_mock_model()


@pytest.fixture
def client(db, mock_model):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_model] = lambda: mock_model
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestFullRequestCycle:
    def test_predict_response_matches_schema(self, client):
        resp = client.post("/predict", json={
            "zone": "front-range",
            "elevation_band": "above_treeline",
            "date": "2024-01-15",
        })
        assert resp.status_code == 200
        body = resp.json()
        parsed = PredictionResponse(**body)
        assert len(parsed.danger_distribution) == 5
        assert parsed.danger_binary in ("elevated", "not_elevated")
        assert 0.0 <= parsed.confidence <= 1.0
        assert isinstance(parsed.problem_types, dict)
        assert isinstance(parsed.explanations, dict)
        assert parsed.model_version is not None
        assert parsed.data_freshness is not None

    def test_predict_then_history_shows_data(self, client):
        predict_resp = client.post("/predict", json={
            "zone": "front-range",
            "elevation_band": "above_treeline",
            "date": "2024-01-15",
        })
        assert predict_resp.status_code == 200

        history_resp = client.get("/history", params={
            "zone": "front-range",
            "elevation_band": "above_treeline",
            "start_date": "2024-01-13",
            "end_date": "2024-01-15",
        })
        assert history_resp.status_code == 200
        records = history_resp.json()
        assert len(records) >= 1
        dates = [r["date"] for r in records]
        assert "2024-01-15" in dates

    def test_stations_have_required_fields(self, client):
        resp = client.get("/stations")
        assert resp.status_code == 200
        stations = resp.json()
        assert len(stations) > 0
        for station in stations:
            assert "station_id" in station
            assert "name" in station
            assert "elevation" in station
            assert "status" in station
            assert station["status"] in ("active", "stale", "offline")

    def test_health_has_all_fields(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "model_version" in body
        assert "last_ingestion" in body
        assert "data_gaps" in body
        assert "stations_online" in body
        assert "stations_total" in body


class TestAuthIntegration:
    def test_auth_enabled_no_key_returns_401(self, db, mock_model):
        app = create_app()
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[get_model] = lambda: mock_model
        add_middleware(app, api_key="test-api-key")
        with TestClient(app) as c:
            resp = c.get("/health")
        assert resp.status_code == 401
        app.dependency_overrides.clear()

    def test_auth_enabled_valid_key_returns_200(self, db, mock_model):
        app = create_app()
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[get_model] = lambda: mock_model
        add_middleware(app, api_key="test-api-key")
        with TestClient(app) as c:
            resp = c.get("/health", headers={"X-API-Key": "test-api-key"})
        assert resp.status_code == 200
        app.dependency_overrides.clear()


class TestRateLimitIntegration:
    def test_101_rapid_requests_last_gets_429(self, db, mock_model):
        app = create_app()
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[get_model] = lambda: mock_model
        add_middleware(app)
        with TestClient(app) as c:
            for i in range(100):
                r = c.get("/health")
                assert r.status_code == 200, f"Request {i+1} failed unexpectedly"
            resp = c.get("/health")
            assert resp.status_code == 429
            assert "Retry-After" in resp.headers
        app.dependency_overrides.clear()
