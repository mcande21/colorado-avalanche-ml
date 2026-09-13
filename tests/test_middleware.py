from __future__ import annotations

import time
from unittest.mock import patch

import duckdb
import pytest
from fastapi.testclient import TestClient

from avalanche_ml.api.app import create_app
from avalanche_ml.api.dependencies import get_db, get_model
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


def _make_app_with_middleware(db_conn, model, api_key=None):
    from avalanche_ml.api.middleware import add_middleware

    app = create_app()
    app.dependency_overrides[get_db] = lambda: db_conn
    app.dependency_overrides[get_model] = lambda: model
    add_middleware(app, api_key=api_key)
    return app


class TestAuthMiddleware:
    def test_no_api_key_env_auth_disabled(self, db, mock_model):
        app = _make_app_with_middleware(db, mock_model, api_key=None)
        with TestClient(app) as c:
            resp = c.get("/health")
        assert resp.status_code == 200
        app.dependency_overrides.clear()

    def test_api_key_set_valid_key_passes(self, db, mock_model):
        app = _make_app_with_middleware(db, mock_model, api_key="secret-key-123")
        with TestClient(app) as c:
            resp = c.get("/health", headers={"X-API-Key": "secret-key-123"})
        assert resp.status_code == 200
        app.dependency_overrides.clear()

    def test_api_key_set_invalid_key_returns_401(self, db, mock_model):
        app = _make_app_with_middleware(db, mock_model, api_key="secret-key-123")
        with TestClient(app) as c:
            resp = c.get("/health", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401
        app.dependency_overrides.clear()

    def test_api_key_set_missing_key_returns_401(self, db, mock_model):
        app = _make_app_with_middleware(db, mock_model, api_key="secret-key-123")
        with TestClient(app) as c:
            resp = c.get("/health")
        assert resp.status_code == 401
        app.dependency_overrides.clear()


class TestRateLimitMiddleware:
    def test_under_limit_passes(self, db, mock_model):
        app = _make_app_with_middleware(db, mock_model)
        with TestClient(app) as c:
            resp = c.get("/health")
        assert resp.status_code == 200
        app.dependency_overrides.clear()

    def test_exceeding_limit_returns_429(self, db, mock_model):
        app = _make_app_with_middleware(db, mock_model)
        with TestClient(app) as c:
            for _ in range(100):
                c.get("/health")
            resp = c.get("/health")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        app.dependency_overrides.clear()

    def test_rate_limit_resets_after_window(self, db, mock_model):
        app = _make_app_with_middleware(db, mock_model)
        with TestClient(app) as c:
            for _ in range(100):
                c.get("/health")
            resp = c.get("/health")
            assert resp.status_code == 429

            with patch("avalanche_ml.api.middleware.time") as mock_time:
                mock_time.monotonic.return_value = time.monotonic() + 61
                resp = c.get("/health")
            assert resp.status_code == 200
        app.dependency_overrides.clear()
