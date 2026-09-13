from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import duckdb
import pytest

from avalanche_ml.db.schema import create_tables


class TestDailyIngestionScheduler:
    @pytest.fixture
    def db(self):
        conn = duckdb.connect(":memory:")
        create_tables(conn)
        yield conn
        conn.close()

    def test_run_daily_ingestion_executes_all_steps(self, db):
        from avalanche_ml.ingestion.scheduler import run_daily_ingestion

        with (
            patch(
                "avalanche_ml.ingestion.scheduler._run_snotel_ingest",
                new_callable=AsyncMock,
            ) as mock_snotel,
            patch(
                "avalanche_ml.ingestion.scheduler._run_caic_ingest",
                new_callable=AsyncMock,
            ) as mock_caic,
            patch(
                "avalanche_ml.ingestion.scheduler._run_feature_computation",
            ) as mock_features,
            patch(
                "avalanche_ml.ingestion.scheduler._run_alignment",
            ) as mock_alignment,
        ):
            run_daily_ingestion(db)

            mock_snotel.assert_called_once()
            mock_caic.assert_called_once()
            mock_features.assert_called_once()
            mock_alignment.assert_called_once()

    def test_one_step_failing_does_not_stop_rest(self, db):
        from avalanche_ml.ingestion.scheduler import run_daily_ingestion

        with (
            patch(
                "avalanche_ml.ingestion.scheduler._run_snotel_ingest",
                new_callable=AsyncMock,
                side_effect=RuntimeError("SNOTEL down"),
            ),
            patch(
                "avalanche_ml.ingestion.scheduler._run_caic_ingest",
                new_callable=AsyncMock,
            ) as mock_caic,
            patch(
                "avalanche_ml.ingestion.scheduler._run_feature_computation",
            ) as mock_features,
            patch(
                "avalanche_ml.ingestion.scheduler._run_alignment",
            ) as mock_alignment,
        ):
            run_daily_ingestion(db)

            mock_caic.assert_called_once()
            mock_features.assert_called_once()
            mock_alignment.assert_called_once()

    def test_logging_captures_success_and_failure(self, db, caplog):
        from avalanche_ml.ingestion.scheduler import run_daily_ingestion

        with (
            patch(
                "avalanche_ml.ingestion.scheduler._run_snotel_ingest",
                new_callable=AsyncMock,
                side_effect=RuntimeError("SNOTEL down"),
            ),
            patch(
                "avalanche_ml.ingestion.scheduler._run_caic_ingest",
                new_callable=AsyncMock,
            ),
            patch(
                "avalanche_ml.ingestion.scheduler._run_feature_computation",
            ),
            patch(
                "avalanche_ml.ingestion.scheduler._run_alignment",
            ),
            caplog.at_level(logging.INFO, logger="avalanche_ml.ingestion.scheduler"),
        ):
            run_daily_ingestion(db)

        log_text = caplog.text
        assert "SNOTEL" in log_text
        assert "failed" in log_text.lower() or "error" in log_text.lower()
        success_count = sum(
            1
            for r in caplog.records
            if "completed" in r.message.lower() or "success" in r.message.lower()
        )
        assert success_count >= 3
