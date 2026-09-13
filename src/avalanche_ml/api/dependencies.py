from __future__ import annotations

import os
from pathlib import Path

import duckdb

_db_conn: duckdb.DuckDBPyConnection | None = None
_model = None


def get_db() -> duckdb.DuckDBPyConnection:
    global _db_conn
    if _db_conn is None:
        db_path = os.environ.get("DUCKDB_PATH", "data/avalanche.duckdb")
        _db_conn = duckdb.connect(db_path)
    return _db_conn


def get_model():
    global _model
    if _model is None:
        model_path = os.environ.get("MODEL_PATH")
        if model_path and Path(model_path).exists():
            from avalanche_ml.models.rf_stage2 import TwoStageRFPipeline
            _model = TwoStageRFPipeline.load(Path(model_path))
    return _model
