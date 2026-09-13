from __future__ import annotations

import datetime

import duckdb
import pandas as pd

from avalanche_ml.features.physics import PHYSICS_FEATURES
from avalanche_ml.features.weather import get_feature_columns

ELEVATION_BANDS = ("above_treeline", "near_treeline", "below_treeline")
PROBLEM_TYPE_FLAGS = ["persistent_slab", "storm_slab", "loose_wet"]

TRAIN_END = datetime.date(2024, 6, 30)
VAL_START = datetime.date(2024, 10, 1)
VAL_END = datetime.date(2025, 1, 31)
TEST_START = datetime.date(2025, 2, 1)


def create_training_matrix_table(conn: duckdb.DuckDBPyConnection) -> None:
    weather_cols = ",\n".join(f"    {c} DOUBLE" for c in get_feature_columns())
    physics_cols = ",\n".join(f"    {c} DOUBLE" for c in PHYSICS_FEATURES)
    problem_cols = ",\n".join(f"    {c} INTEGER DEFAULT 0" for c in PROBLEM_TYPE_FLAGS)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS training_matrix (
            station_id VARCHAR NOT NULL,
            date DATE NOT NULL,
            elevation_band VARCHAR NOT NULL,
            zone_id VARCHAR NOT NULL,
            {weather_cols},
            {physics_cols},
            danger_level INTEGER NOT NULL,
            {problem_cols},
            elevation DOUBLE,
            latitude DOUBLE,
            longitude DOUBLE,
            month INTEGER,
            day_of_year INTEGER,
            days_since_season_start INTEGER,
            PRIMARY KEY (station_id, date, elevation_band, zone_id)
        )
    """)


def _season_start(d: datetime.date) -> datetime.date:
    if d.month >= 10:
        return datetime.date(d.year, 10, 1)
    return datetime.date(d.year - 1, 10, 1)


def assemble_training_matrix(conn: duckdb.DuckDBPyConnection) -> int:
    conn.execute("DELETE FROM training_matrix")

    weather_cols_select = ", ".join(f"w.{c}" for c in get_feature_columns())
    physics_cols_select = ", ".join(f"p.{c}" for c in PHYSICS_FEATURES)
    weather_cols_insert = ", ".join(get_feature_columns())
    physics_cols_insert = ", ".join(PHYSICS_FEATURES)

    conn.execute(f"""
        INSERT INTO training_matrix (
            station_id, date, elevation_band, zone_id,
            {weather_cols_insert},
            {physics_cols_insert},
            danger_level,
            elevation, latitude, longitude,
            month, day_of_year, days_since_season_start
        )
        SELECT
            w.station_id,
            w.date,
            dr.elevation_band,
            zsm.zone_id,
            {weather_cols_select},
            {physics_cols_select},
            dr.danger_level,
            s.elevation,
            s.latitude,
            s.longitude,
            EXTRACT(MONTH FROM w.date),
            EXTRACT(DOY FROM w.date),
            CAST(w.date - (
                CASE WHEN EXTRACT(MONTH FROM w.date) >= 10
                     THEN MAKE_DATE(EXTRACT(YEAR FROM w.date)::INTEGER, 10, 1)
                     ELSE MAKE_DATE((EXTRACT(YEAR FROM w.date) - 1)::INTEGER, 10, 1)
                END
            ) AS INTEGER)
        FROM weather_features w
        INNER JOIN physics_features p
            ON w.station_id = p.station_id AND w.date = p.date
        INNER JOIN zone_station_mapping zsm
            ON w.station_id = zsm.station_id
        INNER JOIN danger_ratings dr
            ON zsm.zone_id = dr.zone_id AND w.date = dr.date
        INNER JOIN stations s
            ON w.station_id = s.station_id
        WHERE dr.danger_level IS NOT NULL
    """)

    _update_problem_type_flags(conn)

    row = conn.execute("SELECT COUNT(*) FROM training_matrix").fetchone()
    return row[0] if row else 0


def _update_problem_type_flags(conn: duckdb.DuckDBPyConnection) -> None:
    problem_map = {
        "Persistent Slab": "persistent_slab",
        "Storm Slab": "storm_slab",
        "Loose Wet": "loose_wet",
    }
    for problem_type, column in problem_map.items():
        conn.execute(f"""
            UPDATE training_matrix tm
            SET {column} = 1
            WHERE EXISTS (
                SELECT 1 FROM problem_types pt
                INNER JOIN zone_station_mapping zsm
                    ON pt.zone_id = zsm.zone_id
                WHERE zsm.station_id = tm.station_id
                  AND zsm.zone_id = tm.zone_id
                  AND pt.date = tm.date
                  AND pt.problem_type = ?
            )
        """, [problem_type])


def temporal_split(
    conn: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = conn.execute(
        f"SELECT * FROM training_matrix WHERE date <= '{TRAIN_END.isoformat()}'"
    ).fetchdf()
    val = conn.execute(
        f"SELECT * FROM training_matrix WHERE date > '{TRAIN_END.isoformat()}' "
        f"AND date <= '{VAL_END.isoformat()}'"
    ).fetchdf()
    test = conn.execute(
        f"SELECT * FROM training_matrix WHERE date > '{VAL_END.isoformat()}'"
    ).fetchdf()
    return train, val, test


def class_distribution_report(conn: duckdb.DuckDBPyConnection) -> dict:
    danger_rows = conn.execute("""
        SELECT danger_level, COUNT(*) as cnt
        FROM training_matrix
        GROUP BY danger_level
        ORDER BY danger_level
    """).fetchall()
    total = sum(r[1] for r in danger_rows)
    danger_levels = [
        {"level": r[0], "count": r[1], "pct": round(r[1] / total * 100, 1) if total else 0}
        for r in danger_rows
    ]

    problem_counts = {}
    for col in PROBLEM_TYPE_FLAGS:
        row = conn.execute(
            f"SELECT SUM({col}) FROM training_matrix"
        ).fetchone()
        problem_counts[col] = int(row[0]) if row and row[0] else 0

    total_rows = conn.execute("SELECT COUNT(*) FROM training_matrix").fetchone()[0]
    weather_cols = get_feature_columns()
    null_check = " OR ".join(f"{c} IS NULL" for c in weather_cols[:5])
    incomplete = conn.execute(
        f"SELECT COUNT(*) FROM training_matrix WHERE {null_check}"
    ).fetchone()[0]
    complete = total_rows - incomplete

    return {
        "danger_levels": danger_levels,
        "problem_types": problem_counts,
        "coverage": {
            "total_rows": total_rows,
            "complete_rows": complete,
            "coverage_pct": round(complete / total_rows * 100, 1) if total_rows else 0,
        },
    }


def export_to_parquet(conn: duckdb.DuckDBPyConnection, path: str) -> None:
    conn.execute(f"COPY training_matrix TO '{path}' (FORMAT PARQUET)")
