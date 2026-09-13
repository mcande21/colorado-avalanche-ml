from __future__ import annotations

import datetime

import duckdb

WINDOWS = [24, 48, 72, 96, 120]
WINDOW_DAYS = {24: 1, 48: 2, 72: 3, 96: 4, 120: 5}
COVERAGE_THRESHOLD = 0.5

TEMP_FEATURES = ["temp_mean", "temp_min", "temp_max", "temp_std", "temp_range", "temp_trend"]
PRECIP_FEATURES = ["precip_sum", "precip_max", "precip_days", "precip_intensity"]
WIND_FEATURES = ["wind_mean", "wind_max", "wind_std", "wind_direction_mode"]
SNOW_FEATURES = ["swe_change", "snow_depth_change", "swe_rate", "new_snow"]

ALL_FEATURES = TEMP_FEATURES + PRECIP_FEATURES + WIND_FEATURES + SNOW_FEATURES


def get_feature_columns() -> list[str]:
    cols = []
    for w in WINDOWS:
        for feat in ALL_FEATURES:
            cols.append(f"{feat}_{w}h")
    return cols


def create_weather_features_table(conn: duckdb.DuckDBPyConnection) -> None:
    coverage_cols = ",\n".join(f"        coverage_{w}h DOUBLE" for w in WINDOWS)
    feature_cols = ",\n".join(f"        {col} DOUBLE" for col in get_feature_columns())
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS weather_features (
            station_id VARCHAR NOT NULL,
            date DATE NOT NULL,
            {coverage_cols},
            {feature_cols},
            PRIMARY KEY (station_id, date)
        )
    """)


def _build_query(
    station_ids: list[str] | None = None,
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
) -> tuple[str, list]:
    window_defs = []
    for w in WINDOWS:
        days_back = WINDOW_DAYS[w] - 1
        window_defs.append(
            f"w{w} AS (PARTITION BY station_id ORDER BY date "
            f"RANGE BETWEEN INTERVAL '{days_back}' DAY PRECEDING AND CURRENT ROW)"
        )

    raw_cols = []
    for w in WINDOWS:
        expected = WINDOW_DAYS[w]
        raw_cols.append(f"COUNT(*) OVER w{w} AS cnt_{w}")
        raw_cols.append(f"AVG(air_temp_mean_f) OVER w{w} AS raw_temp_mean_{w}")
        raw_cols.append(f"MIN(air_temp_min_f) OVER w{w} AS raw_temp_min_{w}")
        raw_cols.append(f"MAX(air_temp_max_f) OVER w{w} AS raw_temp_max_{w}")
        raw_cols.append(f"STDDEV_SAMP(air_temp_mean_f) OVER w{w} AS raw_temp_std_{w}")
        raw_cols.append(
            f"MAX(air_temp_max_f) OVER w{w} - MIN(air_temp_min_f) OVER w{w} "
            f"AS raw_temp_range_{w}"
        )
        raw_cols.append(
            f"REGR_SLOPE(air_temp_mean_f, "
            f"EXTRACT(EPOCH FROM CAST(date AS TIMESTAMP)) / 86400.0) OVER w{w} "
            f"AS raw_temp_trend_{w}"
        )
        raw_cols.append(f"SUM(precip_increment_inches) OVER w{w} AS raw_precip_sum_{w}")
        raw_cols.append(f"MAX(precip_increment_inches) OVER w{w} AS raw_precip_max_{w}")
        raw_cols.append(
            f"SUM(CASE WHEN precip_increment_inches > 0 THEN 1 ELSE 0 END) OVER w{w} "
            f"AS raw_precip_days_{w}"
        )
        raw_cols.append(
            f"swe_inches - FIRST_VALUE(swe_inches IGNORE NULLS) OVER w{w} "
            f"AS raw_swe_change_{w}"
        )
        raw_cols.append(
            f"snow_depth_inches - FIRST_VALUE(snow_depth_inches IGNORE NULLS) OVER w{w} "
            f"AS raw_snow_depth_change_{w}"
        )

    where_parts = []
    params: list = []
    if station_ids:
        placeholders = ", ".join("?" for _ in station_ids)
        where_parts.append(f"station_id IN ({placeholders})")
        params.extend(station_ids)
    if start_date:
        lookback = datetime.timedelta(days=WINDOW_DAYS[120] - 1)
        where_parts.append("date >= ?")
        params.append(start_date - lookback)
    if end_date:
        where_parts.append("date <= ?")
        params.append(end_date)

    where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    date_filter = ""
    if start_date:
        date_filter = f" AND date >= '{start_date.isoformat()}'"
    if end_date:
        if date_filter:
            date_filter += f" AND date <= '{end_date.isoformat()}'"
        else:
            date_filter = f" AND date <= '{end_date.isoformat()}'"

    outer_cols = ["station_id", "date"]
    for w in WINDOWS:
        expected = WINDOW_DAYS[w]
        cov = f"CAST(cnt_{w} AS DOUBLE) / {expected}.0"
        outer_cols.append(f"{cov} AS coverage_{w}h")

    for w in WINDOWS:
        expected = WINDOW_DAYS[w]
        cov_check = f"CAST(cnt_{w} AS DOUBLE) / {expected}.0 >= {COVERAGE_THRESHOLD}"

        outer_cols.append(
            f"CASE WHEN {cov_check} THEN raw_temp_mean_{w} ELSE NULL END AS temp_mean_{w}h"
        )
        outer_cols.append(
            f"CASE WHEN {cov_check} THEN raw_temp_min_{w} ELSE NULL END AS temp_min_{w}h"
        )
        outer_cols.append(
            f"CASE WHEN {cov_check} THEN raw_temp_max_{w} ELSE NULL END AS temp_max_{w}h"
        )
        outer_cols.append(
            f"CASE WHEN {cov_check} THEN raw_temp_std_{w} ELSE NULL END AS temp_std_{w}h"
        )
        outer_cols.append(
            f"CASE WHEN {cov_check} THEN raw_temp_range_{w} ELSE NULL END "
            f"AS temp_range_{w}h"
        )
        outer_cols.append(
            f"CASE WHEN {cov_check} THEN raw_temp_trend_{w} ELSE NULL END "
            f"AS temp_trend_{w}h"
        )
        outer_cols.append(
            f"CASE WHEN {cov_check} THEN raw_precip_sum_{w} ELSE NULL END "
            f"AS precip_sum_{w}h"
        )
        outer_cols.append(
            f"CASE WHEN {cov_check} THEN raw_precip_max_{w} ELSE NULL END "
            f"AS precip_max_{w}h"
        )
        outer_cols.append(
            f"CASE WHEN {cov_check} THEN CAST(raw_precip_days_{w} AS DOUBLE) ELSE NULL END "
            f"AS precip_days_{w}h"
        )
        outer_cols.append(
            f"CASE WHEN {cov_check} AND raw_precip_days_{w} > 0 "
            f"THEN raw_precip_sum_{w} / raw_precip_days_{w} ELSE NULL END "
            f"AS precip_intensity_{w}h"
        )
        for wf in WIND_FEATURES:
            outer_cols.append(f"NULL AS {wf}_{w}h")
        outer_cols.append(
            f"CASE WHEN {cov_check} THEN raw_swe_change_{w} ELSE NULL END "
            f"AS swe_change_{w}h"
        )
        outer_cols.append(
            f"CASE WHEN {cov_check} THEN raw_snow_depth_change_{w} ELSE NULL END "
            f"AS snow_depth_change_{w}h"
        )
        outer_cols.append(
            f"CASE WHEN {cov_check} THEN raw_swe_change_{w} / {expected}.0 ELSE NULL END "
            f"AS swe_rate_{w}h"
        )
        outer_cols.append(
            f"CASE WHEN {cov_check} THEN GREATEST(raw_snow_depth_change_{w}, 0) "
            f"ELSE NULL END AS new_snow_{w}h"
        )

    all_insert_cols = ["station_id", "date"]
    all_insert_cols.extend(f"coverage_{w}h" for w in WINDOWS)
    all_insert_cols.extend(get_feature_columns())

    select_clause = ",\n            ".join(outer_cols)
    raw_clause = ",\n                ".join(raw_cols)
    window_clause = ",\n                ".join(window_defs)
    update_cols = [col for col in all_insert_cols if col not in ("station_id", "date")]
    update_clause = ",\n            ".join(f"{col} = EXCLUDED.{col}" for col in update_cols)

    query = f"""
        INSERT INTO weather_features ({', '.join(all_insert_cols)})
        SELECT {select_clause}
        FROM (
            SELECT
                station_id,
                date,
                {raw_clause}
            FROM snotel_daily
            {where_clause}
            WINDOW {window_clause}
        ) raw
        WHERE 1=1{date_filter}
        ON CONFLICT (station_id, date) DO UPDATE SET
            {update_clause}
    """
    return query, params


def compute_weather_features(
    conn: duckdb.DuckDBPyConnection,
    station_ids: list[str] | None = None,
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
) -> int:
    query, params = _build_query(station_ids, start_date, end_date)
    if params:
        conn.execute(query, params)
    else:
        conn.execute(query)
    row = conn.execute("SELECT COUNT(*) FROM weather_features").fetchone()
    return row[0] if row else 0
