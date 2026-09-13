from __future__ import annotations

import datetime
from collections import defaultdict

import duckdb

PHYSICS_FEATURES = [
    "temp_gradient_days",
    "temp_gradient_consec_days",
    "surface_hoar_index",
    "wind_slab_loading",
    "rain_on_snow_hours",
    "rain_on_snow_amount",
    "snow_depth_anomaly",
    "early_season_flag",
    "freeze_thaw_cycles",
]

F_INCH_TO_K_M = 5.0 / 9.0 / 0.0254
GRADIENT_THRESHOLD = 10.0
SURFACE_HOAR_TEMP_F = 23.0
SURFACE_HOAR_RANGE_F = 20.0
BURIAL_PRECIP_THRESHOLD = 0.5
ROS_PRECIP_RESET = 5.0 / 25.4
FREEZE_THAW_LOOKBACK = 14
FREEZE_TEMP_F = 28.4
CLIMATOLOGY_MIN_YEARS = 10
WIND_SLAB_WINDOW = 3
EARLY_SEASON_CUTOFF_MONTH = 1
EARLY_SEASON_CUTOFF_DAY = 15


def create_physics_features_table(conn: duckdb.DuckDBPyConnection) -> None:
    feature_cols = ",\n".join(f"        {col} DOUBLE" for col in PHYSICS_FEATURES)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS physics_features (
            station_id VARCHAR NOT NULL,
            date DATE NOT NULL,
{feature_cols},
            PRIMARY KEY (station_id, date)
        )
    """)


def _season_start(d: datetime.date) -> datetime.date:
    if d.month >= 10:
        return datetime.date(d.year, 10, 1)
    return datetime.date(d.year - 1, 10, 1)


def _gradient(temp_range_f: float | None, depth_inches: float | None) -> float | None:
    if temp_range_f is None or depth_inches is None or depth_inches <= 0:
        return None
    return (temp_range_f / depth_inches) * F_INCH_TO_K_M


def _compute_climatology(
    conn: duckdb.DuckDBPyConnection,
    station_ids: list[str] | None = None,
) -> dict[tuple[str, int], tuple[float, float, float, int]]:
    where_parts = ["snow_depth_inches IS NOT NULL"]
    params: list = []
    if station_ids:
        placeholders = ", ".join("?" for _ in station_ids)
        where_parts.append(f"station_id IN ({placeholders})")
        params = list(station_ids)

    where = "WHERE " + " AND ".join(where_parts)

    rows = conn.execute(f"""
        SELECT
            station_id,
            EXTRACT(DOY FROM date) AS doy,
            MEDIAN(snow_depth_inches) AS median_depth,
            STDDEV_SAMP(snow_depth_inches) AS std_depth,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY snow_depth_inches) AS p25_depth,
            COUNT(DISTINCT EXTRACT(YEAR FROM date)) AS n_years
        FROM snotel_daily
        {where}
        GROUP BY station_id, EXTRACT(DOY FROM date)
    """, params).fetchall()

    result = {}
    for station_id, doy, median_d, std_d, p25_d, n_years in rows:
        result[(station_id, int(doy))] = (
            float(median_d) if median_d is not None else 0.0,
            float(std_d) if std_d is not None else 0.0,
            float(p25_d) if p25_d is not None else 0.0,
            int(n_years),
        )
    return result


def compute_physics_features(
    conn: duckdb.DuckDBPyConnection,
    station_ids: list[str] | None = None,
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
) -> int:
    climatology = _compute_climatology(conn, station_ids)

    where_parts: list[str] = []
    params: list = []
    if station_ids:
        placeholders = ", ".join("?" for _ in station_ids)
        where_parts.append(f"station_id IN ({placeholders})")
        params.extend(station_ids)
    if start_date:
        season = _season_start(start_date)
        lookback = datetime.timedelta(days=max(FREEZE_THAW_LOOKBACK, WIND_SLAB_WINDOW) + 1)
        effective_start = min(start_date - lookback, season)
        where_parts.append("date >= ?")
        params.append(effective_start)
    if end_date:
        where_parts.append("date <= ?")
        params.append(end_date)

    where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    rows = conn.execute(f"""
        SELECT station_id, date, swe_inches, snow_depth_inches,
               air_temp_min_f, air_temp_max_f, air_temp_mean_f,
               precip_increment_inches
        FROM snotel_daily
        {where_clause}
        ORDER BY station_id, date
    """, params).fetchall()

    station_data: dict[str, list] = defaultdict(list)
    for row in rows:
        station_data[row[0]].append(row)

    results = []
    for station_id, data in station_data.items():
        tgd_count = 0
        tgd_consec = 0
        tgd_consec_max = 0
        shi_count = 0
        last_season_start: datetime.date | None = None

        for i, row in enumerate(data):
            _, date_val, _, snow_depth, temp_min, temp_max, temp_mean, precip = row
            if isinstance(date_val, str):
                date_val = datetime.date.fromisoformat(date_val)

            temp_range = None
            if temp_max is not None and temp_min is not None:
                temp_range = temp_max - temp_min

            precip_val = precip if precip is not None else 0.0

            current_season = _season_start(date_val)
            if last_season_start is None or current_season != last_season_start:
                if last_season_start is not None:
                    tgd_count = 0
                    tgd_consec = 0
                    tgd_consec_max = 0
                    shi_count = 0
                last_season_start = current_season

            is_ros = (
                precip_val > 0
                and temp_max is not None and temp_max > 32.0
                and snow_depth is not None and snow_depth > 0
            )
            ros_hours = 1 if is_ros else 0
            ros_amount = precip_val if is_ros else 0.0

            if is_ros and precip_val > ROS_PRECIP_RESET:
                tgd_count = 0
                tgd_consec = 0

            gradient = _gradient(temp_range, snow_depth)
            if gradient is not None and gradient > GRADIENT_THRESHOLD:
                tgd_count += 1
                tgd_consec += 1
                tgd_consec_max = max(tgd_consec_max, tgd_consec)
            else:
                tgd_consec = 0

            if precip_val > BURIAL_PRECIP_THRESHOLD:
                shi_count = 0
            is_hoar_day = (
                precip_val == 0
                and temp_mean is not None and temp_mean < SURFACE_HOAR_TEMP_F
                and temp_range is not None and temp_range > SURFACE_HOAR_RANGE_F
            )
            if is_hoar_day:
                shi_count += 1

            wind_slab = 0.0
            ws_start = date_val - datetime.timedelta(days=WIND_SLAB_WINDOW - 1)
            for j in range(i, -1, -1):
                jdate = data[j][1]
                if isinstance(jdate, str):
                    jdate = datetime.date.fromisoformat(jdate)
                if jdate < ws_start:
                    break
                p = data[j][7]
                if p is not None:
                    wind_slab += p

            ft_count = 0
            ft_start = date_val - datetime.timedelta(days=FREEZE_THAW_LOOKBACK - 1)
            for j in range(i, -1, -1):
                jdate = data[j][1]
                if isinstance(jdate, str):
                    jdate = datetime.date.fromisoformat(jdate)
                if jdate < ft_start:
                    break
                jmax = data[j][5]
                jmin = data[j][4]
                jdepth = data[j][3]
                if (jmax is not None and jmax > 32.0
                        and jmin is not None and jmin < FREEZE_TEMP_F
                        and jdepth is not None and jdepth > 0):
                    ft_count += 1

            doy = date_val.timetuple().tm_yday
            clim_key = (station_id, doy)
            snow_anomaly = None
            early_flag = 0
            if clim_key in climatology:
                median_d, std_d, p25_d, n_years = climatology[clim_key]
                if n_years >= CLIMATOLOGY_MIN_YEARS and snow_depth is not None:
                    if std_d > 0:
                        snow_anomaly = (snow_depth - median_d) / std_d
                    else:
                        snow_anomaly = 0.0
                if ((date_val.month < EARLY_SEASON_CUTOFF_MONTH
                        or (date_val.month == EARLY_SEASON_CUTOFF_MONTH
                            and date_val.day < EARLY_SEASON_CUTOFF_DAY))
                        and n_years >= CLIMATOLOGY_MIN_YEARS
                        and snow_depth is not None
                        and snow_depth < p25_d):
                    early_flag = 1

            if start_date and date_val < start_date:
                continue
            if end_date and date_val > end_date:
                continue

            results.append((
                station_id, date_val,
                float(tgd_count), float(tgd_consec_max),
                float(shi_count), float(wind_slab),
                float(ros_hours), float(ros_amount), snow_anomaly,
                float(early_flag), float(ft_count),
            ))

    if results:
        cols = ["station_id", "date"] + PHYSICS_FEATURES
        placeholders = ", ".join("?" for _ in cols)
        update_clause = ", ".join(
            f"{col} = EXCLUDED.{col}" for col in PHYSICS_FEATURES
        )
        conn.executemany(f"""
            INSERT INTO physics_features ({', '.join(cols)})
            VALUES ({placeholders})
            ON CONFLICT (station_id, date) DO UPDATE SET
                {update_clause}
        """, results)

    row = conn.execute("SELECT COUNT(*) FROM physics_features").fetchone()
    return row[0] if row else 0
