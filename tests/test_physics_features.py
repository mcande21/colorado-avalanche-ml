import datetime

import pytest

from avalanche_ml.db.schema import create_tables
from avalanche_ml.features.physics import (
    PHYSICS_FEATURES,
    compute_physics_features,
    create_physics_features_table,
)


@pytest.fixture
def physics_db(db):
    create_tables(db)
    create_physics_features_table(db)
    return db


def _insert_daily(conn, station_id, date_str, swe, snow_depth, temp_min, temp_max, temp_mean,
                   precip):
    conn.execute("""
        INSERT INTO snotel_daily (station_id, date, swe_inches, snow_depth_inches,
            air_temp_min_f, air_temp_max_f, air_temp_mean_f, precip_increment_inches)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [station_id, date_str, swe, snow_depth, temp_min, temp_max, temp_mean, precip])


def _insert_years_of_data(conn, station_id, month, day, snow_depths, start_year=2010):
    """Insert one row per year for the same month/day, used for climatology tests."""
    for i, depth in enumerate(snow_depths):
        year = start_year + i
        date_str = f"{year}-{month:02d}-{day:02d}"
        _insert_daily(conn, station_id, date_str, depth * 0.3, depth, 10.0, 30.0, 20.0, 0.0)


class TestTempGradientCalculation:
    def test_gradient_above_threshold(self, physics_db):
        """High gradient (thin snowpack + large temp range) is a faceting day."""
        # range=20F, depth=30in -> gradient=(20/30)*21.87=14.58 K/m > 10
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       10.0, 30.0, 5.0, 25.0, 15.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT temp_gradient_days FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row is not None
        assert row[0] == 1

    def test_gradient_below_threshold(self, physics_db):
        """Low gradient (deep snowpack + small temp range) is not a faceting day."""
        # range=10F, depth=60in -> gradient=(10/60)*21.87=3.65 K/m < 10
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       20.0, 60.0, 20.0, 30.0, 25.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT temp_gradient_days FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row is not None
        assert row[0] == 0

    def test_zero_snow_depth(self, physics_db):
        """Zero snow depth: gradient undefined, not a faceting day."""
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       0.0, 0.0, 5.0, 25.0, 15.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT temp_gradient_days FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row is not None
        assert row[0] == 0

    def test_null_snow_depth(self, physics_db):
        """NULL snow depth: gradient undefined, not a faceting day."""
        physics_db.execute("""
            INSERT INTO snotel_daily (station_id, date, swe_inches, snow_depth_inches,
                air_temp_min_f, air_temp_max_f, air_temp_mean_f, precip_increment_inches)
            VALUES ('TEST:CO:SNTL', '2024-01-05', NULL, NULL, 5.0, 25.0, 15.0, 0.0)
        """)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT temp_gradient_days FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row is not None
        assert row[0] == 0


class TestTempGradientCumulative:
    def test_accumulates_across_days(self, physics_db):
        """Multiple faceting days accumulate."""
        for day in [3, 4, 5]:
            _insert_daily(physics_db, "TEST:CO:SNTL", f"2024-01-{day:02d}",
                           10.0, 30.0, 5.0, 25.0, 15.0, 0.0)
        compute_physics_features(physics_db)
        rows = physics_db.execute("""
            SELECT date, temp_gradient_days FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL'
            ORDER BY date
        """).fetchall()
        assert len(rows) == 3
        assert rows[0][1] == 1
        assert rows[1][1] == 2
        assert rows[2][1] == 3

    def test_resets_at_season_start(self, physics_db):
        """Cumulative count resets on October 1 (season start)."""
        for d in [29, 30]:
            _insert_daily(physics_db, "TEST:CO:SNTL", f"2023-09-{d:02d}",
                           10.0, 30.0, 5.0, 25.0, 15.0, 0.0)
        _insert_daily(physics_db, "TEST:CO:SNTL", "2023-10-01",
                       10.0, 30.0, 5.0, 25.0, 15.0, 0.0)
        compute_physics_features(physics_db)
        rows = physics_db.execute("""
            SELECT date, temp_gradient_days FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL'
            ORDER BY date
        """).fetchall()
        assert rows[0][1] == 1  # Sep 29
        assert rows[1][1] == 2  # Sep 30
        assert rows[2][1] == 1  # Oct 1: reset, then this day counts

    def test_resets_on_significant_rain_on_snow(self, physics_db):
        """Significant rain-on-snow event (>5mm/0.197in) resets TGD counter."""
        # 2 faceting days
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-03",
                       10.0, 30.0, 5.0, 25.0, 15.0, 0.0)
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-04",
                       10.0, 30.0, 5.0, 25.0, 15.0, 0.0)
        # ROS event: precip=0.3 (>0.197), temp_max=35 (>32), depth=30
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       10.0, 30.0, 28.0, 35.0, 31.0, 0.3)
        # Another faceting day after reset
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-06",
                       10.0, 30.0, 5.0, 25.0, 15.0, 0.0)
        compute_physics_features(physics_db)
        rows = physics_db.execute("""
            SELECT date, temp_gradient_days FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL'
            ORDER BY date
        """).fetchall()
        assert rows[0][1] == 1  # Jan 3
        assert rows[1][1] == 2  # Jan 4
        # Jan 5: ROS resets to 0, then gradient check (range=7F, depth=30 -> 5.1 K/m < 10)
        assert rows[2][1] == 0
        assert rows[3][1] == 1  # Jan 6: re-accumulated


class TestSurfaceHoarIndex:
    def test_hoar_conditions_met(self, physics_db):
        """Clear sky + cold + large diurnal range: surface hoar day."""
        # precip=0, mean=15F (<23), range=25F (>20)
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       10.0, 30.0, 2.5, 27.5, 15.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT surface_hoar_index FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row is not None
        assert row[0] == 1

    def test_hoar_not_met_has_precip(self, physics_db):
        """Precipitation present: not a surface hoar day."""
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       10.0, 30.0, 2.5, 27.5, 15.0, 0.1)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT surface_hoar_index FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row is not None
        assert row[0] == 0

    def test_hoar_not_met_too_warm(self, physics_db):
        """Too warm (mean > 23F): not a surface hoar day."""
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       10.0, 30.0, 20.0, 45.0, 30.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT surface_hoar_index FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row is not None
        assert row[0] == 0

    def test_hoar_resets_on_burial(self, physics_db):
        """Significant snowfall (>0.5in) resets hoar counter."""
        # Day 1: hoar conditions
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-03",
                       10.0, 30.0, 2.5, 27.5, 15.0, 0.0)
        # Day 2: big snowfall resets
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-04",
                       11.0, 33.0, 0.0, 20.0, 10.0, 1.0)
        # Day 3: hoar conditions again, re-accumulated from 0
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       11.0, 33.0, 2.5, 27.5, 15.0, 0.0)
        compute_physics_features(physics_db)
        rows = physics_db.execute("""
            SELECT date, surface_hoar_index FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL'
            ORDER BY date
        """).fetchall()
        assert rows[0][1] == 1  # Jan 3
        assert rows[1][1] == 0  # Jan 4: reset, not hoar day
        assert rows[2][1] == 1  # Jan 5: re-accumulated


class TestRainOnSnow:
    def test_ros_detected(self, physics_db):
        """Precip + warm + snowpack: rain-on-snow event."""
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       10.0, 30.0, 28.0, 35.0, 31.0, 0.5)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT rain_on_snow_hours, rain_on_snow_amount FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row is not None
        assert row[0] == 1
        assert row[1] == pytest.approx(0.5)

    def test_ros_not_detected_no_precip(self, physics_db):
        """No precipitation: not ROS."""
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       10.0, 30.0, 28.0, 35.0, 31.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT rain_on_snow_hours, rain_on_snow_amount FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row[0] == 0
        assert row[1] == pytest.approx(0.0)

    def test_ros_not_detected_too_cold(self, physics_db):
        """Max temp <= 32F: not ROS (must be strictly above freezing)."""
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       10.0, 30.0, 20.0, 30.0, 25.0, 0.5)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT rain_on_snow_hours FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row[0] == 0

    def test_ros_not_detected_no_snowpack(self, physics_db):
        """Snow depth = 0: not ROS."""
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       0.0, 0.0, 28.0, 35.0, 31.0, 0.5)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT rain_on_snow_hours FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row[0] == 0

    def test_ros_edge_temp_exactly_32(self, physics_db):
        """Temp max exactly 32F: not ROS (strictly > 32F required)."""
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       10.0, 30.0, 25.0, 32.0, 28.0, 0.5)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT rain_on_snow_hours FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row[0] == 0


class TestFreezThawCycles:
    def test_all_freeze_thaw(self, physics_db):
        """14 days all meeting freeze-thaw criteria."""
        for day in range(1, 15):
            _insert_daily(physics_db, "TEST:CO:SNTL", f"2024-01-{day:02d}",
                           10.0, 30.0, 25.0, 35.0, 30.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT freeze_thaw_cycles FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-14'
        """).fetchone()
        assert row is not None
        assert row[0] == 14

    def test_mixed_conditions(self, physics_db):
        """Mix of freeze-thaw and non-freeze-thaw days in 14-day window."""
        # 5 freeze-thaw: max=35 (>32), min=25 (<28.4)
        for day in range(1, 6):
            _insert_daily(physics_db, "TEST:CO:SNTL", f"2024-01-{day:02d}",
                           10.0, 30.0, 25.0, 35.0, 30.0, 0.0)
        # 5 NOT: max=35, min=30 (>28.4)
        for day in range(6, 11):
            _insert_daily(physics_db, "TEST:CO:SNTL", f"2024-01-{day:02d}",
                           10.0, 30.0, 30.0, 35.0, 32.0, 0.0)
        # 4 NOT: max=28 (<32)
        for day in range(11, 15):
            _insert_daily(physics_db, "TEST:CO:SNTL", f"2024-01-{day:02d}",
                           10.0, 30.0, 20.0, 28.0, 24.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT freeze_thaw_cycles FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-14'
        """).fetchone()
        assert row is not None
        assert row[0] == 5

    def test_no_snowpack_excluded(self, physics_db):
        """Freeze-thaw on bare ground does not count."""
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       0.0, 0.0, 25.0, 35.0, 30.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT freeze_thaw_cycles FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row[0] == 0


class TestSnowDepthAnomaly:
    def test_above_normal(self, physics_db):
        """Depth well above climatological median: positive anomaly."""
        _insert_years_of_data(physics_db, "TEST:CO:SNTL", 1, 5,
                              [20, 22, 24, 26, 28, 30, 32, 34, 36, 38])
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       15.0, 50.0, 10.0, 30.0, 20.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT snow_depth_anomaly FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row is not None
        assert row[0] is not None
        assert row[0] > 0

    def test_below_normal(self, physics_db):
        """Depth well below climatological median: negative anomaly."""
        _insert_years_of_data(physics_db, "TEST:CO:SNTL", 1, 5,
                              [20, 22, 24, 26, 28, 30, 32, 34, 36, 38])
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       3.0, 10.0, 10.0, 30.0, 20.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT snow_depth_anomaly FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row is not None
        assert row[0] is not None
        assert row[0] < 0

    def test_null_insufficient_years(self, physics_db):
        """Fewer than 10 years of data: anomaly is NULL."""
        _insert_years_of_data(physics_db, "TEST:CO:SNTL", 1, 5,
                              [20, 22, 24, 26, 28])
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       10.0, 30.0, 10.0, 30.0, 20.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT snow_depth_anomaly FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row is not None
        assert row[0] is None


class TestEarlySeasonFlag:
    def test_thin_snowpack_before_jan15(self, physics_db):
        """Before Jan 15 + below 25th percentile: flag = 1."""
        _insert_years_of_data(physics_db, "TEST:CO:SNTL", 1, 10,
                              [20, 22, 24, 26, 28, 30, 32, 34, 36, 38])
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-10",
                       6.0, 20.0, 10.0, 30.0, 20.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT early_season_flag FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-10'
        """).fetchone()
        assert row is not None
        assert row[0] == 1

    def test_normal_snowpack_before_jan15(self, physics_db):
        """Before Jan 15 but above 25th percentile: flag = 0."""
        _insert_years_of_data(physics_db, "TEST:CO:SNTL", 1, 10,
                              [20, 22, 24, 26, 28, 30, 32, 34, 36, 38])
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-10",
                       10.0, 30.0, 10.0, 30.0, 20.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT early_season_flag FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-10'
        """).fetchone()
        assert row is not None
        assert row[0] == 0

    def test_thin_snowpack_after_jan15(self, physics_db):
        """After Jan 15: flag = 0 regardless of snowpack."""
        _insert_years_of_data(physics_db, "TEST:CO:SNTL", 1, 20,
                              [20, 22, 24, 26, 28, 30, 32, 34, 36, 38])
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-20",
                       6.0, 20.0, 10.0, 30.0, 20.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT early_season_flag FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-20'
        """).fetchone()
        assert row is not None
        assert row[0] == 0


class TestWindSlabLoading:
    def test_three_day_precip_sum(self, physics_db):
        """Wind slab loading = 3-day rolling precip sum."""
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-03",
                       10.0, 30.0, 5.0, 25.0, 15.0, 0.5)
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-04",
                       11.0, 32.0, 0.0, 20.0, 10.0, 1.0)
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       12.0, 34.0, 5.0, 25.0, 15.0, 0.3)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT wind_slab_loading FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(1.8)

    def test_no_precip(self, physics_db):
        """No precipitation: loading = 0."""
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       10.0, 30.0, 5.0, 25.0, 15.0, 0.0)
        compute_physics_features(physics_db)
        row = physics_db.execute("""
            SELECT wind_slab_loading FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(0.0)


class TestIntegration:
    def test_all_features_computed(self, physics_db):
        """All 8 features produced for station-date with sufficient history."""
        for year in range(2010, 2020):
            _insert_daily(physics_db, "TEST:CO:SNTL", f"{year}-01-05",
                           10.0, 30.0, 5.0, 25.0, 15.0, 0.0)
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       10.0, 30.0, 5.0, 25.0, 15.0, 0.0)
        compute_physics_features(physics_db)
        cols = ", ".join(PHYSICS_FEATURES)
        row = physics_db.execute(f"""
            SELECT {cols} FROM physics_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()
        assert row is not None
        for i, feat in enumerate(PHYSICS_FEATURES):
            assert row[i] is not None, f"{feat} is NULL"

    def test_multiple_stations(self, physics_db):
        """Features computed independently per station."""
        _insert_daily(physics_db, "A:CO:SNTL", "2024-01-05",
                       10.0, 30.0, 5.0, 25.0, 15.0, 0.0)
        _insert_daily(physics_db, "B:CO:SNTL", "2024-01-05",
                       20.0, 60.0, 20.0, 30.0, 25.0, 0.0)
        compute_physics_features(physics_db)
        count = physics_db.execute(
            "SELECT COUNT(*) FROM physics_features"
        ).fetchone()[0]
        assert count == 2

    def test_upsert_on_recompute(self, physics_db):
        """Recomputing features should update, not duplicate."""
        _insert_daily(physics_db, "TEST:CO:SNTL", "2024-01-05",
                       10.0, 30.0, 5.0, 25.0, 15.0, 0.0)
        compute_physics_features(physics_db)
        count_before = physics_db.execute(
            "SELECT COUNT(*) FROM physics_features"
        ).fetchone()[0]
        compute_physics_features(physics_db)
        count_after = physics_db.execute(
            "SELECT COUNT(*) FROM physics_features"
        ).fetchone()[0]
        assert count_after == count_before

    def test_feature_count(self):
        """9 physics proxy features (8 original + consecutive gradient days)."""
        assert len(PHYSICS_FEATURES) == 9

    def test_filtered_computation(self, physics_db):
        """Can compute features for specific station and date range."""
        for day in range(1, 6):
            _insert_daily(physics_db, "TEST:CO:SNTL", f"2024-01-{day:02d}",
                           10.0, 30.0, 5.0, 25.0, 15.0, 0.0)
        compute_physics_features(
            physics_db,
            station_ids=["TEST:CO:SNTL"],
            start_date=datetime.date(2024, 1, 3),
            end_date=datetime.date(2024, 1, 5),
        )
        count = physics_db.execute(
            "SELECT COUNT(*) FROM physics_features"
        ).fetchone()[0]
        assert count == 3


class TestTableSchema:
    def test_columns_match_feature_list(self, physics_db):
        """Physics features table columns match PHYSICS_FEATURES list."""
        cols = physics_db.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'physics_features'
            AND column_name NOT IN ('station_id', 'date')
            ORDER BY column_name
        """).fetchall()
        col_names = sorted([c[0] for c in cols])
        assert col_names == sorted(PHYSICS_FEATURES)
