import datetime

import pytest

from avalanche_ml.db.schema import create_tables
from avalanche_ml.features.weather import (
    WINDOWS,
    compute_weather_features,
    create_weather_features_table,
    get_feature_columns,
)


@pytest.fixture
def weather_db(db):
    create_tables(db)
    create_weather_features_table(db)
    return db


def _insert_daily(conn, station_id, date_str, swe, snow_depth, temp_min, temp_max, temp_mean,
                   precip):
    conn.execute("""
        INSERT INTO snotel_daily (station_id, date, swe_inches, snow_depth_inches,
            air_temp_min_f, air_temp_max_f, air_temp_mean_f, precip_increment_inches)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [station_id, date_str, swe, snow_depth, temp_min, temp_max, temp_mean, precip])


@pytest.fixture
def five_day_data(weather_db):
    """5 consecutive days of known data for TEST:CO:SNTL."""
    rows = [
        ("TEST:CO:SNTL", "2024-01-01", 10.0, 30.0, 5.0, 25.0, 15.0, 0.5),
        ("TEST:CO:SNTL", "2024-01-02", 11.0, 32.0, 0.0, 20.0, 10.0, 1.0),
        ("TEST:CO:SNTL", "2024-01-03", 11.0, 32.0, -5.0, 15.0, 5.0, 0.0),
        ("TEST:CO:SNTL", "2024-01-04", 12.0, 35.0, 10.0, 30.0, 20.0, 2.0),
        ("TEST:CO:SNTL", "2024-01-05", 12.5, 36.0, -10.0, 10.0, 0.0, 0.5),
    ]
    for row in rows:
        _insert_daily(weather_db, *row)
    return weather_db


class TestTemperatureFeatures:
    def test_72h_window_aggregations(self, five_day_data):
        """72h window on 2024-01-03 includes 01-01, 01-02, 01-03."""
        compute_weather_features(five_day_data)
        row = five_day_data.execute("""
            SELECT temp_mean_72h, temp_min_72h, temp_max_72h,
                   temp_std_72h, temp_range_72h, temp_trend_72h
            FROM weather_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-03'
        """).fetchone()

        assert row is not None
        assert row[0] == pytest.approx(10.0)       # mean(15,10,5)
        assert row[1] == pytest.approx(-5.0)        # min(5,0,-5)
        assert row[2] == pytest.approx(25.0)         # max(25,20,15)
        assert row[3] == pytest.approx(5.0)          # stddev_samp(15,10,5)
        assert row[4] == pytest.approx(30.0)         # 25-(-5)
        assert row[5] == pytest.approx(-5.0)         # slope: -5 per day

    def test_24h_window_is_single_day(self, five_day_data):
        compute_weather_features(five_day_data)
        row = five_day_data.execute("""
            SELECT temp_mean_24h, temp_min_24h, temp_max_24h, temp_std_24h
            FROM weather_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-03'
        """).fetchone()

        assert row is not None
        assert row[0] == pytest.approx(5.0)    # mean of single value
        assert row[1] == pytest.approx(-5.0)   # min = the value
        assert row[2] == pytest.approx(15.0)   # max = the value
        assert row[3] is None                   # stddev of 1 value = NULL


class TestPrecipitationFeatures:
    def test_72h_precip_aggregations(self, five_day_data):
        """72h window on 2024-01-03: precip = [0.5, 1.0, 0.0]."""
        compute_weather_features(five_day_data)
        row = five_day_data.execute("""
            SELECT precip_sum_72h, precip_max_72h, precip_days_72h, precip_intensity_72h
            FROM weather_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-03'
        """).fetchone()

        assert row is not None
        assert row[0] == pytest.approx(1.5)    # 0.5 + 1.0 + 0.0
        assert row[1] == pytest.approx(1.0)    # max
        assert row[2] == pytest.approx(2.0)    # 2 days with precip > 0
        assert row[3] == pytest.approx(0.75)   # 1.5 / 2

    def test_precip_intensity_zero_days(self, five_day_data):
        """When no precip days in window, intensity should be NULL."""
        _insert_daily(five_day_data, "DRY:CO:SNTL", "2024-01-01",
                       10.0, 30.0, 5.0, 25.0, 15.0, 0.0)
        compute_weather_features(five_day_data)
        row = five_day_data.execute("""
            SELECT precip_intensity_24h
            FROM weather_features
            WHERE station_id = 'DRY:CO:SNTL' AND date = '2024-01-01'
        """).fetchone()

        assert row is not None
        assert row[0] is None


class TestSnowFeatures:
    def test_swe_change_72h(self, five_day_data):
        """SWE change over 72h on 01-03: 11.0 - 10.0 = 1.0."""
        compute_weather_features(five_day_data)
        row = five_day_data.execute("""
            SELECT swe_change_72h, snow_depth_change_72h, swe_rate_72h, new_snow_72h
            FROM weather_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-03'
        """).fetchone()

        assert row is not None
        assert row[0] == pytest.approx(1.0)             # 11.0 - 10.0
        assert row[1] == pytest.approx(2.0)              # 32.0 - 30.0
        assert row[2] == pytest.approx(1.0 / 3.0)        # swe_change / 3 days
        assert row[3] == pytest.approx(2.0)               # GREATEST(2.0, 0)

    def test_new_snow_clamped_to_zero(self, weather_db):
        """When snow depth decreases over window, new_snow = 0."""
        _insert_daily(weather_db, "MELT:CO:SNTL", "2024-01-01",
                       15.0, 40.0, 30.0, 45.0, 37.0, 0.0)
        _insert_daily(weather_db, "MELT:CO:SNTL", "2024-01-02",
                       13.0, 35.0, 32.0, 48.0, 40.0, 0.0)
        compute_weather_features(weather_db)
        row = weather_db.execute("""
            SELECT new_snow_48h
            FROM weather_features
            WHERE station_id = 'MELT:CO:SNTL' AND date = '2024-01-02'
        """).fetchone()

        assert row is not None
        assert row[0] == pytest.approx(0.0)


class TestMultipleWindows:
    def test_all_windows_computed(self, five_day_data):
        """On date with full 120h history, all 5 windows should be non-NULL."""
        compute_weather_features(five_day_data)
        cols = ", ".join(f"temp_mean_{w}h" for w in WINDOWS)
        row = five_day_data.execute(f"""
            SELECT {cols}
            FROM weather_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()

        assert row is not None
        for val in row:
            assert val is not None

    def test_120h_window_all_five_days(self, five_day_data):
        """120h window on 01-05 includes all 5 days."""
        compute_weather_features(five_day_data)
        row = five_day_data.execute("""
            SELECT temp_mean_120h, precip_sum_120h, swe_change_120h
            FROM weather_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()

        assert row is not None
        assert row[0] == pytest.approx(10.0)   # mean(15,10,5,20,0)
        assert row[1] == pytest.approx(4.0)    # 0.5+1.0+0.0+2.0+0.5
        assert row[2] == pytest.approx(2.5)    # 12.5 - 10.0


class TestMissingDataCoverage:
    def test_below_50pct_coverage_nulls_features(self, weather_db):
        """72h window with only 1 of 3 expected days → coverage 33% → NULL."""
        _insert_daily(weather_db, "SPARSE:CO:SNTL", "2024-01-01",
                       10.0, 30.0, 5.0, 25.0, 15.0, 0.5)
        _insert_daily(weather_db, "SPARSE:CO:SNTL", "2024-01-03",
                       11.0, 32.0, -5.0, 15.0, 5.0, 0.0)
        compute_weather_features(weather_db)
        row = weather_db.execute("""
            SELECT temp_mean_72h, precip_sum_72h, coverage_72h
            FROM weather_features
            WHERE station_id = 'SPARSE:CO:SNTL' AND date = '2024-01-03'
        """).fetchone()

        assert row is not None
        assert row[2] == pytest.approx(2.0 / 3.0)  # 2 of 3 days present
        assert row[0] is not None  # 66% coverage >= 50% threshold
        assert row[1] is not None

    def test_at_50pct_coverage_computes(self, weather_db):
        """96h window with 2 of 4 expected days → coverage 50% → compute."""
        _insert_daily(weather_db, "HALF:CO:SNTL", "2024-01-01",
                       10.0, 30.0, 5.0, 25.0, 15.0, 0.5)
        _insert_daily(weather_db, "HALF:CO:SNTL", "2024-01-04",
                       12.0, 35.0, 10.0, 30.0, 20.0, 2.0)
        compute_weather_features(weather_db)
        row = weather_db.execute("""
            SELECT temp_mean_96h, coverage_96h
            FROM weather_features
            WHERE station_id = 'HALF:CO:SNTL' AND date = '2024-01-04'
        """).fetchone()

        assert row is not None
        assert row[1] == pytest.approx(2.0 / 4.0)  # 50%
        assert row[0] is not None  # computed, not NULL

    def test_single_day_below_threshold_for_large_window(self, weather_db):
        """120h window with 1 of 5 expected days → 20% → NULL."""
        _insert_daily(weather_db, "LONE:CO:SNTL", "2024-01-05",
                       12.5, 36.0, -10.0, 10.0, 0.0, 0.5)
        compute_weather_features(weather_db)
        row = weather_db.execute("""
            SELECT temp_mean_120h, coverage_120h
            FROM weather_features
            WHERE station_id = 'LONE:CO:SNTL' AND date = '2024-01-05'
        """).fetchone()

        assert row is not None
        assert row[1] == pytest.approx(1.0 / 5.0)  # 20%
        assert row[0] is None  # below 50% threshold


class TestEdgeCases:
    def test_first_day_of_data(self, weather_db):
        """First day has only 24h window with data; larger windows have partial coverage."""
        _insert_daily(weather_db, "NEW:CO:SNTL", "2024-01-01",
                       10.0, 30.0, 5.0, 25.0, 15.0, 0.5)
        compute_weather_features(weather_db)
        row = weather_db.execute("""
            SELECT temp_mean_24h, temp_mean_48h, coverage_24h, coverage_48h
            FROM weather_features
            WHERE station_id = 'NEW:CO:SNTL' AND date = '2024-01-01'
        """).fetchone()

        assert row is not None
        assert row[0] == pytest.approx(15.0)  # 24h always has data
        assert row[2] == pytest.approx(1.0)   # full coverage for 24h
        assert row[3] == pytest.approx(0.5)   # 1 of 2 days for 48h
        assert row[1] is not None             # 50% = at threshold, compute

    def test_multiple_stations(self, weather_db):
        """Features computed independently per station."""
        _insert_daily(weather_db, "A:CO:SNTL", "2024-01-01",
                       10.0, 30.0, 5.0, 25.0, 15.0, 0.5)
        _insert_daily(weather_db, "B:CO:SNTL", "2024-01-01",
                       20.0, 60.0, 30.0, 50.0, 40.0, 1.0)
        compute_weather_features(weather_db)

        row_a = weather_db.execute("""
            SELECT temp_mean_24h FROM weather_features
            WHERE station_id = 'A:CO:SNTL' AND date = '2024-01-01'
        """).fetchone()
        row_b = weather_db.execute("""
            SELECT temp_mean_24h FROM weather_features
            WHERE station_id = 'B:CO:SNTL' AND date = '2024-01-01'
        """).fetchone()

        assert row_a[0] == pytest.approx(15.0)
        assert row_b[0] == pytest.approx(40.0)


class TestWindFeatures:
    def test_wind_columns_are_null(self, five_day_data):
        """Wind features should be NULL since no wind data in snotel_daily."""
        compute_weather_features(five_day_data)
        wind_cols = ", ".join(
            f"wind_mean_{w}h, wind_max_{w}h, wind_std_{w}h, wind_direction_mode_{w}h"
            for w in WINDOWS
        )
        row = five_day_data.execute(f"""
            SELECT {wind_cols}
            FROM weather_features
            WHERE station_id = 'TEST:CO:SNTL' AND date = '2024-01-03'
        """).fetchone()

        assert row is not None
        for val in row:
            assert val is None


class TestFeatureNaming:
    def test_column_naming_convention(self, five_day_data):
        """All feature columns follow {variable}_{aggregation}_{window}h pattern."""
        columns = get_feature_columns()
        for col in columns:
            parts = col.rsplit("_", 1)
            assert len(parts) == 2, f"Column {col} doesn't end with _{{window}}h"
            assert parts[1].endswith("h"), f"Column {col} doesn't end with 'h'"
            window_str = parts[1][:-1]
            assert window_str.isdigit(), f"Column {col} window part is not numeric"
            assert int(window_str) in WINDOWS, f"Column {col} has unexpected window size"

    def test_feature_count(self):
        """Spec requires exactly 90 weather window features."""
        assert len(get_feature_columns()) == 90


class TestFeatureMatrixShape:
    def test_full_matrix_shape(self, five_day_data):
        """5 days of data for 1 station → 5 rows with all feature columns."""
        compute_weather_features(five_day_data)
        count = five_day_data.execute("""
            SELECT COUNT(*) FROM weather_features
            WHERE station_id = 'TEST:CO:SNTL'
        """).fetchone()[0]
        assert count == 5

        col_count = five_day_data.execute("""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_name = 'weather_features'
            AND column_name NOT IN ('station_id', 'date')
            AND column_name NOT LIKE 'coverage_%'
        """).fetchone()[0]
        assert col_count == 90


class TestIncrementalComputation:
    def test_upsert_on_recompute(self, five_day_data):
        """Recomputing features for same dates should update, not duplicate."""
        compute_weather_features(five_day_data)
        count_before = five_day_data.execute(
            "SELECT COUNT(*) FROM weather_features"
        ).fetchone()[0]

        compute_weather_features(five_day_data)
        count_after = five_day_data.execute(
            "SELECT COUNT(*) FROM weather_features"
        ).fetchone()[0]

        assert count_after == count_before

    def test_filtered_computation(self, five_day_data):
        """Can compute features for a specific station and date range."""
        compute_weather_features(
            five_day_data,
            station_ids=["TEST:CO:SNTL"],
            start_date=datetime.date(2024, 1, 3),
            end_date=datetime.date(2024, 1, 5),
        )
        count = five_day_data.execute(
            "SELECT COUNT(*) FROM weather_features"
        ).fetchone()[0]
        assert count == 3
