import os
import tempfile

import duckdb
import pytest

from avalanche_ml.db.schema import create_tables
from avalanche_ml.features.alignment import (
    assemble_training_matrix,
    class_distribution_report,
    create_training_matrix_table,
    export_to_parquet,
    temporal_split,
)
from avalanche_ml.features.physics import PHYSICS_FEATURES, create_physics_features_table
from avalanche_ml.features.weather import create_weather_features_table, get_feature_columns


@pytest.fixture
def alignment_db(db):
    create_tables(db)
    create_weather_features_table(db)
    create_physics_features_table(db)
    create_training_matrix_table(db)
    return db


def _insert_station(conn, station_id, lat=39.0, lon=-106.0, elev=10000.0):
    conn.execute("""
        INSERT INTO stations (station_id, name, latitude, longitude, elevation)
        VALUES (?, ?, ?, ?, ?)
    """, [station_id, f"Station {station_id}", lat, lon, elev])


def _insert_zone_mapping(conn, zone_id, station_id):
    conn.execute("""
        INSERT INTO zone_station_mapping (zone_id, station_id)
        VALUES (?, ?)
    """, [zone_id, station_id])


def _insert_danger(conn, zone_id, date_str, band, level, source="caic"):
    conn.execute("""
        INSERT INTO danger_ratings (zone_id, date, elevation_band, danger_level, source)
        VALUES (?, ?, ?, ?, ?)
    """, [zone_id, date_str, band, level, source])


def _insert_problem(conn, zone_id, date_str, problem_type, source="caic"):
    conn.execute("""
        INSERT INTO problem_types
            (zone_id, date, problem_type, likelihood, ordering, source)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [zone_id, date_str, problem_type, "Likely", 1, source])


def _insert_weather_features(conn, station_id, date_str):
    cols = get_feature_columns()
    all_cols = ["station_id", "date"] + cols
    placeholders = ", ".join("?" for _ in all_cols)
    values = [station_id, date_str] + [1.0] * len(cols)
    conn.execute(f"""
        INSERT INTO weather_features ({', '.join(all_cols)})
        VALUES ({placeholders})
    """, values)


def _insert_physics_features(conn, station_id, date_str):
    cols = ["station_id", "date"] + PHYSICS_FEATURES
    placeholders = ", ".join("?" for _ in cols)
    values = [station_id, date_str] + [1.0] * len(PHYSICS_FEATURES)
    conn.execute(f"""
        INSERT INTO physics_features ({', '.join(cols)})
        VALUES ({placeholders})
    """, values)


def _seed_complete_row(conn, station_id, zone_id, date_str, band, danger_level,
                       problem_types=None):
    _insert_weather_features(conn, station_id, date_str)
    _insert_physics_features(conn, station_id, date_str)
    _insert_danger(conn, zone_id, date_str, band, danger_level)
    for pt in (problem_types or []):
        _insert_problem(conn, zone_id, date_str, pt)


@pytest.fixture
def seeded_db(alignment_db):
    _insert_station(alignment_db, "S1", lat=39.5, lon=-106.0, elev=11000.0)
    _insert_zone_mapping(alignment_db, "Z1", "S1")
    _seed_complete_row(alignment_db, "S1", "Z1", "2022-01-15",
                       "above_treeline", 3, ["Persistent Slab", "Storm Slab"])
    _seed_complete_row(alignment_db, "S1", "Z1", "2022-01-16",
                       "above_treeline", 2, ["Storm Slab"])
    return alignment_db


class TestStationZoneJoin:
    def test_station_gets_correct_zone_label(self, seeded_db):
        count = assemble_training_matrix(seeded_db)
        assert count > 0
        row = seeded_db.execute("""
            SELECT danger_level FROM training_matrix
            WHERE station_id = 'S1' AND date = '2022-01-15'
              AND elevation_band = 'above_treeline'
        """).fetchone()
        assert row is not None
        assert row[0] == 3

    def test_station_in_multiple_zones(self, alignment_db):
        _insert_station(alignment_db, "S1")
        _insert_zone_mapping(alignment_db, "Z1", "S1")
        _insert_zone_mapping(alignment_db, "Z2", "S1")
        _insert_weather_features(alignment_db, "S1", "2022-01-10")
        _insert_physics_features(alignment_db, "S1", "2022-01-10")
        _insert_danger(alignment_db, "Z1", "2022-01-10", "above_treeline", 3)
        _insert_danger(alignment_db, "Z2", "2022-01-10", "above_treeline", 4)
        assemble_training_matrix(alignment_db)
        rows = alignment_db.execute("""
            SELECT station_id, danger_level FROM training_matrix
            WHERE date = '2022-01-10' AND elevation_band = 'above_treeline'
            ORDER BY danger_level
        """).fetchall()
        assert len(rows) == 2
        assert rows[0][1] == 3
        assert rows[1][1] == 4

    def test_station_in_no_zone_excluded(self, alignment_db):
        _insert_station(alignment_db, "ORPHAN")
        _insert_weather_features(alignment_db, "ORPHAN", "2022-01-10")
        _insert_physics_features(alignment_db, "ORPHAN", "2022-01-10")
        count = assemble_training_matrix(alignment_db)
        assert count == 0


class TestFeatureMatrixColumns:
    def test_has_weather_features(self, seeded_db):
        assemble_training_matrix(seeded_db)
        weather_cols = get_feature_columns()
        sample = weather_cols[:3]
        select = ", ".join(sample)
        row = seeded_db.execute(f"""
            SELECT {select} FROM training_matrix LIMIT 1
        """).fetchone()
        assert row is not None
        for val in row:
            assert val is not None

    def test_has_physics_features(self, seeded_db):
        assemble_training_matrix(seeded_db)
        select = ", ".join(PHYSICS_FEATURES)
        row = seeded_db.execute(f"""
            SELECT {select} FROM training_matrix LIMIT 1
        """).fetchone()
        assert row is not None
        for val in row:
            assert val is not None

    def test_has_label_columns(self, seeded_db):
        assemble_training_matrix(seeded_db)
        row = seeded_db.execute("""
            SELECT danger_level, persistent_slab, storm_slab, loose_wet
            FROM training_matrix
            WHERE station_id = 'S1' AND date = '2022-01-15'
              AND elevation_band = 'above_treeline'
        """).fetchone()
        assert row is not None
        assert row[0] == 3
        assert row[1] == 1  # persistent_slab present
        assert row[2] == 1  # storm_slab present
        assert row[3] == 0  # loose_wet not present

    def test_has_metadata_columns(self, seeded_db):
        assemble_training_matrix(seeded_db)
        row = seeded_db.execute("""
            SELECT elevation_band, elevation, latitude, longitude,
                   month, day_of_year, days_since_season_start
            FROM training_matrix
            WHERE station_id = 'S1' AND date = '2022-01-15'
              AND elevation_band = 'above_treeline'
        """).fetchone()
        assert row is not None
        assert row[0] == "above_treeline"
        assert row[1] == 11000.0
        assert row[2] == 39.5
        assert row[3] == -106.0
        assert row[4] == 1   # January
        assert row[5] == 15  # day 15
        assert row[6] > 0    # days since Oct 1


class TestRowGranularity:
    def test_rows_per_station_date_band(self, seeded_db):
        assemble_training_matrix(seeded_db)
        rows = seeded_db.execute("""
            SELECT COUNT(*) FROM training_matrix
            WHERE station_id = 'S1' AND date = '2022-01-15'
        """).fetchone()
        assert rows[0] == 1  # one band seeded

    def test_multiple_bands_produce_multiple_rows(self, alignment_db):
        _insert_station(alignment_db, "S1")
        _insert_zone_mapping(alignment_db, "Z1", "S1")
        _insert_weather_features(alignment_db, "S1", "2022-02-01")
        _insert_physics_features(alignment_db, "S1", "2022-02-01")
        for band, level in [("above_treeline", 4), ("near_treeline", 3),
                            ("below_treeline", 2)]:
            _insert_danger(alignment_db, "Z1", "2022-02-01", band, level)
        assemble_training_matrix(alignment_db)
        count = alignment_db.execute("""
            SELECT COUNT(*) FROM training_matrix
            WHERE station_id = 'S1' AND date = '2022-02-01'
        """).fetchone()[0]
        assert count == 3


class TestMissingData:
    def test_missing_label_excluded(self, alignment_db):
        _insert_station(alignment_db, "S1")
        _insert_zone_mapping(alignment_db, "Z1", "S1")
        _insert_weather_features(alignment_db, "S1", "2022-03-01")
        _insert_physics_features(alignment_db, "S1", "2022-03-01")
        count = assemble_training_matrix(alignment_db)
        assert count == 0

    def test_missing_weather_features_flagged(self, alignment_db):
        _insert_station(alignment_db, "S1")
        _insert_zone_mapping(alignment_db, "Z1", "S1")
        _insert_danger(alignment_db, "Z1", "2022-03-01", "above_treeline", 3)
        _insert_physics_features(alignment_db, "S1", "2022-03-01")
        count = assemble_training_matrix(alignment_db)
        assert count == 0

    def test_missing_physics_features_flagged(self, alignment_db):
        _insert_station(alignment_db, "S1")
        _insert_zone_mapping(alignment_db, "Z1", "S1")
        _insert_danger(alignment_db, "Z1", "2022-03-01", "above_treeline", 3)
        _insert_weather_features(alignment_db, "S1", "2022-03-01")
        count = assemble_training_matrix(alignment_db)
        assert count == 0


class TestTemporalSplit:
    def test_split_boundaries(self, alignment_db):
        _insert_station(alignment_db, "S1")
        _insert_zone_mapping(alignment_db, "Z1", "S1")
        dates_and_levels = [
            ("2020-01-15", 2),
            ("2024-01-15", 3),
            ("2024-11-01", 4),
            ("2025-03-01", 3),
        ]
        for date_str, level in dates_and_levels:
            _seed_complete_row(alignment_db, "S1", "Z1", date_str,
                               "above_treeline", level)
        assemble_training_matrix(alignment_db)
        train, val, test = temporal_split(alignment_db)
        assert len(train) == 2   # 2020, 2024-01 dates <= 2024-06-30
        assert len(val) == 1     # 2024-11-01 in val window
        assert len(test) == 1    # 2025-03-01 in test window

    def test_split_returns_dataframes_with_columns(self, seeded_db):
        assemble_training_matrix(seeded_db)
        train, _val, _test = temporal_split(seeded_db)
        assert "danger_level" in train.columns
        assert "station_id" in train.columns


class TestClassDistribution:
    def test_distribution_report(self, seeded_db):
        assemble_training_matrix(seeded_db)
        report = class_distribution_report(seeded_db)
        assert "danger_levels" in report
        assert "problem_types" in report
        assert "coverage" in report

    def test_danger_level_counts(self, seeded_db):
        assemble_training_matrix(seeded_db)
        report = class_distribution_report(seeded_db)
        dl = report["danger_levels"]
        total = sum(d["count"] for d in dl)
        assert total == 2  # 2 rows seeded

    def test_coverage_stats(self, seeded_db):
        assemble_training_matrix(seeded_db)
        report = class_distribution_report(seeded_db)
        cov = report["coverage"]
        assert cov["total_rows"] == 2
        assert cov["complete_rows"] == 2
        assert cov["coverage_pct"] == 100.0


class TestParquetExport:
    def test_export_roundtrip(self, seeded_db):
        assemble_training_matrix(seeded_db)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "matrix.parquet")
            export_to_parquet(seeded_db, path)
            assert os.path.exists(path)
            reimported = duckdb.connect(":memory:")
            df = reimported.execute(
                f"SELECT * FROM read_parquet('{path}')"
            ).fetchdf()
            assert len(df) == 2
            assert "danger_level" in df.columns
            reimported.close()


class TestFullPipeline:
    def test_end_to_end_synthetic(self, alignment_db):
        from avalanche_ml.features.physics import compute_physics_features
        from avalanche_ml.features.weather import compute_weather_features

        _insert_station(alignment_db, "S1", lat=39.5, lon=-106.0, elev=11000.0)
        _insert_zone_mapping(alignment_db, "Z1", "S1")

        for i in range(1, 11):
            d = f"2022-01-{i:02d}"
            alignment_db.execute("""
                INSERT INTO snotel_daily
                    (station_id, date, swe_inches, snow_depth_inches,
                     air_temp_min_f, air_temp_max_f, air_temp_mean_f,
                     precip_increment_inches)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, ["S1", d, 10.0 + i * 0.5, 30.0 + i, 0.0, 25.0, 12.0, 0.3])

        compute_weather_features(alignment_db, station_ids=["S1"])
        compute_physics_features(alignment_db, station_ids=["S1"])

        for i in range(1, 11):
            d = f"2022-01-{i:02d}"
            _insert_danger(alignment_db, "Z1", d, "above_treeline", (i % 5) + 1)
            if i % 3 == 0:
                _insert_problem(alignment_db, "Z1", d, "Persistent Slab")

        count = assemble_training_matrix(alignment_db)
        assert count == 10

        report = class_distribution_report(alignment_db)
        assert report["coverage"]["total_rows"] == 10

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            export_to_parquet(alignment_db, path)
            reimported = duckdb.connect(":memory:")
            df = reimported.execute(
                f"SELECT * FROM read_parquet('{path}')"
            ).fetchdf()
            assert len(df) == 10
            reimported.close()
