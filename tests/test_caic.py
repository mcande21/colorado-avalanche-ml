import datetime

import httpx
import pytest
import respx

from avalanche_ml.db.schema import create_tables
from avalanche_ml.ingestion.caic import CaicClient

CAIC_API_BASE = "https://avalanche.state.co.us/api-proxy/avid"
OAP_BASE = "https://raw.githubusercontent.com/openavproject/data"


@pytest.fixture
def caic_db(db):
    create_tables(db)
    return db


@pytest.fixture
def client(caic_db):
    return CaicClient(db=caic_db)


@pytest.fixture
def danger_rating_response():
    return [
        {
            "zone_id": "BTAC",
            "date": "2024-01-15",
            "above_treeline": 3,
            "near_treeline": 3,
            "below_treeline": 2,
        },
        {
            "zone_id": "BTAC",
            "date": "2024-01-16",
            "above_treeline": 4,
            "near_treeline": 3,
            "below_treeline": 2,
        },
    ]


@pytest.fixture
def problem_type_response():
    return [
        {
            "zone_id": "BTAC",
            "date": "2024-01-15",
            "problems": [
                {
                    "type": "Persistent Slab",
                    "likelihood": "Likely",
                    "size_min": 2.0,
                    "size_max": 3.0,
                    "aspects": "N,NE,E",
                    "elevation_bands": "above_treeline,near_treeline",
                },
                {
                    "type": "Storm Slab",
                    "likelihood": "Possible",
                    "size_min": 1.5,
                    "size_max": 2.0,
                    "aspects": "N,NE,E,SE,S",
                    "elevation_bands": "above_treeline,near_treeline,below_treeline",
                },
            ],
        },
    ]


@pytest.fixture
def oap_csv_content():
    return (
        "date,zone_id,danger_above,danger_near,danger_below\n"
        "2018-01-10,BTAC,3,2,1\n"
        "2018-01-11,BTAC,4,3,2\n"
        "2018-01-10,VS,2,2,1\n"
    )


@pytest.fixture
def oap_problems_csv_content():
    return (
        "date,zone_id,problem_type,likelihood\n"
        "2018-01-10,BTAC,Persistent Slab,Likely\n"
        "2018-01-10,BTAC,Wind Slab,Possible\n"
        "2018-01-11,BTAC,Storm Slab,Likely\n"
    )


@pytest.fixture
def zone_features_response():
    return {
        "features": [
            {
                "attributes": {"zone_id": "BTAC", "name": "Vail & Summit County"},
                "geometry": {
                    "rings": [[
                        [-106.5, 39.3], [-106.0, 39.3],
                        [-106.0, 39.8], [-106.5, 39.8], [-106.5, 39.3],
                    ]],
                },
            },
        ],
    }


# --- Schema tests ---


class TestCaicSchema:
    def test_danger_ratings_table_exists(self, caic_db):
        tables = [r[0] for r in caic_db.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()]
        assert "danger_ratings" in tables

    def test_problem_types_table_exists(self, caic_db):
        tables = [r[0] for r in caic_db.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()]
        assert "problem_types" in tables

    def test_zone_station_mapping_table_exists(self, caic_db):
        tables = [r[0] for r in caic_db.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()]
        assert "zone_station_mapping" in tables


# --- Danger rating ingestion ---


class TestDangerRatingIngestion:
    @respx.mock
    async def test_stores_danger_ratings(self, client, caic_db, danger_rating_response):
        respx.get(f"{CAIC_API_BASE}/products/all").mock(
            return_value=httpx.Response(200, json=danger_rating_response)
        )
        count = await client.ingest_danger_ratings(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 16),
        )
        assert count == 6
        rows = caic_db.execute("SELECT * FROM danger_ratings").fetchall()
        assert len(rows) == 6

    @respx.mock
    async def test_parses_elevation_bands(self, client, caic_db, danger_rating_response):
        respx.get(f"{CAIC_API_BASE}/products/all").mock(
            return_value=httpx.Response(200, json=danger_rating_response)
        )
        await client.ingest_danger_ratings(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 16),
        )
        bands = sorted([r[0] for r in caic_db.execute(
            "SELECT DISTINCT elevation_band FROM danger_ratings"
        ).fetchall()])
        assert bands == ["above_treeline", "below_treeline", "near_treeline"]

    @respx.mock
    async def test_danger_levels_1_to_5(self, client, caic_db, danger_rating_response):
        respx.get(f"{CAIC_API_BASE}/products/all").mock(
            return_value=httpx.Response(200, json=danger_rating_response)
        )
        await client.ingest_danger_ratings(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 16),
        )
        levels = sorted({r[0] for r in caic_db.execute(
            "SELECT danger_level FROM danger_ratings"
        ).fetchall()})
        for lvl in levels:
            assert 1 <= lvl <= 5

    @respx.mock
    async def test_source_tagged_caic(self, client, caic_db, danger_rating_response):
        respx.get(f"{CAIC_API_BASE}/products/all").mock(
            return_value=httpx.Response(200, json=danger_rating_response)
        )
        await client.ingest_danger_ratings(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 16),
        )
        sources = {r[0] for r in caic_db.execute(
            "SELECT DISTINCT source FROM danger_ratings"
        ).fetchall()}
        assert sources == {"caic"}

    @respx.mock
    async def test_null_danger_for_missing_date(self, client, caic_db):
        respx.get(f"{CAIC_API_BASE}/products/all").mock(
            return_value=httpx.Response(200, json=[
                {
                    "zone_id": "BTAC",
                    "date": "2024-01-15",
                    "above_treeline": None,
                    "near_treeline": None,
                    "below_treeline": None,
                },
            ])
        )
        count = await client.ingest_danger_ratings(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 15),
        )
        assert count == 3
        rows = caic_db.execute(
            "SELECT danger_level FROM danger_ratings WHERE zone_id = 'BTAC'"
        ).fetchall()
        for row in rows:
            assert row[0] is None


# --- Problem type ingestion ---


class TestProblemTypeIngestion:
    @respx.mock
    async def test_stores_problem_types(self, client, caic_db, problem_type_response):
        respx.get(f"{CAIC_API_BASE}/products/all").mock(
            return_value=httpx.Response(200, json=problem_type_response)
        )
        count = await client.ingest_problem_types(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 15),
        )
        assert count == 2
        rows = caic_db.execute("SELECT * FROM problem_types").fetchall()
        assert len(rows) == 2

    @respx.mock
    async def test_preserves_problem_ordering(self, client, caic_db, problem_type_response):
        respx.get(f"{CAIC_API_BASE}/products/all").mock(
            return_value=httpx.Response(200, json=problem_type_response)
        )
        await client.ingest_problem_types(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 15),
        )
        rows = caic_db.execute(
            "SELECT problem_type, ordering FROM problem_types ORDER BY ordering"
        ).fetchall()
        assert rows[0][0] == "Persistent Slab"
        assert rows[0][1] == 0
        assert rows[1][0] == "Storm Slab"
        assert rows[1][1] == 1

    @respx.mock
    async def test_normalizes_problem_type_names(self, client, caic_db):
        response = [{
            "zone_id": "BTAC",
            "date": "2024-01-15",
            "problems": [
                {
                    "type": "persistent slab",
                    "likelihood": "Likely",
                    "size_min": 2.0,
                    "size_max": 3.0,
                    "aspects": "N",
                    "elevation_bands": "above_treeline",
                },
            ],
        }]
        respx.get(f"{CAIC_API_BASE}/products/all").mock(
            return_value=httpx.Response(200, json=response)
        )
        await client.ingest_problem_types(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 15),
        )
        row = caic_db.execute(
            "SELECT problem_type FROM problem_types"
        ).fetchone()
        assert row[0] == "Persistent Slab"

    @respx.mock
    async def test_stores_likelihood_and_size(self, client, caic_db, problem_type_response):
        respx.get(f"{CAIC_API_BASE}/products/all").mock(
            return_value=httpx.Response(200, json=problem_type_response)
        )
        await client.ingest_problem_types(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 15),
        )
        row = caic_db.execute(
            "SELECT likelihood, size_min, size_max FROM problem_types "
            "WHERE problem_type = 'Persistent Slab'"
        ).fetchone()
        assert row[0] == "Likely"
        assert abs(row[1] - 2.0) < 0.01
        assert abs(row[2] - 3.0) < 0.01


# --- OAP fallback ---


class TestOapFallback:
    @respx.mock
    async def test_oap_danger_ingestion(self, client, caic_db, oap_csv_content):
        respx.get(f"{OAP_BASE}/main/labels/danger_ratings.csv").mock(
            return_value=httpx.Response(200, text=oap_csv_content)
        )
        count = await client.ingest_oap_danger_ratings()
        assert count == 9
        rows = caic_db.execute("SELECT * FROM danger_ratings").fetchall()
        assert len(rows) == 9

    @respx.mock
    async def test_oap_tagged_source(self, client, caic_db, oap_csv_content):
        respx.get(f"{OAP_BASE}/main/labels/danger_ratings.csv").mock(
            return_value=httpx.Response(200, text=oap_csv_content)
        )
        await client.ingest_oap_danger_ratings()
        sources = {r[0] for r in caic_db.execute(
            "SELECT DISTINCT source FROM danger_ratings"
        ).fetchall()}
        assert sources == {"oap"}

    @respx.mock
    async def test_oap_problem_ingestion(self, client, caic_db, oap_problems_csv_content):
        respx.get(f"{OAP_BASE}/main/labels/problem_types.csv").mock(
            return_value=httpx.Response(200, text=oap_problems_csv_content)
        )
        count = await client.ingest_oap_problem_types()
        assert count == 3
        rows = caic_db.execute("SELECT * FROM problem_types").fetchall()
        assert len(rows) == 3

    @respx.mock
    async def test_caic_failure_falls_back_to_oap(
        self, client, caic_db, oap_csv_content
    ):
        respx.get(f"{CAIC_API_BASE}/products/all").mock(
            return_value=httpx.Response(503)
        )
        respx.get(f"{OAP_BASE}/main/labels/danger_ratings.csv").mock(
            return_value=httpx.Response(200, text=oap_csv_content)
        )
        count = await client.ingest_danger_ratings_with_fallback(
            start_date=datetime.date(2018, 1, 10),
            end_date=datetime.date(2018, 1, 11),
        )
        assert count > 0
        sources = {r[0] for r in caic_db.execute(
            "SELECT DISTINCT source FROM danger_ratings"
        ).fetchall()}
        assert sources == {"oap"}


# --- Zone-station mapping ---


class TestZoneStationMapping:
    @respx.mock
    async def test_maps_station_to_zone(self, client, caic_db, zone_features_response):
        caic_db.execute(
            "INSERT INTO stations VALUES "
            "('335:CO:SNTL', 'Berthoud Summit', 39.5, -106.25, 11300, "
            "'10190005', 'CO', 'Summit', TRUE, CURRENT_TIMESTAMP)"
        )
        respx.get(url__regex=r".*FeatureServer.*").mock(
            return_value=httpx.Response(200, json=zone_features_response)
        )
        count = await client.update_zone_station_mapping()
        assert count == 1
        row = caic_db.execute(
            "SELECT zone_id, station_id FROM zone_station_mapping"
        ).fetchone()
        assert row[0] == "BTAC"
        assert row[1] == "335:CO:SNTL"

    @respx.mock
    async def test_station_in_multiple_zones(self, client, caic_db):
        caic_db.execute(
            "INSERT INTO stations VALUES "
            "('335:CO:SNTL', 'Berthoud Summit', 39.5, -106.25, 11300, "
            "'10190005', 'CO', 'Summit', TRUE, CURRENT_TIMESTAMP)"
        )
        multi_zone = {
            "features": [
                {
                    "attributes": {"zone_id": "BTAC", "name": "Vail & Summit County"},
                    "geometry": {
                        "rings": [[
                            [-106.5, 39.3], [-106.0, 39.3],
                            [-106.0, 39.8], [-106.5, 39.8], [-106.5, 39.3],
                        ]],
                    },
                },
                {
                    "attributes": {"zone_id": "VS", "name": "Aspen"},
                    "geometry": {
                        "rings": [[
                            [-106.5, 39.0], [-106.0, 39.0],
                            [-106.0, 39.6], [-106.5, 39.6], [-106.5, 39.0],
                        ]],
                    },
                },
            ],
        }
        respx.get(url__regex=r".*FeatureServer.*").mock(
            return_value=httpx.Response(200, json=multi_zone)
        )
        count = await client.update_zone_station_mapping()
        assert count == 2
        rows = caic_db.execute(
            "SELECT zone_id FROM zone_station_mapping "
            "WHERE station_id = '335:CO:SNTL' ORDER BY zone_id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][0] == "BTAC"
        assert rows[1][0] == "VS"


# --- Deduplication ---


class TestDeduplication:
    @respx.mock
    async def test_danger_upsert_no_duplicates(self, client, caic_db, danger_rating_response):
        route = respx.get(f"{CAIC_API_BASE}/products/all")
        route.mock(return_value=httpx.Response(200, json=danger_rating_response))
        await client.ingest_danger_ratings(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 16),
        )
        route.mock(return_value=httpx.Response(200, json=danger_rating_response))
        await client.ingest_danger_ratings(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 16),
        )
        count = caic_db.execute("SELECT COUNT(*) FROM danger_ratings").fetchone()[0]
        assert count == 6

    @respx.mock
    async def test_problem_upsert_no_duplicates(self, client, caic_db, problem_type_response):
        route = respx.get(f"{CAIC_API_BASE}/products/all")
        route.mock(return_value=httpx.Response(200, json=problem_type_response))
        await client.ingest_problem_types(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 15),
        )
        route.mock(return_value=httpx.Response(200, json=problem_type_response))
        await client.ingest_problem_types(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 15),
        )
        count = caic_db.execute("SELECT COUNT(*) FROM problem_types").fetchone()[0]
        assert count == 2


# --- Class distribution ---


class TestClassDistribution:
    def test_danger_distribution(self, client, caic_db):
        caic_db.execute("""
            INSERT INTO danger_ratings (zone_id, date, elevation_band, danger_level, source)
            VALUES
            ('BTAC', '2024-01-15', 'above_treeline', 3, 'caic'),
            ('BTAC', '2024-01-15', 'near_treeline', 2, 'caic'),
            ('BTAC', '2024-01-15', 'below_treeline', 1, 'caic'),
            ('BTAC', '2024-01-16', 'above_treeline', 3, 'caic'),
            ('BTAC', '2024-01-16', 'near_treeline', 3, 'caic'),
            ('BTAC', '2024-01-16', 'below_treeline', 2, 'caic')
        """)
        dist = client.report_class_distribution(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 16),
        )
        assert dist["danger_counts"][1] == 1
        assert dist["danger_counts"][2] == 2
        assert dist["danger_counts"][3] == 3
        total = sum(dist["danger_counts"].values())
        assert total == 6

    def test_problem_distribution(self, client, caic_db):
        caic_db.execute("""
            INSERT INTO problem_types
            (zone_id, date, problem_type, likelihood, ordering, source)
            VALUES
            ('BTAC', '2024-01-15', 'Persistent Slab', 'Likely', 0, 'caic'),
            ('BTAC', '2024-01-15', 'Storm Slab', 'Possible', 1, 'caic'),
            ('BTAC', '2024-01-16', 'Persistent Slab', 'Likely', 0, 'caic')
        """)
        dist = client.report_class_distribution(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 16),
        )
        assert dist["problem_counts"]["Persistent Slab"] == 2
        assert dist["problem_counts"]["Storm Slab"] == 1

    def test_distribution_percentages(self, client, caic_db):
        caic_db.execute("""
            INSERT INTO danger_ratings (zone_id, date, elevation_band, danger_level, source)
            VALUES
            ('BTAC', '2024-01-15', 'above_treeline', 3, 'caic'),
            ('BTAC', '2024-01-15', 'near_treeline', 3, 'caic'),
            ('BTAC', '2024-01-15', 'below_treeline', 1, 'caic'),
            ('BTAC', '2024-01-16', 'above_treeline', 1, 'caic')
        """)
        dist = client.report_class_distribution(
            start_date=datetime.date(2024, 1, 15),
            end_date=datetime.date(2024, 1, 16),
        )
        assert abs(dist["danger_pct"][3] - 50.0) < 0.1
        assert abs(dist["danger_pct"][1] - 50.0) < 0.1
