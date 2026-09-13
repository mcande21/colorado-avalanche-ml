import datetime

import httpx
import pytest
import respx

from avalanche_ml.db.schema import create_tables
from avalanche_ml.ingestion.snotel import SnotelClient

AWDB_BASE = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"


@pytest.fixture
def snotel_db(db):
    create_tables(db)
    return db


@pytest.fixture
def client(snotel_db):
    return SnotelClient(db=snotel_db)


# --- Station discovery ---


@respx.mock
async def test_discover_stores_stations(client, snotel_db, snotel_stations_response):
    respx.get(f"{AWDB_BASE}/stations").mock(
        return_value=httpx.Response(200, json=snotel_stations_response)
    )
    count = await client.discover()
    assert count == 3
    rows = snotel_db.execute("SELECT * FROM stations ORDER BY station_id").fetchall()
    assert len(rows) == 3


@respx.mock
async def test_discover_parses_metadata(client, snotel_db, snotel_stations_response):
    respx.get(f"{AWDB_BASE}/stations").mock(
        return_value=httpx.Response(200, json=snotel_stations_response)
    )
    await client.discover()
    row = snotel_db.execute(
        "SELECT station_id, name, latitude, longitude, elevation, huc "
        "FROM stations WHERE station_id = '335:CO:SNTL'"
    ).fetchone()
    assert row[0] == "335:CO:SNTL"
    assert row[1] == "Berthoud Summit"
    assert abs(row[2] - 39.7975) < 0.001
    assert abs(row[3] - (-105.7778)) < 0.001
    assert abs(row[4] - 11300.0) < 0.1
    assert row[5] == "10190005"


@respx.mock
async def test_discover_updates_existing_stations(client, snotel_db, snotel_stations_response):
    route = respx.get(f"{AWDB_BASE}/stations")
    route.mock(return_value=httpx.Response(200, json=snotel_stations_response))
    await client.discover()

    updated = [dict(s) for s in snotel_stations_response]
    updated[0]["name"] = "Berthoud Summit Updated"
    route.mock(return_value=httpx.Response(200, json=updated))
    await client.discover()

    row = snotel_db.execute(
        "SELECT name FROM stations WHERE station_id = '335:CO:SNTL'"
    ).fetchone()
    assert row[0] == "Berthoud Summit Updated"


# --- Daily observation ingestion ---


@respx.mock
async def test_ingest_daily_stores_readings(client, snotel_db, snotel_daily_response):
    respx.get(f"{AWDB_BASE}/data").mock(
        return_value=httpx.Response(200, json=snotel_daily_response)
    )
    count = await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 3),
        resolution="daily",
    )
    assert count == 3
    rows = snotel_db.execute("SELECT * FROM snotel_daily ORDER BY date").fetchall()
    assert len(rows) == 3


@respx.mock
async def test_ingest_daily_parses_values(client, snotel_db, snotel_daily_response):
    respx.get(f"{AWDB_BASE}/data").mock(
        return_value=httpx.Response(200, json=snotel_daily_response)
    )
    await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 3),
        resolution="daily",
    )
    row = snotel_db.execute(
        "SELECT swe_inches, snow_depth_inches, air_temp_min_f, air_temp_max_f, "
        "air_temp_mean_f, precip_increment_inches "
        "FROM snotel_daily WHERE date = '2024-01-01'"
    ).fetchone()
    assert abs(row[0] - 12.5) < 0.01
    assert abs(row[1] - 38.0) < 0.01
    assert abs(row[2] - 5.0) < 0.01
    assert abs(row[3] - 22.0) < 0.01
    assert abs(row[4] - 13.5) < 0.01
    assert abs(row[5] - 0.3) < 0.01


@respx.mock
async def test_ingest_daily_upserts_on_overlap(client, snotel_db, snotel_daily_response):
    import copy
    route = respx.get(f"{AWDB_BASE}/data")
    route.mock(return_value=httpx.Response(200, json=snotel_daily_response))
    await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 3),
        resolution="daily",
    )
    updated = copy.deepcopy(snotel_daily_response)
    updated[0]["values"][0]["swe"] = 99.9
    route.mock(return_value=httpx.Response(200, json=updated))
    await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 3),
        resolution="daily",
    )
    rows = snotel_db.execute("SELECT * FROM snotel_daily").fetchall()
    assert len(rows) == 3
    row = snotel_db.execute(
        "SELECT swe_inches FROM snotel_daily WHERE date = '2024-01-01'"
    ).fetchone()
    assert abs(row[0] - 99.9) < 0.01


# --- Hourly observation ingestion ---


@respx.mock
async def test_ingest_hourly_stores_readings(client, snotel_db, snotel_hourly_response):
    respx.get(f"{AWDB_BASE}/data").mock(
        return_value=httpx.Response(200, json=snotel_hourly_response)
    )
    count = await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 1),
        resolution="hourly",
    )
    assert count == 4
    rows = snotel_db.execute("SELECT * FROM snotel_hourly ORDER BY timestamp").fetchall()
    assert len(rows) == 4


# --- Data quality validation ---


@respx.mock
async def test_ingest_flags_negative_swe(client, snotel_db):
    bad_response = [{
        "stationTriplet": "335:CO:SNTL",
        "beginDate": "2024-01-01",
        "endDate": "2024-01-01",
        "values": [
            {"date": "2024-01-01", "swe": -5.0, "snowDepth": 38.0,
             "airTempMin": 5.0, "airTempMax": 22.0, "airTempAvg": 13.5,
             "precipIncrement": 0.3},
        ],
    }]
    respx.get(f"{AWDB_BASE}/data").mock(
        return_value=httpx.Response(200, json=bad_response)
    )
    await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 1),
        resolution="daily",
    )
    row = snotel_db.execute(
        "SELECT quality_flag FROM snotel_daily WHERE date = '2024-01-01'"
    ).fetchone()
    assert row[0] is not None
    assert "swe_negative" in row[0]


@respx.mock
async def test_ingest_flags_temp_out_of_range(client, snotel_db):
    bad_response = [{
        "stationTriplet": "335:CO:SNTL",
        "beginDate": "2024-01-01",
        "endDate": "2024-01-01",
        "values": [
            {"date": "2024-01-01", "swe": 12.0, "snowDepth": 38.0,
             "airTempMin": -70.0, "airTempMax": 22.0, "airTempAvg": -24.0,
             "precipIncrement": 0.3},
        ],
    }]
    respx.get(f"{AWDB_BASE}/data").mock(
        return_value=httpx.Response(200, json=bad_response)
    )
    await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 1),
        resolution="daily",
    )
    row = snotel_db.execute(
        "SELECT quality_flag FROM snotel_daily WHERE date = '2024-01-01'"
    ).fetchone()
    assert row[0] is not None
    assert "temp_out_of_range" in row[0]


@respx.mock
async def test_ingest_flags_discontinuity(client, snotel_db):
    disc_response = [{
        "stationTriplet": "335:CO:SNTL",
        "beginDate": "2024-01-01",
        "endDate": "2024-01-02",
        "values": [
            {"date": "2024-01-01", "swe": 10.0, "snowDepth": 30.0,
             "airTempMin": 5.0, "airTempMax": 22.0, "airTempAvg": 13.5,
             "precipIncrement": 0.3},
            {"date": "2024-01-02", "swe": 35.0, "snowDepth": 30.0,
             "airTempMin": 5.0, "airTempMax": 22.0, "airTempAvg": 13.5,
             "precipIncrement": 0.3},
        ],
    }]
    respx.get(f"{AWDB_BASE}/data").mock(
        return_value=httpx.Response(200, json=disc_response)
    )
    await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 2),
        resolution="daily",
    )
    row = snotel_db.execute(
        "SELECT quality_flag FROM snotel_daily WHERE date = '2024-01-02'"
    ).fetchone()
    assert row[0] is not None
    assert "swe_discontinuity" in row[0]


@respx.mock
async def test_ingest_no_flag_on_clean_data(client, snotel_db, snotel_daily_response):
    respx.get(f"{AWDB_BASE}/data").mock(
        return_value=httpx.Response(200, json=snotel_daily_response)
    )
    await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 3),
        resolution="daily",
    )
    rows = snotel_db.execute(
        "SELECT quality_flag FROM snotel_daily"
    ).fetchall()
    for row in rows:
        assert row[0] is None


# --- Watermark-based incremental ingestion ---


@respx.mock
async def test_ingest_sets_watermark(client, snotel_db, snotel_daily_response):
    respx.get(f"{AWDB_BASE}/data").mock(
        return_value=httpx.Response(200, json=snotel_daily_response)
    )
    await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 3),
        resolution="daily",
    )
    wm = snotel_db.execute(
        "SELECT last_timestamp FROM ingestion_watermarks "
        "WHERE source = 'snotel' AND entity_id = '335:CO:SNTL' AND resolution = 'daily'"
    ).fetchone()
    assert wm is not None
    assert wm[0].date() == datetime.date(2024, 1, 3)


@respx.mock
async def test_incremental_ingest_uses_watermark(client, snotel_db, snotel_daily_response):
    respx.get(f"{AWDB_BASE}/data").mock(
        return_value=httpx.Response(200, json=snotel_daily_response)
    )
    await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 3),
        resolution="daily",
    )

    new_response = [{
        "stationTriplet": "335:CO:SNTL",
        "beginDate": "2024-01-04",
        "endDate": "2024-01-04",
        "values": [
            {"date": "2024-01-04", "swe": 14.0, "snowDepth": 42.0,
             "airTempMin": 0.0, "airTempMax": 20.0, "airTempAvg": 10.0,
             "precipIncrement": 0.2},
        ],
    }]
    respx.get(f"{AWDB_BASE}/data").mock(
        return_value=httpx.Response(200, json=new_response)
    )
    count = await client.ingest_incremental(
        station_id="335:CO:SNTL",
        resolution="daily",
    )
    assert count == 1
    total = snotel_db.execute("SELECT COUNT(*) FROM snotel_daily").fetchone()[0]
    assert total == 4


@respx.mock
async def test_backfill_ignores_watermark(client, snotel_db, snotel_daily_response):
    snotel_db.execute(
        "INSERT INTO ingestion_watermarks VALUES "
        "('snotel', '335:CO:SNTL', 'daily', '2024-01-10', CURRENT_TIMESTAMP)"
    )
    respx.get(f"{AWDB_BASE}/data").mock(
        return_value=httpx.Response(200, json=snotel_daily_response)
    )
    count = await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 3),
        resolution="daily",
    )
    assert count == 3


# --- Error handling ---


@respx.mock
async def test_retry_on_5xx(client, snotel_db, snotel_daily_response):
    route = respx.get(f"{AWDB_BASE}/data")
    route.side_effect = [
        httpx.Response(503),
        httpx.Response(503),
        httpx.Response(200, json=snotel_daily_response),
    ]
    count = await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 3),
        resolution="daily",
    )
    assert count == 3
    assert route.call_count == 3


@respx.mock
async def test_retry_exhaustion_raises(client, snotel_db):
    respx.get(f"{AWDB_BASE}/data").mock(
        return_value=httpx.Response(503)
    )
    with pytest.raises(httpx.HTTPStatusError):
        await client.ingest(
            station_id="335:CO:SNTL",
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 3),
            resolution="daily",
        )


@respx.mock
async def test_rate_limit_429_retry(client, snotel_db, snotel_daily_response):
    route = respx.get(f"{AWDB_BASE}/data")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json=snotel_daily_response),
    ]
    count = await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 3),
        resolution="daily",
    )
    assert count == 3


@respx.mock
async def test_empty_response_returns_zero(client, snotel_db):
    respx.get(f"{AWDB_BASE}/data").mock(
        return_value=httpx.Response(200, json=[])
    )
    count = await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 3),
        resolution="daily",
    )
    assert count == 0


# --- Partial / NULL data ---


@respx.mock
async def test_partial_data_stores_nulls(client, snotel_db):
    partial_response = [{
        "stationTriplet": "335:CO:SNTL",
        "beginDate": "2024-01-01",
        "endDate": "2024-01-01",
        "values": [
            {"date": "2024-01-01", "swe": 12.5, "snowDepth": None,
             "airTempMin": 5.0, "airTempMax": None, "airTempAvg": None,
             "precipIncrement": 0.3},
        ],
    }]
    respx.get(f"{AWDB_BASE}/data").mock(
        return_value=httpx.Response(200, json=partial_response)
    )
    count = await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 1),
        resolution="daily",
    )
    assert count == 1
    row = snotel_db.execute(
        "SELECT snow_depth_inches, air_temp_max_f FROM snotel_daily"
    ).fetchone()
    assert row[0] is None
    assert row[1] is None


# --- DuckDB insertion verification ---


@respx.mock
async def test_ingest_daily_correct_station_id(client, snotel_db, snotel_daily_response):
    respx.get(f"{AWDB_BASE}/data").mock(
        return_value=httpx.Response(200, json=snotel_daily_response)
    )
    await client.ingest(
        station_id="335:CO:SNTL",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 3),
        resolution="daily",
    )
    ids = snotel_db.execute(
        "SELECT DISTINCT station_id FROM snotel_daily"
    ).fetchall()
    assert len(ids) == 1
    assert ids[0][0] == "335:CO:SNTL"
