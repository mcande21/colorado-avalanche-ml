import duckdb
import pytest


@pytest.fixture
def db():
    """In-memory DuckDB connection for tests."""
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def snotel_stations_response():
    """Mock AWDB API response for station metadata."""
    return [
        {
            "stationTriplet": "335:CO:SNTL",
            "name": "Berthoud Summit",
            "latitude": 39.7975,
            "longitude": -105.7778,
            "elevation": 11300.0,
            "huc": "10190005",
            "stateCode": "CO",
            "countyName": "Clear Creek",
        },
        {
            "stationTriplet": "412:CO:SNTL",
            "name": "Copper Mountain",
            "latitude": 39.4972,
            "longitude": -106.1625,
            "elevation": 10520.0,
            "huc": "10190004",
            "stateCode": "CO",
            "countyName": "Summit",
        },
        {
            "stationTriplet": "415:CO:SNTL",
            "name": "Cumbres Trestle",
            "latitude": 36.9883,
            "longitude": -106.4478,
            "elevation": 10040.0,
            "huc": "13020101",
            "stateCode": "CO",
            "countyName": "Conejos",
        },
    ]


@pytest.fixture
def snotel_daily_response():
    """Mock AWDB API response for daily data."""
    return [
        {
            "stationTriplet": "335:CO:SNTL",
            "beginDate": "2024-01-01",
            "endDate": "2024-01-03",
            "values": [
                {"date": "2024-01-01", "swe": 12.5, "snowDepth": 38.0,
                 "airTempMin": 5.0, "airTempMax": 22.0, "airTempAvg": 13.5,
                 "precipIncrement": 0.3},
                {"date": "2024-01-02", "swe": 12.8, "snowDepth": 39.0,
                 "airTempMin": -2.0, "airTempMax": 18.0, "airTempAvg": 8.0,
                 "precipIncrement": 0.5},
                {"date": "2024-01-03", "swe": 13.5, "snowDepth": 41.0,
                 "airTempMin": 10.0, "airTempMax": 28.0, "airTempAvg": 19.0,
                 "precipIncrement": 0.8},
            ],
        },
    ]


@pytest.fixture
def snotel_hourly_response():
    """Mock AWDB API response for hourly data."""
    return [
        {
            "stationTriplet": "335:CO:SNTL",
            "beginDate": "2024-01-01 00:00",
            "endDate": "2024-01-01 03:00",
            "values": [
                {"dateTime": "2024-01-01 00:00", "swe": 12.5, "snowDepth": 38.0,
                 "airTemp": 15.0, "precipAccum": 0.0},
                {"dateTime": "2024-01-01 01:00", "swe": 12.5, "snowDepth": 38.1,
                 "airTemp": 14.0, "precipAccum": 0.1},
                {"dateTime": "2024-01-01 02:00", "swe": 12.6, "snowDepth": 38.2,
                 "airTemp": 12.0, "precipAccum": 0.2},
                {"dateTime": "2024-01-01 03:00", "swe": 12.6, "snowDepth": 38.3,
                 "airTemp": 11.0, "precipAccum": 0.3},
            ],
        },
    ]
