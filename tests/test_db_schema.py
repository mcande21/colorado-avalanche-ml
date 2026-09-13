from avalanche_ml.db.schema import create_tables


def test_create_tables_creates_stations(db):
    create_tables(db)
    tables = db.execute("SELECT table_name FROM information_schema.tables").fetchall()
    table_names = [t[0] for t in tables]
    assert "stations" in table_names


def test_create_tables_creates_snotel_daily(db):
    create_tables(db)
    tables = db.execute("SELECT table_name FROM information_schema.tables").fetchall()
    table_names = [t[0] for t in tables]
    assert "snotel_daily" in table_names


def test_create_tables_creates_snotel_hourly(db):
    create_tables(db)
    tables = db.execute("SELECT table_name FROM information_schema.tables").fetchall()
    table_names = [t[0] for t in tables]
    assert "snotel_hourly" in table_names


def test_create_tables_creates_ingestion_watermarks(db):
    create_tables(db)
    tables = db.execute("SELECT table_name FROM information_schema.tables").fetchall()
    table_names = [t[0] for t in tables]
    assert "ingestion_watermarks" in table_names


def test_stations_table_has_expected_columns(db):
    create_tables(db)
    cols = db.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'stations' ORDER BY ordinal_position"
    ).fetchall()
    col_names = [c[0] for c in cols]
    assert "station_id" in col_names
    assert "name" in col_names
    assert "latitude" in col_names
    assert "longitude" in col_names
    assert "elevation" in col_names
    assert "active" in col_names


def test_create_tables_is_idempotent(db):
    create_tables(db)
    create_tables(db)
    tables = db.execute("SELECT table_name FROM information_schema.tables").fetchall()
    table_names = [t[0] for t in tables]
    assert "stations" in table_names
