import duckdb


def create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stations (
            station_id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            latitude DOUBLE NOT NULL,
            longitude DOUBLE NOT NULL,
            elevation DOUBLE NOT NULL,
            huc VARCHAR,
            state_code VARCHAR,
            county VARCHAR,
            active BOOLEAN DEFAULT TRUE,
            catalog_updated_at TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS snotel_hourly (
            station_id VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            swe_inches DOUBLE,
            snow_depth_inches DOUBLE,
            air_temp_f DOUBLE,
            precip_accum_inches DOUBLE,
            quality_flag VARCHAR,
            PRIMARY KEY (station_id, timestamp)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS snotel_daily (
            station_id VARCHAR NOT NULL,
            date DATE NOT NULL,
            swe_inches DOUBLE,
            snow_depth_inches DOUBLE,
            air_temp_min_f DOUBLE,
            air_temp_max_f DOUBLE,
            air_temp_mean_f DOUBLE,
            precip_increment_inches DOUBLE,
            quality_flag VARCHAR,
            PRIMARY KEY (station_id, date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS danger_ratings (
            zone_id VARCHAR NOT NULL,
            date DATE NOT NULL,
            elevation_band VARCHAR NOT NULL,
            danger_level INTEGER,
            source VARCHAR NOT NULL DEFAULT 'caic',
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (zone_id, date, elevation_band, source)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS problem_types (
            zone_id VARCHAR NOT NULL,
            date DATE NOT NULL,
            problem_type VARCHAR NOT NULL,
            likelihood VARCHAR,
            size_min DOUBLE,
            size_max DOUBLE,
            aspects VARCHAR,
            elevation_bands VARCHAR,
            ordering INTEGER,
            source VARCHAR NOT NULL DEFAULT 'caic',
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (zone_id, date, problem_type, source)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS zone_station_mapping (
            zone_id VARCHAR NOT NULL,
            station_id VARCHAR NOT NULL,
            PRIMARY KEY (zone_id, station_id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_watermarks (
            source VARCHAR NOT NULL,
            entity_id VARCHAR NOT NULL,
            resolution VARCHAR NOT NULL,
            last_timestamp TIMESTAMP NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source, entity_id, resolution)
        )
    """)
