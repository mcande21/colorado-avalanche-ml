## Purpose

Ingest CAIC avalanche danger ratings and problem type labels as training targets, with the Open Avalanche Project dataset as a fallback source for historical coverage.

## ADDED Requirements

### Requirement: CAIC danger rating ingestion
The system SHALL ingest daily avalanche danger ratings from CAIC for all Colorado forecast zones, capturing per-elevation-band danger levels (1-5 scale: Low, Moderate, Considerable, High, Extreme).

#### Scenario: Successful danger rating retrieval
- **WHEN** the CAIC ingestion process is invoked for a date range
- **THEN** the system SHALL retrieve danger ratings via caic-python, store them in DuckDB with columns (zone_id, date, elevation_band, danger_level), where elevation_band is one of (below_treeline, near_treeline, above_treeline) and danger_level is an integer 1-5

#### Scenario: CAIC source unavailable
- **WHEN** the caic-python client fails to retrieve data (scraping failure, site structure change, or connection error)
- **THEN** the system SHALL log the failure with error details, attempt retrieval from the Open Avalanche Project fallback, and record which source provided each label

### Requirement: Avalanche problem type ingestion
The system SHALL ingest avalanche problem type classifications from CAIC forecasts, capturing problem type, likelihood, size, and aspect/elevation associations.

#### Scenario: Forecast with multiple problem types
- **WHEN** a CAIC forecast lists multiple avalanche problem types for a zone and date
- **THEN** the system SHALL store each problem type as a separate record with columns (zone_id, date, problem_type, likelihood, size_min, size_max, aspects, elevation_bands), preserving the ordering from the forecast

#### Scenario: Supported problem type vocabulary
- **WHEN** problem types are ingested
- **THEN** the system SHALL normalize them to the standard set: Persistent Slab, Storm Slab, Loose Wet, Loose Dry, Wind Slab, Wet Slab, Deep Slab, Cornice Fall, Glide

### Requirement: Open Avalanche Project CSV ingestion (primary pre-2021)
The system SHALL ingest the Open Avalanche Project CSV dataset as the primary label source for Dec 2013 through Apr 2021.

#### Scenario: OAP CSV download and filtering
- **WHEN** OAP ingestion is invoked
- **THEN** the system SHALL download from `https://github.com/scottcha/OpenAvalancheProject/raw/master/Data/CleanedForecastsNWAC_CAIC_UAC_CAC.V1.2013-2021.zip`, filter to the ~11,675 Colorado rows across 10 zones (Dec 2013 - Apr 2021), and tag records with source='oap'

#### Scenario: OAP problem type ingestion
- **WHEN** OAP data is parsed
- **THEN** the system SHALL extract all 8 problem types (LooseDry, LooseWet, StormSlabs, WindSlab, PersistentSlab, DeepPersistentSlab, WetSlabs, Cornices, Glide) with their likelihood, size, and aspect-elevation octagon per type, mapping to the standard vocabulary

#### Scenario: OAP danger rating extraction
- **WHEN** OAP data is parsed
- **THEN** the system SHALL extract Day1 danger ratings per elevation band (above/near/below treeline) and store them in caic_danger with source='oap'

### Requirement: avalanche.org v2 API extended ingestion (Nov 2019 - present)
The system SHALL ingest avalanche forecast data from the avalanche.org v2 API, extending back to Nov 2019.

#### Scenario: Product listing retrieval
- **WHEN** API ingestion is invoked for a date range
- **THEN** the system SHALL query `GET products?avalanche_center_id=CAIC&date_start=&date_end=` to retrieve forecast product listings, paginating as needed

#### Scenario: Product detail with problem types
- **WHEN** a product listing is retrieved
- **THEN** the system SHALL fetch `GET product/{id}` for each product to extract the `forecast_avalanche_problems` array, capturing problem type name, likelihood, location (aspect+elevation), and size

#### Scenario: Problem type vocabulary from API
- **WHEN** problem types are ingested from the API
- **THEN** the system SHALL reference the 9 canonical problem types from `GET avalanche-problems` and normalize to the standard vocabulary

#### Scenario: API coverage window
- **WHEN** API ingestion date range is configured
- **THEN** the system SHALL support ingestion back to Nov 2019 (confirmed API availability), not just Nov 2022 as originally scoped

### Requirement: Kaggle supplementary dataset
The system SHALL support ingestion from the Schwartzreich Kaggle dataset as a supplementary and cross-validation source.

#### Scenario: Kaggle dataset ingestion
- **WHEN** supplementary data ingestion is invoked
- **THEN** the system SHALL ingest from `https://www.kaggle.com/datasets/justinschwartzreich/colorado-avalanche-danger-and-weather-2013-2022`, containing 28,140 region-band-days (Dec 2013 - Apr 2022) across 8 CAIC zones with GHCN weather features, and tag records with source='kaggle'

### Requirement: Three-source coverage strategy
The system SHALL assemble continuous label coverage across 12 seasons using three complementary sources.

#### Scenario: Coverage assembly
- **WHEN** training data is assembled
- **THEN** the system SHALL combine OAP (Dec 2013 - Apr 2021) + avalanche.org API (Nov 2019 - present) for continuous 12-season coverage, with Kaggle as supplementary validation

#### Scenario: Overlap cross-validation
- **WHEN** data from OAP and API overlap (Nov 2019 - Apr 2021)
- **THEN** the system SHALL cross-validate danger ratings and problem types between sources, flagging and logging discrepancies exceeding 10% mismatch rate per zone

#### Scenario: Source provenance tracking
- **WHEN** labels are stored from any source
- **THEN** each record SHALL include a source column ('caic', 'oap', 'api', or 'kaggle') and an ingested_at timestamp

#### Scenario: Source priority
- **WHEN** multiple sources provide labels for the same zone-date-band
- **THEN** the system SHALL prefer API over OAP over Kaggle, with the selected source recorded in the source column

### Requirement: Zone-to-station mapping
The system SHALL maintain a mapping between CAIC forecast zones and nearby SNOTEL stations to enable label-feature alignment.

#### Scenario: Zone mapping initialization
- **WHEN** zone-station mapping is initialized
- **THEN** the system SHALL associate each CAIC zone with SNOTEL stations using the OAP zone boundary GeoJSON (`Data/USAvalancheRegions.geojson` from the OpenAvalancheProject repo) for polygon containment testing, with CAIC ArcGIS FeatureServer as fallback

#### Scenario: GeoJSON polygon containment
- **WHEN** zone-station spatial matching is performed
- **THEN** the system SHALL use GeoJSON polygon containment (point-in-polygon) rather than haversine distance approximation, producing more accurate mappings in zones with irregular boundaries

#### Scenario: Station within multiple zones
- **WHEN** a SNOTEL station falls within the boundary of multiple CAIC zones
- **THEN** the system SHALL associate the station with all overlapping zones

### Requirement: Label temporal alignment
The system SHALL align danger rating labels with observation dates, accounting for CAIC forecast issuance timing.

#### Scenario: Forecast date alignment
- **WHEN** a CAIC forecast is issued for a given date
- **THEN** the system SHALL associate the danger rating with the calendar date it describes (the forecast valid date), not the issuance date

#### Scenario: Missing forecast days
- **WHEN** no CAIC forecast exists for a given zone and date
- **THEN** the system SHALL record the date as having no label (NULL danger_level), not interpolate from adjacent days

### Requirement: Class distribution reporting
The system SHALL compute and report the class distribution of ingested labels to support imbalance-aware training.

#### Scenario: Distribution summary
- **WHEN** label ingestion completes for a date range
- **THEN** the system SHALL log the count and percentage of each danger level (1-5) and each problem type across all zones and elevation bands
