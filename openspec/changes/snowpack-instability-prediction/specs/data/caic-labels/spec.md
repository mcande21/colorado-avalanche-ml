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

### Requirement: Open Avalanche Project fallback
The system SHALL support ingestion from the Open Avalanche Project labeled dataset (2015-2021) as a fallback and historical supplement.

#### Scenario: Historical backfill from OAP
- **WHEN** CAIC data is unavailable for dates within the OAP coverage period (2015-2021)
- **THEN** the system SHALL ingest OAP labels, mapping them to the same schema as CAIC labels, and tag records with source='oap'

#### Scenario: Source provenance tracking
- **WHEN** labels are stored from any source
- **THEN** each record SHALL include a source column ('caic' or 'oap') and an ingested_at timestamp

### Requirement: Zone-to-station mapping
The system SHALL maintain a mapping between CAIC forecast zones and nearby SNOTEL stations to enable label-feature alignment.

#### Scenario: Zone mapping initialization
- **WHEN** zone-station mapping is initialized
- **THEN** the system SHALL associate each CAIC zone with SNOTEL stations within or proximal to the zone boundary, using the CAIC ArcGIS FeatureServer for zone geometries and station coordinates for spatial matching

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
