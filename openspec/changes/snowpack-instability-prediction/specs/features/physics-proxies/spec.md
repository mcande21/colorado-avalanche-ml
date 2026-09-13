## Purpose

Compute physics-informed proxy features that approximate SNOWPACK model outputs using only station observations, capturing key snowpack instability mechanisms.

## ADDED Requirements

### Requirement: Temperature gradient days
The system SHALL compute a cumulative temperature gradient proxy indicating faceting potential within the snowpack.

#### Scenario: Temperature gradient computation
- **WHEN** features are computed for a station and date
- **THEN** the system SHALL compute daily temperature gradient as (air_temp_range_f / snow_depth_inches), converted to K/m, and flag days where the gradient exceeds 10 K/m as "faceting days"

#### Scenario: Cumulative gradient days with duration weighting
- **WHEN** temperature gradient is computed
- **THEN** the system SHALL produce temp_gradient_days weighted by consecutive days above threshold, not just a daily crossing count — Colorado continental snowpack exceeds 10 K/m gradient for weeks at a time, so cumulative duration (consecutive days above threshold) is the Colorado-specific signal. The counter resets on the last reset event (rain-on-snow or melt-freeze cycle that exceeds a threshold)

#### Scenario: Zero snow depth handling
- **WHEN** snow depth is zero or NULL
- **THEN** the system SHALL set the temperature gradient to NULL for that day rather than dividing by zero

### Requirement: Surface hoar index
The system SHALL compute a surface hoar formation potential index based on clear-sky radiation, humidity, and wind conditions.

#### Scenario: Surface hoar index computation
- **WHEN** features are computed for a station and date
- **THEN** the system SHALL compute surface_hoar_index as the product of (clear_sky_indicator * relative_humidity_fraction) for hours where wind_speed < 2 m/s, summed over the preceding 24h

#### Scenario: Clear sky indicator derivation
- **WHEN** direct solar radiation measurements are unavailable
- **THEN** the system SHALL derive clear_sky_indicator from the absence of precipitation and low humidity (RH < 70%) during daylight hours, or from HRRR downward shortwave radiation if available

#### Scenario: No calm wind hours
- **WHEN** all hours in the 24h window have wind_speed >= 2 m/s
- **THEN** the surface_hoar_index SHALL be 0.0 for that station-date

### Requirement: Wind slab loading index
The system SHALL compute a wind slab loading potential based on wind speed, wind direction relative to terrain aspect, and concurrent precipitation.

#### Scenario: Wind slab loading computation
- **WHEN** features are computed for a station and date
- **THEN** the system SHALL compute wind_slab_loading as the 24h sum of (wind_speed * sin(wind_direction - aspect) * precip_rate) where precip_rate is the concurrent hourly precipitation

#### Scenario: Aspect determination
- **WHEN** station aspect is not directly available from metadata
- **THEN** the system SHALL derive aspect from the DEM at the station location, or use the dominant aspect of the surrounding 500m radius terrain

### Requirement: Rain-on-snow detection
The system SHALL detect rain-on-snow events, which are critical indicators of wet avalanche potential and snowpack structural change.

#### Scenario: Rain-on-snow event identification
- **WHEN** features are computed for a station and date
- **THEN** the system SHALL flag hours where precipitation > 0 AND air_temperature > 0°C (32°F) AND snow_depth > 0, and produce rain_on_snow_hours (count of flagged hours in 24h) and rain_on_snow_amount (total precipitation during flagged hours)

#### Scenario: Rain-on-snow as gradient reset
- **WHEN** a rain-on-snow event exceeds 5mm liquid equivalent in 24h
- **THEN** the event SHALL reset the cumulative temperature gradient days counter to zero

### Requirement: Snow depth anomaly
The system SHALL compute snow depth relative to the 30-year SNOTEL climatological normal for each station and day-of-year.

#### Scenario: Anomaly computation
- **WHEN** features are computed for a station and date
- **THEN** the system SHALL compute snow_depth_anomaly as (observed_depth - climatological_median) / climatological_std for the matching station and day-of-year

#### Scenario: Climatological normal source
- **WHEN** the 30-year normal is needed for a station
- **THEN** the system SHALL derive it from the AWDB period-of-record statistics or compute it from stored historical daily observations if at least 10 years of data are available

#### Scenario: Insufficient historical data
- **WHEN** fewer than 10 years of historical observations exist for a station
- **THEN** the system SHALL set snow_depth_anomaly to NULL for that station

### Requirement: Season context features
The system SHALL compute seasonal context features capturing early-season and transitional snowpack states.

#### Scenario: Early season thin snowpack
- **WHEN** features are computed for a station and date
- **THEN** the system SHALL produce early_season_flag as 1 when the date is before January 15 AND snow depth is below the 25th percentile of the climatological normal for that date

#### Scenario: Freeze-thaw cycle count
- **WHEN** features are computed for a station and date
- **THEN** the system SHALL produce freeze_thaw_cycles as the count of days in the preceding 14 days where air temperature crossed 0°C (32°F) in both directions within a 24h period

### Requirement: Physics proxy feature vector
The system SHALL produce a consistent set of physics proxy features per station-date.

#### Scenario: Complete proxy feature set
- **WHEN** all physics proxy features are computed
- **THEN** the feature vector SHALL include: temp_gradient_days, surface_hoar_index, wind_slab_loading, rain_on_snow_hours, rain_on_snow_amount, snow_depth_anomaly, early_season_flag, freeze_thaw_cycles — yielding 8 physics proxy features per station-date
