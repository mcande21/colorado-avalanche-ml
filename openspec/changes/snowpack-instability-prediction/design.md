## Context

See [proposal.md](proposal.md) for motivation. This design addresses a greenfield Python ML pipeline that ingests heterogeneous time-series data from three external sources (SNOTEL, CAIC, HRRR), engineers features in a local columnar store, trains two competing model architectures, and serves predictions via REST API. The constraints shaping this design:

- **No managed infrastructure.** The system runs as a single Docker container with local DuckDB — no cloud databases, no distributed compute. This bounds complexity but limits horizontal scaling.
- **Heterogeneous temporal resolution.** SNOTEL reports hourly, CAIC labels are daily per zone, HRRR grids are hourly on a 3km mesh. Temporal alignment is the central data engineering challenge.
- **Severe class imbalance.** High-danger days (level 4-5) represent ~1.1% of samples. Both model architectures must handle this or produce a useless classifier.
- **Safety-critical domain.** Predictions influence backcountry decisions. Every prediction must include SHAP explanations and confidence metrics — black-box outputs are unacceptable.
- **CAIC data fragility.** CAIC has no official API; `caic-python` scrapes the website. This is the single most fragile dependency.

## Goals / Non-Goals

**Goals:**

- Define the component architecture and data flow connecting all 10 capabilities
- Establish the storage strategy (DuckDB + Parquet + MLflow) and why it fits
- Specify the phased implementation order and inter-phase dependencies
- Resolve build-vs-buy decisions for each component

**Non-Goals:**

- Multi-region deployment or horizontal scaling — single-container is the target
- Real-time streaming ingestion — daily batch is sufficient for avalanche forecasting
- Mobile or web frontend — API-only in this change
- Production alerting/paging infrastructure — health endpoint is the monitoring surface

## System Architecture

```
                                    ┌──────────────────────┐
                                    │    FastAPI Service    │
                                    │  /predict /history    │
                                    │  /stations /health    │
                                    └─────────┬────────────┘
                                              │
                              ┌───────────────┼───────────────┐
                              │               │               │
                     ┌────────▼──┐   ┌────────▼──┐   ┌───────▼───────┐
                     │ RF Model  │   │LSTM Model │   │  Ensemble     │
                     │(2-stage)  │   │(hierarchi-│   │  (weighted    │
                     │           │   │ cal)      │   │   average)    │
                     └────────┬──┘   └────────┬──┘   └───────┬───────┘
                              │               │               │
                              └───────────────┼───────────────┘
                                              │
                              ┌────────────────────────────────┐
                              │      DuckDB Feature Store      │
                              │  ┌──────────┐ ┌─────────────┐ │
                              │  │ Weather  │ │  Physics    │ │
                              │  │ Windows  │ │  Proxies    │ │
                              │  └────┬─────┘ └──────┬──────┘ │
                              │       │              │        │
                              │  ┌────▼──────────────▼─────┐  │
                              │  │  Raw Observations       │  │
                              │  │  (SNOTEL hourly/daily)  │  │
                              │  └────┬──────────────┬─────┘  │
                              └───────┼──────────────┼────────┘
                                      │              │
                    ┌─────────────────┼──────────────┼──────────────┐
                    │                 │              │              │
           ┌────────▼──┐    ┌────────▼──┐  ┌───────▼───┐  ┌───────▼───┐
           │  SNOTEL   │    │   CAIC    │  │   HRRR    │  │   DEM     │
           │  Ingestion│    │  Labels   │  │ Ingestion │  │  Terrain  │
           │  (AWDB)   │    │(caic-py)  │  │ (Herbie)  │  │  (3DEP)  │
           └───────────┘    └───────────┘  └───────────┘  └───────────┘
```

**Data flow, end to end:**

1. **Ingestion layer** pulls raw data from three external APIs into DuckDB tables with watermark-based incremental updates
2. **Feature layer** reads raw observations, computes rolling window aggregations (90 features) and physics proxy features (8 features) per station-date, writes to feature tables
3. **Label layer** aligns CAIC danger ratings and problem types to station-date-elevation-band via zone-station mapping
4. **Model layer** reads feature matrix + aligned labels, trains RF and LSTM with temporal splits, logs to MLflow
5. **Serving layer** loads trained models, reads latest features from DuckDB, returns predictions with SHAP explanations
6. **Spatial layer** (Phase 2) adds HRRR grid processing, DEM terrain features, and XGBoost downscaling to enable prediction at non-station locations

## Decisions

### D1: DuckDB over PostgreSQL for the feature store

**Choice:** DuckDB (embedded, file-based columnar database).

**Rationale:** The workload is analytical — time-series aggregation, window functions, columnar scans over millions of rows. DuckDB handles this natively without a server process. The entire system runs in a single container; adding a PostgreSQL dependency doubles operational complexity for zero benefit. DuckDB's Parquet import/export is native, fitting the data pipeline pattern. Write concurrency is irrelevant — only one ingestion process writes at a time.

**Alternative considered:** PostgreSQL with TimescaleDB. Rejected because it requires a separate server, connection management, and schema migrations for a single-user analytical workload. The only advantage (concurrent writes) is unnecessary here.

### D2: Random Forest baseline before LSTM

**Choice:** Implement RF first, LSTM second. RF is the operational model; LSTM is the experimental comparison.

**Rationale:** Published avalanche ML research (Schwartzreich & Rodriguez 2026, Swiss SLF) shows RF achieves operational accuracy and deep learning has not yet demonstrated consistent superiority in this domain. RF trains in minutes, requires no GPU, and produces deterministic results. SHAP on RF is fast (tree SHAP). The LSTM may improve temporal pattern capture (e.g., multi-day storm loading), but that is a hypothesis to test, not a given. Building RF first delivers a working system earlier and establishes the performance floor.

**Alternative considered:** Skip RF, go straight to LSTM. Rejected because (a) RF may be sufficient, (b) no GPU infrastructure exists yet, (c) published benchmarks favor RF, (d) RF provides the comparison baseline the LSTM spec requires.

### D3: Two-stage classification (problem types → danger level)

**Choice:** Stage 1 predicts which avalanche problem types are active; Stage 2 uses those predictions as features alongside weather/physics to predict danger level 1-5.

**Rationale:** This matches the causal structure of avalanche forecasting — danger level is a function of which problems exist and their severity. The published two-stage approach outperforms direct danger-level prediction because problem type labels provide structured intermediate supervision. CAIC forecasters think in problem types; mirroring this in the model architecture also makes SHAP explanations more interpretable to domain experts.

**Alternative considered:** Direct 5-class danger prediction. Rejected because published results show lower accuracy (57% direct vs 76% binary via two-stage), and the intermediate problem type predictions are independently valuable output.

### D4: Binary/3-class primary output, full 5-class secondary

**Choice:** Primary predictions use binary (dangerous/not-dangerous) or 3-class (Low-Moderate / Considerable / High-Extreme) groupings. Full 5-class distribution is included as a secondary output.

**Rationale:** 5-class accuracy tops out around 57% with available data. Grouping adjacent classes yields actionable accuracy (76% binary). The probability distribution over all 5 levels still ships with every prediction for users who want granularity, but the headline prediction uses the grouping that the model can actually support.

### D5: Per-elevation-band models, not multi-task

**Choice:** Train separate model instances for each elevation band (above/near/below treeline).

**Rationale:** Problem type distributions differ sharply by elevation. Wind Slab dominates above treeline; Loose Wet dominates below. A single model with elevation as a categorical input would need to learn these different distributions implicitly. Separate models are simpler, independently tunable, and match how CAIC structures their forecasts. The LSTM spec allows choosing between separate-model and shared-model based on validation performance, but the default architecture is separate.

### D6: SHAP explanations as a mandatory output, not optional

**Choice:** Every prediction includes SHAP values for the top 10 contributing features.

**Rationale:** Safety-critical domain. Backcountry users and forecasters need to understand *why* the model predicts high danger — is it wind loading? Temperature gradient? Rain-on-snow? SHAP is the only explanation method that provides theoretically grounded, consistent feature attribution. TreeSHAP is fast for RF; KernelSHAP or DeepSHAP for LSTM is slower but runs offline at prediction time, not in a latency-critical path.

### D7: Temporal splits only, no random cross-validation

**Choice:** Strictly temporal train/val/test splits: training Dec 2013 through Jun 2022, validation 2022-24 seasons, test 2024-25 season.

**Rationale:** Weather time series have strong temporal autocorrelation. Random splitting leaks future information into training. Published avalanche ML papers that use random splits report inflated metrics. Temporal splits simulate the real deployment scenario: predict today's danger from past data only.

### D8: SNOTEL bias correction for HRRR precipitation

**Choice:** Station-based correction using a rolling 30-day ratio of SNOTEL observed to HRRR predicted precipitation, with inverse-distance weighting for gridpoints far from stations.

**Rationale:** HRRR has a documented 25-65% precipitation bias in Colorado mountain terrain. Raw HRRR precipitation values would produce systematically wrong snow loading estimates, poisoning the features that matter most for avalanche prediction. The correction is simple (multiplicative ratio), leverages the ground-truth stations we already ingest, and is applied before any downstream processing.

**Alternative considered:** No correction (use raw HRRR). Rejected — the bias range is too large. Also considered a learned correction via the XGBoost downscaling model, but that introduces a Phase 2 dependency into Phase 1 ingestion.

### D9: DEM terrain features for spatial downscaling via XGBoost

**Choice:** XGBoost model mapping (HRRR gridpoint values + terrain features) → observed SNOTEL values, enabling prediction at arbitrary mountain locations.

**Rationale:** The relationship between coarse-grid (3km) weather and local conditions is primarily governed by terrain — elevation, slope, aspect, sheltering. XGBoost handles heterogeneous feature interactions well and is fast to train/infer. Leave-one-station-out cross-validation tests spatial generalization. The alternative (TopoPyScale) is a physics-based approach that the spec requires as a benchmark, not a replacement.

### D10: Cold air pool handling as a terrain-conditional correction

**Choice:** TPI-based detection (TPI@500m < -50m) triggers a correction factor for nighttime temperature under calm/clear conditions.

**Rationale:** Valley locations in Colorado routinely see 10-20°C inversions that HRRR misses at 3km resolution. Without correction, downscaled temperatures in valleys are systematically too warm, affecting temperature gradient features and wet-avalanche prediction. The correction is conditioned on measurable terrain + weather state (low TPI + low wind + clear sky), not a blanket offset.

## Expanded Data Strategy (Research Update)

Based on research findings (Schwartzreich & Rodriguez 2026, OAP dataset analysis, avalanche.org API exploration), the data pipeline and model architecture have been significantly expanded.

### Three-Source Data Pipeline

| Source | Coverage | Records | Problem Types | Notes |
|--------|----------|---------|---------------|-------|
| **OAP CSV** | Dec 2013 - Apr 2021 | ~11,675 CO rows, 10 zones | 8 types with likelihood/size/octagon | Primary pre-2021 source |
| **avalanche.org v2 API** | Nov 2019 - present | Per-product detail | 9 canonical types via product/{id} | Extended back from Nov 2022 |
| **Kaggle (Schwartzreich)** | Dec 2013 - Apr 2022 | 28,140 region-band-days, 8 zones | Danger ratings + GHCN weather | Supplementary cross-validation |

**Continuous coverage:** OAP (Dec 2013 - Apr 2021) + API (Nov 2019 - present) = 12 seasons with no gaps. The Nov 2019 - Apr 2021 overlap enables cross-validation between sources.

### Zone Boundary GeoJSON

OAP provides `Data/USAvalancheRegions.geojson` with zone polygons. This replaces the haversine approximation for zone-station mapping with exact polygon containment testing, producing more accurate mappings in zones with irregular boundaries.

### Updated Temporal Split

With 12 seasons of data (expanded from ~8):
- **Train:** Dec 2013 - Jun 2022 (~9 seasons)
- **Validation:** Oct 2022 - Jun 2024 (2 seasons)
- **Test:** Oct 2024 - Jun 2025 (1 season)

### Per-Band Model Architecture (12 Models)

Per Schwartzreich 2026, the system now trains per-elevation-band models:
- **Stage 1:** 9 models (3 problem types x 3 elevation bands) — binary classifiers for Persistent Slab, Slab Problem (storm+wind merged), Loose Wet
- **Stage 2:** 3 models (1 per elevation band) — danger level 1-5 using frozen Stage-1 predictions
- **Total:** 12 models per architecture (RF and LSTM/Transformer)

Stage 2 uses out-of-sample Stage-1 ensemble predictions (never actual labels) via staged chronological split protocol.

### Key Architecture Changes from Research

- **SMOTE dropped:** Cost-sensitive class weights only ({0:1, 1:5} baseline). Research confirmed SMOTE hurts generalization.
- **Soft-voting ensemble:** Top-3 RF configs by macro-F1 combined via averaged probability distributions.
- **Threshold at serving time:** Deploy with predict_proba, threshold adjustable per context. Default t=0.30 for safety.
- **Transformer for Persistent Slab:** Schwartzreich found Transformer (64-unit, 2-layer) beat RF, LSTM, GRU for persistent slab prediction. Added as Stage-1 option.
- **Duration accumulation:** Physics proxy temp_gradient_days weighted by consecutive days above 10 K/m threshold (Colorado continental snowpack signal).

### Benchmark Targets (Schwartzreich 2026)

| Band | Macro-F1 |
|------|----------|
| Below Treeline (BTL) | 0.544 |
| Near Treeline (NTL) | 0.525 |
| Above Treeline (ATL) | 0.508 |

## Final Model Selection

204+ RF experiments and 18 Transformer experiments (logged to MLflow) informed the final Stage-1 architecture. The result is a hybrid, not a single-model choice:

- **Persistent Slab (Stage 1):** Transformer (64-unit, 2-layer, 4-head attention, lookback=7, lr=5e-4). Beats RF (0.787 macro-F1) and the Schwartzreich & Rodriguez 2026 published benchmark (0.656-0.762) at every elevation band — ATL 0.826, NTL 0.821, BTL 0.816 — a lift of +0.064 to +0.165 F1 over the published numbers. Recall of 87-93% is safety-appropriate given Persistent Slab is the most dangerous problem type.
- **Storm Slab + Loose Wet (Stage 1):** Cost-sensitive Random Forest (balanced class weights, n_estimators=300, max_depth=10). RF remained competitive for these types; the added complexity of a Transformer wasn't justified by the experiment results.
- **Danger Level (Stage 2):** Random Forest, using frozen Stage-1 predictions as input features — including the Transformer's Persistent Slab probabilities alongside RF's Storm Slab and Loose Wet probabilities.
- **Serving:** predict_proba with a default classification threshold of t=0.30, adjustable per deployment context without retraining.

Key findings from the experiment battery:

- **SMOTE hurts generalization** — dropped in favor of cost-sensitive class weights only.
- **Threshold tuning at serving time is the best operational lever** — more impactful than further hyperparameter search.
- **`temp_gradient_consec_days`** (consecutive days above the 10 K/m faceting threshold) adds a free +0.007 to +0.010 macro-F1 lift over the daily crossing-count variant.
- **The Transformer captures temporal persistent-slab patterns RF misses** — multi-day faceting/weak-layer buildup is inherently sequential, which RF's feature-window approach only partially encodes.

This supersedes decision D2 (RF-first) for the Persistent Slab problem type specifically: the LSTM/Transformer hypothesis from D2 was tested and, for this one target, won. RF remains the production choice everywhere else in Stage 1 and for all of Stage 2.

## Component Design

### Ingestion Layer

Three async clients, each with watermark-based incremental ingestion:

| Client | Source | Auth | Schedule | Storage |
|--------|--------|------|----------|---------|
| `SnotelClient` | AWDB REST API | None | Hourly + daily tables | DuckDB `snotel_hourly`, `snotel_daily` |
| `CaicClient` | caic-python (scraper) | None | Daily | DuckDB `caic_danger`, `caic_problems` |
| `HrrrClient` | AWS S3 via Herbie | None | Hourly analysis grids | DuckDB `hrrr_station`, `hrrr_grid` |

Each client implements: `discover()` for metadata, `ingest(station/zone, start, end)` for data pull, `backfill(start, end)` for historical load. Watermarks live in a `ingestion_watermarks` table keyed on (source, entity_id, resolution).

Rate limiting: exponential backoff with 3 retries on 5xx, pause on 429 with Retry-After. CAIC fallback: on scraper failure, fall back to OAP dataset for 2015-2021 coverage.

### Feature Store Schema (DuckDB)

Core tables:

- `stations` — metadata (id, name, lat, lon, elevation, huc, active, zone_ids)
- `snotel_hourly` — raw hourly obs, quality-flagged
- `snotel_daily` — raw daily obs, quality-flagged
- `caic_danger` — (zone_id, date, elevation_band, danger_level, source)
- `caic_problems` — (zone_id, date, problem_type, likelihood, size, aspects, elevation_bands, source)
- `zone_station_map` — many-to-many zone↔station via spatial join on CAIC ArcGIS geometries
- `hrrr_station` — HRRR interpolated to station locations, bias-corrected
- `hrrr_grid` — full gridded HRRR, partitioned by date
- `features_weather` — 90 rolling window features per station-date
- `features_physics` — 8 physics proxy features per station-date
- `features_matrix` — materialized join of weather + physics + station metadata, the model input
- `predictions` — stored predictions for monitoring and history API

Temporal alignment: CAIC labels join to station-dates through `zone_station_map`. One zone maps to many stations; one station may map to multiple zones. The feature matrix is keyed on (station_id, date, elevation_band).

### Model Pipeline

**Training:**

1. Query `features_matrix` joined with `caic_danger` for the training date range
2. Apply temporal split (train: Dec 2013 - Jun 2022, val: 2022-24 seasons, test: 2024-25 season)
3. Stage 1: fit binary RF classifiers on weather+physics → 3 problem types per band (9 models) with cost-sensitive weights
4. Generate frozen Stage 1 predictions on training data (staged chronological split, out-of-sample only)
5. Stage 2: fit RF on weather+physics+frozen_Stage1_probs → danger level (3 models, 1 per band; 12 total)
6. Build soft-voting ensemble from top-3 configs by macro-F1
6. Evaluate on test set: macro F1, per-class F1, ordinal accuracy (±1), ROC-AUC, confusion matrix
7. Log everything to MLflow: params, metrics, model artifact, SHAP summary plot

**LSTM addition (Phase 1C):** Same pipeline but with PyTorch DataLoaders constructing multi-rate sequences (168 hourly, 30 daily, 60 seasonal steps). Joint training of both stages with combined loss. Focal loss replaces cross-entropy. Compared to RF on identical test set with paired bootstrap significance test.

**Ensemble:** When both models are trained, weighted average of probability distributions using inverse validation loss weights. Falls back to single model when only one is available.

### Serving Layer

FastAPI app with:

- Model loading at startup (pickle for RF, checkpoint for LSTM)
- Prediction caching in DuckDB `predictions` table (cache key: zone+band+date+model_version)
- Daily scheduler (APScheduler or similar) triggering the ingestion → feature → prediction pipeline at 06:00 UTC
- API key auth via middleware, rate limiting via slowapi
- Docker: single-stage build, health check on `/health`, env vars for DUCKDB_PATH, MODEL_PATH, API_KEY

### Spatial Downscaling (Phase 2)

- DEM terrain features computed once per location, cached
- XGBoost trained on (HRRR values at station, terrain features) → (SNOTEL observed) pairs
- Cold air pool correction applied as a post-processing step conditioned on TPI + weather state
- Output schema matches station observations, so existing feature pipelines process downscaled data identically
- TopoPyScale run as benchmark for the evaluation report, not as a production component

## Data Flow

**Daily operational cycle:**

```
06:00 UTC ─── SNOTEL pull (past 48h) ──┐
              CAIC pull  (past 48h) ──┤
              HRRR pull  (past 48h) ──┤
                                      ▼
              Quality check + upsert to DuckDB
                                      │
                                      ▼
              Recompute features for affected dates
              (dates within 120h of new data)
                                      │
                                      ▼
              Generate predictions for today (+ tomorrow if HRRR forecast avail)
                                      │
                                      ▼
              Store predictions ── available via /predict and /history
```

**Feature assembly per station-date:**

```
Raw hourly obs ──► rolling windows (5 sizes × 4 variables × 4-6 stats) ──► 90 features
                ──► physics proxies (gradient, hoar, slab, rain, anomaly, season) ──► 8 features
Station metadata ──► elevation, lat/lon, zone_ids ──► context features
                                                                              ────► 98+ feature vector
```

**Training data assembly:**

```
features_matrix (station_id, date, elevation_band, 98+ features)
    JOIN caic_danger ON zone_station_map, date, elevation_band
    WHERE date within training range
    ──► (X, y) pairs for supervised training
```

## Implementation Phasing

### Phase 1A: Ingestion + Features + DuckDB (Foundation)

Capabilities: `data/snotel-ingestion`, `data/caic-labels`, `features/weather-windows`, `features/physics-proxies`

Delivers: Working data pipeline with 98+ features materialized daily. Enables model experimentation in notebooks before the formal training pipeline exists.

Dependencies: None (greenfield).

### Phase 1B: RF Model + Evaluation + MLflow (Baseline)

Capabilities: `models/two-stage-rf`, `prediction/elevation-band`

Delivers: Trained RF with evaluation metrics, SHAP explanations, per-elevation-band predictions. The first model that can answer "what is today's danger level?"

Dependencies: Phase 1A (needs features and labels).

### Phase 1C: LSTM Model + Comparison (Experimental)

Capability: `models/hierarchical-lstm`

Delivers: Trained LSTM with head-to-head comparison against RF on identical test set. Statistical significance testing. The answer to "does deep learning help here?"

Dependencies: Phase 1B (needs RF baseline for comparison, shares data splits).

### Phase 1D: API + Deployment (Serving)

Capability: `api/prediction-service`

Delivers: Dockerized FastAPI service with all endpoints, daily scheduler, auth, rate limiting. The system becomes operational.

Dependencies: Phase 1B (needs at least the RF model to serve). Phase 1C optional — ensemble activates if LSTM is available.

### Phase 1E: Data Expansion (Research-driven)

Delivers: Three-source data pipeline (OAP + API + Kaggle) with continuous 12-season coverage. GeoJSON zone boundaries replacing haversine mapping. Cross-validated labels in the overlap period.

Dependencies: Phase 1A (extends existing ingestion infrastructure).

### Phase 1F: Architecture Upgrade (Research-driven)

Delivers: Per-elevation-band 12-model architecture, frozen Stage-1 predictions, soft-voting ensemble, Transformer Persistent Slab option, duration-weighted physics features. Evaluated against Schwartzreich 2026 benchmark.

Dependencies: Phase 1E (needs expanded dataset) + Phase 1B (refactors existing RF architecture).

### Phase 2: Spatial Downscaling (Micro-weather)

Capabilities: `data/hrrr-ingestion`, `spatial/downscaling`

Delivers: HRRR ingestion with bias correction, XGBoost downscaling model, terrain features, cold air pool handling. Predictions at arbitrary mountain locations, not just station points.

Dependencies: Phase 1A (needs SNOTEL data for bias correction training). HRRR ingestion can begin in parallel with Phase 1B.

### Phase 3: Integration (Terrain-resolved Predictions)

Delivers: Full pipeline where downscaled weather at arbitrary locations flows through the same feature engineering and model prediction path as station data. API supports location-based queries in addition to zone-based.

Dependencies: Phase 1D + Phase 2.

## Risks / Trade-offs

**[CAIC scraper fragility]** → caic-python depends on HTML structure that CAIC can change without notice. → *Mitigation:* Three-source strategy eliminates single-source dependency: OAP covers Dec 2013 - Apr 2021, avalanche.org API covers Nov 2019 - present, Kaggle provides supplementary cross-validation. For dates after Apr 2021, the API is the primary source (not scraping). The ingestion layer isolates any source failure — features still compute, predictions still serve from the last available model.

**[HRRR precipitation bias (25-65%)]** → Raw HRRR precip values produce wrong snow loading estimates. → *Mitigation:* Station-based rolling 30-day correction factor applied at ingestion time. The correction is only as good as station coverage — remote areas with no nearby SNOTEL station get IDW-averaged corrections that may still be wrong. Phase 2 spatial downscaling partially addresses this.

**[Class imbalance (1.1% high-danger)]** → Models will default to predicting "Low" without intervention. → *Mitigation:* Cost-sensitive class weights ({0:1, 1:5} baseline) and serving-time threshold tuning (default t=0.30). SMOTE dropped per research findings — hurts generalization. Focal loss for LSTM. Binary/3-class grouping as primary output reduces the impact. Evaluate with per-class F1, not just accuracy.

**[Temporal autocorrelation causing metric inflation]** → Adjacent days have similar weather and similar labels. → *Mitigation:* Strictly temporal splits. No random cross-validation. Season-boundary gaps between splits (summer months excluded). This reduces effective training data but produces honest metrics.

**[DuckDB single-writer limitation]** → Only one process can write at a time. → *Mitigation:* The daily pipeline runs sequentially (ingest → features → predict), so concurrent writes don't occur in normal operation. If a manual backfill runs during the scheduled pipeline, one will block. This is acceptable for a single-container deployment — flag via /health if it becomes an issue.

**[LSTM may not outperform RF]** → Published avalanche ML research has not shown consistent DL advantage. → *Mitigation:* RF is the production model. LSTM is an experiment. If LSTM loses, the system still ships with a working RF model. The two-phase implementation ensures LSTM never blocks delivery.

**[SNOTEL station sparsity]** → 118 stations across all of Colorado means some zones have poor coverage. → *Mitigation:* Phase 2 spatial downscaling addresses this for weather features. For training labels, the zone-station mapping explicitly handles multi-station zones and flags zones with insufficient station coverage. Predictions for low-coverage zones include a confidence penalty.

## Experimental Results & Lessons Learned

**[Alignment bug in `_transformer_predict_proba`]** → 96.5% of rows were misaligned — the method sorted rows for batched inference then never restored original order before returning probabilities. → *Fix:* Track original index through the sort and unsort before returning. Committed `6c29b97`.

**[Stage 2 GRU underperforms RF]** → Implemented a GRU option for Stage 2 (temporal danger classification) as a one-parameter architecture switch alongside the existing RF path. With default hyperparameters it underperforms RF. → *Status:* Architecture is clean and swappable; needs hyperparameter tuning to be competitive. Committed `906f962`.

**[Per-class threshold tuning — dead end]** → Tuned per-class decision thresholds to try to improve separation between danger classes 2/3/4. → *Result:* No improvement. The model's predicted probabilities don't separate these classes well enough for threshold adjustment to help; the problem is discriminative power, not calibration.

**[Ordinal loss (Frank & Hall) — dead end]** → Tried an ordinal classification loss to exploit the ordered structure of danger levels. → *Result:* Worse than standard classification across all elevation bands (-4.5 to -7.3pp macro-F1). Ordinal loss boosts High-danger recall but tanks precision enough to net negative overall.

**[Kaggle baseline diagnostic — benchmark match confirmed]** → Ran our balanced RF on Schwartzreich's exact Kaggle dataset (8-zone granularity) to isolate whether the benchmark gap was a model or data issue. → *Result:* Our RF hits ATL 0.501 vs their pipeline's 0.508 — within 0.7pp. On comparable data, we match the published benchmark.

**[Schwartzreich gap explained: zone granularity, not model quality]** → On our own 27-zone data we trail the benchmark by a wider margin (ATL 0.397, -0.111 vs 0.508). The diagnostic above shows this gap is not a modeling deficiency. → *Root cause:* 27 zones means ~350 samples/zone vs ~1200 samples/zone at Schwartzreich's 8-zone granularity — a per-zone data sparsity problem, not an architecture or feature problem. → *Implication:* Stop tuning against this benchmark; further gains require more data per zone or a different problem framing, not model changes.

**[Experiment volume]** → 222+ experiments run across RF, ExtraTrees, XGBoost, Transformer, GRU, threshold tuning, and ordinal loss variants, logged to MLflow.

**Next direction:** Given the benchmark gap is explained and closing it further has diminishing returns, the next phase shifts from benchmark-chasing to physics-informed innovation — new proxy features capturing snowpack processes (temperature gradient metamorphism, surface hoar formation, wind loading) that RF/GRU can't infer from raw weather inputs alone, including cross-domain signals and eventual integration with the `terrain-weather-ml` project for higher-resolution micro-weather inputs.

## Open Questions

- **OAP label schema mapping completeness.** The Open Avalanche Project dataset format is documented but the exact field mapping to CAIC's current schema has not been validated. This affects only the fallback path and can be resolved during Phase 1A implementation without changing the design.
- **HRRR archive depth.** AWS HRRR archive availability before 2016 is inconsistent. If the full 2015-2021 training window needs HRRR data, some early dates may have gaps. This affects only Phase 2 and can be assessed empirically during HRRR ingestion implementation.
- **GPU availability for LSTM training.** Phase 1C assumes access to a GPU for reasonable LSTM training times (180-day sequences with multi-rate branches). If training on CPU only, the LSTM comparison may need to use shorter sequences or reduced architecture. This does not block Phases 1A-1D.
