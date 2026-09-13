# Colorado Avalanche ML — Session Handoff

## Current State

Phase 1 complete. 298+ tests, all green. Models trained on 1.3M rows (10 seasons).

### What Works
- Two-stage pipeline: Transformer Stage 1 (persistent slab F1 0.826) + RF Stage 2
- 12-model per-band architecture with frozen Stage 1 predictions
- Full data pipeline: SNOTEL (118 stations) + CAIC labels (OAP + avalanche.org API)
- 90 weather features + 8 physics proxies + consecutive gradient days
- FastAPI service with Docker deployment
- 222+ experiments logged

### What Doesn't Work (Yet)
- GRU Stage 2 underperforms RF with default hyperparams (architecture is clean, needs tuning)
- Threshold tuning: dead end (model can't separate classes 2/3/4)
- Ordinal loss: worse than standard RF across all bands

### The Schwartzreich Gap — Resolved
- On their exact 8-zone data: our RF hits ATL 0.501 vs their pipeline 0.508
- On our 27-zone data: ATL 0.397 — gap is zone granularity (350 vs 1200 samples/zone)
- Stop chasing the benchmark. Our model is sound.

## Next Direction: Physics-Informed Innovation

Cooper's direction: go back to the physics of weather and snow. Find proxies that the current model can't see.

Suggested research areas:
1. **Novel physics proxies** — what physical processes produce avalanches that our features don't capture?
2. **Cross-domain proxy discovery** — seemingly unconnected signals that are predictive
3. **Better snowpack physics encoding** — our temp gradient proxy is validated but simplified
4. **Surface hoar formation** — our index is basic; the real physics involves radiation balance, humidity, wind
5. **Wind loading** — critical for slab formation, limited by no wind data at SNOTEL
6. **Integration with terrain-weather-ml** — when that project delivers sub-km weather, our features get much richer

## Sibling Project: terrain-weather-ml

Repo: github.com/mcande21/terrain-weather-ml
Session: "Terrain" (running in parallel)
State: OpenSpec planning complete, Tier 1 implementation in progress (terrain encoder, Alpine data pipeline, CFD pipeline)
Architecture: StormCast backbone + DEVINE terrain head + LoRA adaptation
When it delivers: sub-km weather at arbitrary terrain locations → feeds into this project's Phase 2/3

## Repos

- colorado-avalanche-ml: github.com/mcande21/colorado-avalanche-ml (~35 commits on master)
- terrain-weather-ml: github.com/mcande21/terrain-weather-ml (Tier 1 building)

## Key Files

- `src/avalanche_ml/models/per_band_pipeline.py` — main pipeline (hybrid Transformer+RF Stage 1, RF Stage 2, GRU Stage 2 option)
- `src/avalanche_ml/features/physics.py` — physics proxy features (START HERE for innovation)
- `src/avalanche_ml/features/weather.py` — weather rolling window features
- `scripts/run_gap_experiments.py` — 204-model experiment battery with checkpoint/resume
- `openspec/changes/snowpack-instability-prediction/` — full spec lifecycle

## Memory

Query `qm memory recall "avalanche" --project colorado-avalanche-ml` for all persisted findings.
