# Phase 6 — Stretch: Data-Driven Departure Model (ACN-Data)

**Goal:** replace the made-up early-departure distribution with one estimated from real data.

**Status:** ⚪ Blocked by the Deliverable 1 gate

## Branch
`<name>/feat/p6-acn-data`

## Tasks
- [ ] Get a free ACN-Data API token (https://ev.caltech.edu/dataset); store it in `.env` (never committed)
- [ ] `scripts/download_acn.py` → raw data in `data/acn/` (gitignored); document in `data/README.md`
- [ ] Clean sessions with user-requested departure; compute `ΔT = T_requested − T_actual`
- [ ] Estimate: overall early-departure probability, distribution of Δ, `p_early(t)` by hour of day, day-to-day variation of the early-departure rate
- [ ] `ACNDepartureModel` in `departure_models.py` (same interface as synthetic)
- [ ] Map ACN time scale to EV2Gym timesteps; document in an ADR
- [ ] Re-run key Phase 4 comparison (PPO-Trusting vs PPO-RiskAware vs Oracle) with the empirical model
- [ ] Figures: empirical ΔT histogram, p_early(t) curve

## Exit criteria
- Empirical model integrated and tested; key comparison reproduced.
