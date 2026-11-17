# Phase 0 — Repository Foundation

**Goal:** a clean, organized repo that anyone (human or AI agent) can clone, set up and contribute to following the same rules.

**Status:** 🟢 Done (waiting for PR review/merge + branch protection)

## Branches
| Branch | Content |
|---|---|
| `leopold/docs/p0-plans-and-rules` | Roadmap, phase plans, coding rules, contributing guide, RL concepts, experiment log, ADR template, move proposal docs to `docs/proposal/`, README |
| `leopold/chore/p0-repo-setup` | `.gitignore`, `requirements*.txt`, `pyproject.toml`, `src/smartev2g/` package skeleton, smoke test, CI workflow, PR/issue templates, pre-commit |

## Tasks
- [x] Move proposal and literature review into `docs/proposal/`
- [x] `docs/plan/ROADMAP.md` + one file per phase
- [x] `CODING_RULES.md`, `CONTRIBUTING.md`, `CLAUDE.md`, `AGENTS.md`
- [x] `docs/concepts/RL_CONCEPTS.md`
- [x] `docs/plan/METRICS_AND_LOGGING.md`
- [x] `docs/experiments/EXPERIMENT_LOG.md` template
- [x] `docs/decisions/` with ADR template + ADR-0001
- [x] `.gitignore` (venv, caches, results, data, checkpoints, wandb, .env)
- [x] `requirements.txt` (pinned runtime) + `requirements-dev.txt` (pytest, ruff, pre-commit)
- [x] `pyproject.toml` (src layout, ruff + pytest config)
- [x] `src/smartev2g/` skeleton + `tests/test_smoke.py`
- [x] `.github/workflows/ci.yml`, PR template, issue template
- [x] `.pre-commit-config.yaml`
- [x] Create local `.venv`, install, verify imports, pin versions
- [ ] (User) enable branch protection on `main` in GitHub settings: require PR + green CI

## Exit criteria
- Fresh clone → `uv venv --python 3.12 .venv && uv pip install -r requirements.txt -r requirements-dev.txt && uv pip install -e .` → `pytest` and `ruff check .` pass. ✅ verified 2026-11-17 (11 tests pass in a fresh venv)
- CI green on the PR.

## Notes from setup (2026-11-17)
- Lab machine has no `python3.12-venv` system package, so `python -m venv` fails; we use **uv** (`~/.local/bin/uv`) instead.
- EV2Gym 2.0.0 imports packages it does not declare (`pandapower`, `numba`, `multicopula`, `psutil`, `pillow`) and still uses `pkg_resources` → pinned explicitly in `requirements.txt` with `setuptools<81`.
- CUDA available (torch 2.15.0, 4× H100). One random 112-step EV2Gym episode takes ~0.03 s on CPU.
