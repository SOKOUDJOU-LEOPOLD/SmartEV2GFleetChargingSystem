# Phase 0 — Repository Foundation

**Goal:** a clean, organized repo that anyone (human or AI agent) can clone, set up and contribute to following the same rules.

**Status:** 🟡 In progress

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
- [ ] `.gitignore` (venv, caches, results, data, checkpoints, wandb, .env)
- [ ] `requirements.txt` (pinned runtime) + `requirements-dev.txt` (pytest, ruff, pre-commit)
- [ ] `pyproject.toml` (src layout, ruff + pytest config)
- [ ] `src/smartev2g/` skeleton + `tests/test_smoke.py`
- [ ] `.github/workflows/ci.yml`, PR template, issue template
- [ ] `.pre-commit-config.yaml`
- [ ] Create local `.venv`, install, verify imports, pin versions
- [ ] (User) enable branch protection on `main` in GitHub settings: require PR + green CI

## Exit criteria
- Fresh clone → `python3.12 -m venv .venv && pip install -r requirements.txt -r requirements-dev.txt` → `pytest` and `ruff check .` pass.
- CI green on the PR.
