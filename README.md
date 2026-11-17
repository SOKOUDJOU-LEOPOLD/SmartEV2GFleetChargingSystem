# Smart EV2G Fleet Charging System

**Risk-Aware Deep Reinforcement Learning for EV Charging Under Unexpected Departures**
Yeshas Krishna · Leopold Sokoudjou Gatsing

## Research question
Can a model-free RL agent optimize V2G fleet scheduling under partial observability, implicitly managing the risk of unannounced early departures, and what economic or grid-performance trade-offs emerge from this robust policy?

In plain words: drivers announce when they will leave a charging site, but some leave **earlier** without warning. We train a PPO controller on the [EV2Gym](https://github.com/StavrosOrf/EV2Gym) simulator and check whether it learns, by itself, to keep a safety reserve of charge — and what that costs in profit and grid flexibility.

| Agent | Role |
|---|---|
| **PPO-RiskAware** | **Ours** — trained with hidden early departures |
| PPO-Trusting | Same agent, trained assuming announced times are always right |
| PPO-Oracle | Can see true departure times (upper bound) |
| AFAP | Rule: charge as fast as possible |

## Start here
1. [`CODING_RULES.md`](CODING_RULES.md) — rules for every contributor (human or AI agent)
2. [`docs/plan/ROADMAP.md`](docs/plan/ROADMAP.md) — master plan and phase status
3. [`CONTRIBUTING.md`](CONTRIBUTING.md) — setup and day-to-day git workflow

## Documentation map
| Folder / file | Content |
|---|---|
| `docs/proposal/` | Project proposal and literature review |
| `docs/plan/` | Roadmap, one plan per phase, metrics & logging spec |
| `docs/concepts/RL_CONCEPTS.md` | Plain-language RL glossary with references |
| `docs/experiments/EXPERIMENT_LOG.md` | One line per experiment run |
| `docs/decisions/` | Architecture/decision records (why we chose X) |

## Quick setup
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate      # or: python3.12 -m venv .venv
uv pip install -r requirements.txt -r requirements-dev.txt && uv pip install -e .
pytest
```
