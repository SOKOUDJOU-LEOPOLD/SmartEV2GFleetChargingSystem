# Contributing

Read [`CODING_RULES.md`](CODING_RULES.md) first — it is the full rulebook. This page is the quick workflow.

## Setup (once)
```bash
git clone https://github.com/SOKOUDJOU-LEOPOLD/SmartEV2GFleetChargingSystem.git
cd SmartEV2GFleetChargingSystem
# Recommended: uv (fast, no sudo needed; install: https://docs.astral.sh/uv/)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt -r requirements-dev.txt
uv pip install -e .
# Alternative without uv (needs the python3.12-venv system package):
#   python3.12 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt -r requirements-dev.txt && pip install -e .
pre-commit install
wandb login            # only needed for real training runs
```

## Every task
```bash
git checkout main && git pull
git checkout -b <firstname>/<type>/<phase>-<short-name>   # e.g. yeshas/feat/p2-departure-wrapper
# ... work, with small commits:  git commit -m "feat(wrapper): ..."
ruff format . && ruff check . && pytest
git fetch origin && git rebase origin/main
git push -u origin <branch>
# open a Pull Request into main, fill in the template, ask your teammate to review
```

After merge: tick the tasks in `docs/plan/phase-*.md` (if not done in the PR) and delete the branch.
