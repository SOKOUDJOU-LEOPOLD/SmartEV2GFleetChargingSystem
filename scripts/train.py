"""Train one PPO agent (one experiment config + one seed).

Usage:
    python scripts/train.py --config configs/experiments/p1_sanity_ppo.yaml --seed 0
    # long runs: detach so they survive a closed terminal (CODING_RULES §7)
    setsid nohup python scripts/train.py --config ... --seed 0 > results/logs/<name>.log 2>&1 &
    # continue a run that stopped (uses its latest checkpoint)
    python scripts/train.py --config ... --seed 0 --resume results/<run_id>

Outputs go to results/<run_id>/ (see docs/plan/METRICS_AND_LOGGING.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from smartev2g.agents.train import train
from smartev2g.utils.run_info import RESULTS_DIR


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="experiment config (configs/experiments/)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--resume", type=Path, help="results/<run_id> folder to continue")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--total-timesteps", type=int, help="override training.total_timesteps")
    p.add_argument("--n-envs", type=int, help="override training.n_envs")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], help="override training.device")
    p.add_argument("--no-test", action="store_true", help="skip the final test evaluation")
    args = p.parse_args()
    if args.seed < 0:
        p.error("--seed must be >= 0")

    overrides = {
        key: value
        for key, value in {
            "total_timesteps": args.total_timesteps,
            "n_envs": args.n_envs,
            "device": args.device,
        }.items()
        if value is not None
    }
    run_dir = train(
        args.config,
        args.seed,
        results_dir=args.results_dir,
        resume_dir=args.resume,
        overrides=overrides,
        run_test=not args.no_test,
    )
    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
