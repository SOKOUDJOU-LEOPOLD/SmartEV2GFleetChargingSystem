"""CPU vs GPU timing test (Phase 1): which device trains our PPO fastest?

Runs the SAME short training (same config, seed, number of steps) with several
settings and records environment steps per second (training only: no validation,
no test). Also tries different numbers of PyTorch CPU threads, since the network
is tiny and using every core can be slower than using a few.

Usage:
    python scripts/timing_test.py --config configs/experiments/p1_sanity_ppo.yaml --steps 100000

Saved in results/<run_id>/: timing.csv, run_info.json (+ one sub-folder per setting).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch

from smartev2g.agents.train import train
from smartev2g.utils.run_info import RESULTS_DIR, RunInfo, make_run_dir, make_run_id

# (device, torch threads); threads=None = PyTorch default (all cores).
SETTINGS = [("cpu", 1), ("cpu", 4), ("cpu", 16), ("cpu", None), ("cuda", 4)]
NEVER = 10**12  # "never" for validation / checkpoint frequency during the timing runs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args()

    os.environ["WANDB_MODE"] = "disabled"  # timing only; nothing to log online
    run_dir = make_run_dir(make_run_id("timing-test"), root=args.results_dir)
    info = RunInfo(run_dir, "scripts/timing_test.py", args.config, [args.seed], vars(args))

    default_threads = torch.get_num_threads()  # PyTorch's own choice (usually all cores)
    rows = []
    for device, threads in SETTINGS:
        threads = threads or default_threads  # thread setting persists, so always set it
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available: skipping GPU setting")
            continue
        sub = train(
            args.config,
            args.seed,
            results_dir=run_dir,
            overrides={
                "total_timesteps": args.steps,
                "device": device,
                "torch_threads": threads,
                "eval_every_steps": NEVER,
                "checkpoint_every_steps": NEVER,
            },
            run_test=False,
            verbose=False,
        )
        res = json.loads((sub / "run_info.json").read_text())["results"]
        row = {
            "device": device,
            "torch_threads": threads,
            "is_default_threads": threads == default_threads,
            "steps": res["total_timesteps"],
            "train_seconds": res["train_seconds"],
            "steps_per_second": res["steps_per_second"],
        }
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "timing.csv", index=False)
    best = df.loc[df["steps_per_second"].idxmax()]
    info.finish(results={"settings": rows, "fastest": best.to_dict()})
    print(df.to_string(index=False))
    print(f"Fastest: {best['device']} with {best['torch_threads']} threads. Saved: {run_dir}")


if __name__ == "__main__":
    main()
