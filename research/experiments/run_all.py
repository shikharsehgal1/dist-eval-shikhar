"""Run all experiments in the validation programme.

Executes every experiment_*/run.py and collects their summaries into a single
scorecard CSV. Each experiment is run in a subprocess so failures are isolated.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "_all_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    "experiment_01_distribution_metrics",
    "experiment_02_recoverable_training",
    "experiment_03_self_engine_oracle",
    "experiment_04_trajectory_monitor",
    "experiment_05_trajectory_memory",
    "experiment_06_recursion_engine",
    "experiment_07_distributed_eval",
    "experiment_08_agent_harness",
    "experiment_09_training_sim_fidelity",
    "experiment_10_bayesian_optimization",
]


def run_experiment(name: str) -> dict:
    """Run one experiment script and return its summary.json."""
    script = ROOT / name / "run.py"
    print(f"\n{'='*60}")
    print(f"Running {name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR in {name}: {result.stderr}")
        return {"experiment": name, "status": "failed", "error": result.stderr}

    summary_path = ROOT / name / "results" / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}
    summary["experiment"] = name
    summary["status"] = "ok"
    return summary


def main() -> None:
    summaries = []
    for name in EXPERIMENTS:
        summaries.append(run_experiment(name))

    df = pd.json_normalize(summaries)
    df.to_csv(OUTPUT_DIR / "scorecard.csv", index=False)

    with open(OUTPUT_DIR / "scorecard.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n{'='*60}")
    print("All experiments complete")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nScorecard saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
