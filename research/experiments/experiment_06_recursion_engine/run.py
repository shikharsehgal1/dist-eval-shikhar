"""Experiment 6: Recursion engine — decomposing STUCK tasks into sub-tasks.

Simulates synthetic tasks with known sub-task structure and compares flat retry,
random decomposition, and recursive decomposition on sub-task solve rate and
parent-task graduation.

This is a simulation stub; the real RecursionEngine can be swapped in once
checkpoint-annotated tasks are available.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))


import numpy as np
import pandas as pd

SEED = 42
N_TASKS = 10
N_RUNS = 100
OUTPUT_DIR = Path(__file__).parent / "results"


def generate_task(task_id: int, rng: np.random.Generator) -> dict:
    """Generate a synthetic task with 2 sub-tasks and known boundaries."""
    n_subtasks = 2
    # Random sub-task success probabilities; parent task solves only if both solve
    sub_probs = [rng.uniform(0.2, 0.8) for _ in range(n_subtasks)]
    parent_prob = np.prod(sub_probs)  # both sub-tasks must succeed
    return {
        "task_id": f"task_{task_id}",
        "n_subtasks": n_subtasks,
        "sub_probs": sub_probs,
        "parent_prob": parent_prob,
    }


def simulate_flat_retry(task: dict, rng: np.random.Generator) -> dict:
    """Flat retry: only the full task is attempted."""
    solved = rng.random() < task["parent_prob"]
    return {
        "task": task["task_id"],
        "strategy": "flat_retry",
        "parent_solved": int(solved),
        "sub_tasks_solved": task["n_subtasks"] if solved else 0,
    }


def simulate_random_decomposition(task: dict, rng: np.random.Generator) -> dict:
    """Random decomposition: train on each sub-task independently with 50% chance."""
    sub_solved = sum(1 for p in task["sub_probs"] if rng.random() < p * 0.5)
    return {
        "task": task["task_id"],
        "strategy": "random_decomposition",
        "parent_solved": int(sub_solved == task["n_subtasks"]),
        "sub_tasks_solved": sub_solved,
    }


def simulate_recursive_decomposition(task: dict, rng: np.random.Generator) -> dict:
    """Recursive decomposition: train on each sub-task independently with full focus."""
    sub_solved = sum(1 for p in task["sub_probs"] if rng.random() < min(1.0, p + 0.2))
    return {
        "task": task["task_id"],
        "strategy": "recursion_engine",
        "parent_solved": int(sub_solved == task["n_subtasks"]),
        "sub_tasks_solved": sub_solved,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    tasks = [generate_task(i, rng) for i in range(N_TASKS)]
    strategies = {
        "flat_retry": simulate_flat_retry,
        "random_decomposition": simulate_random_decomposition,
        "recursion_engine": simulate_recursive_decomposition,
    }

    rows = []
    for _ in range(N_RUNS):
        for task in tasks:
            for strategy_name, sim_fn in strategies.items():
                rows.append(sim_fn(task, rng))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    summary = df.groupby("strategy").agg(
        mean_parent_solved=("parent_solved", "mean"),
        mean_sub_tasks_solved=("sub_tasks_solved", "mean"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({
            "n_tasks": N_TASKS,
            "n_runs": N_RUNS,
            "summary": summary.to_dict(orient="records"),
        }, f, indent=2)

    print("Experiment 6 — Recursion engine (simulation stub)")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
