"""Experiment 7: Distributed evaluation reduces meta-distribution noise.

Generates synthetic agents with known true distributions, then compares
single-run bootstrap CIs to multi-run meta-distribution CIs and measures the
underconfidence ratio.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))


import numpy as np
import pandas as pd

from disteval.bootstrap import stratified_bootstrap_ci
from disteval.records import EpisodeRecord, RecordStore
from disteval.repeat import is_gap_real, meta_distribution

SEED = 42
N_RUNS = 10
N_TASKS = 6
N_ATTEMPTS = 3
OUTPUT_DIR = Path(__file__).parent / "results"


def generate_agent_runs(agent_name: str, mean_score: float, std_score: float, n_runs: int = N_RUNS, seed: int = SEED) -> list[RecordStore]:
    """Generate N independent evaluation runs for one synthetic agent."""
    rng = np.random.default_rng(seed)
    stores = []
    for run_idx in range(n_runs):
        store = RecordStore()
        for task_id in range(N_TASKS):
            task_name = f"task_{task_id}"
            for attempt in range(N_ATTEMPTS):
                score = float(np.clip(rng.normal(mean_score, std_score), 0.0, 1.0))
                store.add(EpisodeRecord(
                    run_id=f"{agent_name}_run_{run_idx}",
                    model=agent_name,
                    task=task_name,
                    episode=attempt,
                    score=score,
                    success=score >= 0.99,
                ))
        stores.append(store)
    return stores


def mean_stat(df: pd.DataFrame) -> float:
    return float(df["score"].mean())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    agent_a = generate_agent_runs("agent_A", mean_score=0.75, std_score=0.10, seed=SEED)
    agent_b = generate_agent_runs("agent_B", mean_score=0.70, std_score=0.10, seed=SEED + 1)
    agent_c = generate_agent_runs("agent_C", mean_score=0.75, std_score=0.10, seed=SEED + 2)

    rows = []

    # Single-agent single-run bootstrap CI
    for agent_name, stores in [("A", agent_a), ("B", agent_b), ("C", agent_c)]:
        ci = stratified_bootstrap_ci(stores[0].df(), mean_stat, strata_cols=["task"], n_reps=2000, seed=SEED)
        rows.append({
            "agent": agent_name,
            "baseline": "single_agent_single_run",
            "ci_width": ci["width"],
            "meta_ci_width": None,
            "underconfidence_ratio": None,
        })

    # Single-agent multi-run meta CI
    for agent_name, stores in [("A", agent_a), ("B", agent_b), ("C", agent_c)]:
        meta = meta_distribution(stores, mean_stat, ci=0.95)
        boot_widths = [
            stratified_bootstrap_ci(s.df(), mean_stat, strata_cols=["task"], n_reps=2000, seed=i)["width"]
            for i, s in enumerate(stores)
        ]
        mean_boot_width = float(np.mean(boot_widths))
        ratio = meta["ci_width"] / mean_boot_width if mean_boot_width > 0 else float("inf")
        rows.append({
            "agent": agent_name,
            "baseline": "single_agent_multi_run",
            "ci_width": None,
            "meta_ci_width": meta["ci_width"],
            "underconfidence_ratio": ratio,
        })

    # Multi-agent multi-run (full distributed)
    all_stores = agent_a + agent_b + agent_c
    meta = meta_distribution(all_stores, mean_stat, ci=0.95)
    boot_widths = [
        stratified_bootstrap_ci(s.df(), mean_stat, strata_cols=["task"], n_reps=2000, seed=i)["width"]
        for i, s in enumerate(all_stores)
    ]
    mean_boot_width = float(np.mean(boot_widths))
    ratio = meta["ci_width"] / mean_boot_width if mean_boot_width > 0 else float("inf")
    rows.append({
        "agent": "all",
        "baseline": "multi_agent_multi_run",
        "ci_width": None,
        "meta_ci_width": meta["ci_width"],
        "underconfidence_ratio": ratio,
    })

    # P(A > B) single-run vs multi-run
    p_single = is_gap_real([agent_a[0]], [agent_b[0]], mean_stat)["P(A>B on a fresh re-run)"]
    p_multi = is_gap_real(agent_a, agent_b, mean_stat)["P(A>B on a fresh re-run)"]

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({
            "p_a_beats_b_single_run": p_single,
            "p_a_beats_b_multi_run": p_multi,
            "true_gap": 0.05,
            "rows": rows,
        }, f, indent=2)

    print("Experiment 7 — Distributed evaluation")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nP(A > B) single-run: {p_single:.3f}")
    print(f"P(A > B) multi-run:  {p_multi:.3f}")
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
