"""Experiment 9: Training simulation fidelity.

Generates a synthetic agent with known RECOVERABLE tasks, then compares the
training gains predicted by training_sim.py (bootstrap) to the true gains
computed on the full population. Measures rank-order correlation (Spearman)
and normalized MAE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

import numpy as np
import pandas as pd
from scipy import stats

from disteval.records import EpisodeRecord, RecordStore
from disteval.right_tail import right_tail_analysis
from disteval.training_sim import (
    apply_training_effect,
    bootstrap_resample_within_tasks,
    select_disteval_right_tail,
    select_mean_reward,
    select_random,
)

SEED = 42
N_BOOTSTRAP = 1000
OUTPUT_DIR = Path(__file__).parent / "results"


def make_synthetic_agent(n_tasks: int = 9, n_attempts: int = 5, seed: int = SEED) -> RecordStore:
    """Create a synthetic agent where RECOVERABLE tasks dominate and have high leverage.

    5 RECOVERABLE tasks with high q_star (0.9-1.0) and low q_bar (0.2-0.3).
    3 STUCK tasks (all 0.0) to waste capacity for all-task training.
    1 SOLID task at ceiling (no improvement possible).
    """
    rng = np.random.default_rng(seed)
    store = RecordStore()
    task_kinds = ["recoverable"] * 5 + ["stuck"] * 3 + ["solid"]
    rng.shuffle(task_kinds)
    for task_id, kind in enumerate(task_kinds, start=1):
        if kind == "solid":
            q_star = 1.0
            scores = [1.0] * n_attempts
        elif kind == "recoverable":
            q_star = float(rng.uniform(0.9, 1.0))
            q_bar = float(rng.uniform(0.2, 0.3))
            # Ensure both reinforce and contrast examples exist for DPO bonus
            n_reinforce = max(2, n_attempts // 2)
            n_contrast = n_attempts - n_reinforce
            reinforce = [float(np.clip(rng.normal(q_star, 0.02), 0.9 * q_star, 1.0)) for _ in range(n_reinforce)]
            contrast = [float(np.clip(rng.normal(q_bar, 0.05), 0.0, 0.9 * q_star)) for _ in range(n_contrast)]
            scores = reinforce + contrast
            rng.shuffle(scores)
        else:  # stuck
            scores = [0.0] * n_attempts
        for attempt, score in enumerate(scores):
            store.add(EpisodeRecord(
                run_id="synthetic", model="synthetic_agent", task=f"task_{task_id}",
                episode=attempt, score=float(score), success=float(score) >= 0.99,
            ))
    return store


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    store = make_synthetic_agent()
    records = store._records
    report = right_tail_analysis(store, model_name="synthetic_agent")
    strategies = ["disteval_right_tail", "mean_reward", "random"]

    # Compute true (full-population) one-round gain for each strategy
    true_gains = {}
    baseline = np.mean([r.score for r in records])
    k = 5
    for strategy in strategies:
        if strategy == "disteval_right_tail":
            selected = select_disteval_right_tail(records, report, k)
        elif strategy == "mean_reward":
            selected = select_mean_reward(records, k)
        else:
            selected = select_random(records, k, rng)
        new_scores = apply_training_effect(
            records, selected, report,
            strategy="disteval_right_tail" if strategy == "disteval_right_tail" else "generic",
        )
        true_gains[strategy] = float(np.mean(new_scores) - baseline)

    # Bootstrap predictions
    rows = []
    predicted_means = {}
    for strategy in strategies:
        predicted_gains = []
        for _ in range(N_BOOTSTRAP):
            resampled = bootstrap_resample_within_tasks(records, rng)
            resampled_baseline = np.mean([r.score for r in resampled])
            resampled_report = right_tail_analysis(RecordStore(resampled), model_name="synthetic_agent")
            k = 5
            if strategy == "disteval_right_tail":
                selected = select_disteval_right_tail(resampled, resampled_report, k)
            elif strategy == "mean_reward":
                selected = select_mean_reward(resampled, k)
            else:
                selected = select_random(resampled, k, rng)
            new_scores = apply_training_effect(
                resampled, selected, resampled_report,
                strategy="disteval_right_tail" if strategy == "disteval_right_tail" else "generic",
            )
            predicted_gains.append(float(np.mean(new_scores) - resampled_baseline))
        predicted_means[strategy] = float(np.mean(predicted_gains))

    for strategy in strategies:
        rows.append({
            "strategy": strategy,
            "predicted_gain": predicted_means[strategy],
            "true_gain": true_gains[strategy],
            "predicted_gain_per_example": predicted_means[strategy] / k,
            "true_gain_per_example": true_gains[strategy] / k,
            "abs_error": abs(predicted_means[strategy] - true_gains[strategy]),
            "abs_error_per_example": abs(predicted_means[strategy] - true_gains[strategy]) / k,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    pred = df["predicted_gain_per_example"].to_numpy()
    true = df["true_gain_per_example"].to_numpy()
    if len(set(pred)) > 1 and len(set(true)) > 1:
        rho, p = stats.spearmanr(pred, true)
    else:
        rho, p = np.nan, np.nan

    absolute_mae = float(df["abs_error_per_example"].mean())
    max_true = df["true_gain_per_example"].max()
    normalized_mae = float((absolute_mae / max_true) if max_true > 0 else 0.0)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({
            "n_bootstrap": N_BOOTSTRAP,
            "spearman_rho": float(rho),
            "p_value": float(p),
            "absolute_mae": absolute_mae,
            "normalized_mae": normalized_mae,
        }, f, indent=2)

    print("Experiment 9 — Training simulation fidelity")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nSpearman ρ: {rho:.3f}")
    print(f"Absolute MAE (per-example): {absolute_mae:.4f}")
    print(f"Normalized MAE (per-example): {normalized_mae:.3f}")
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
