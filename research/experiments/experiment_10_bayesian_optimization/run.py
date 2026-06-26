"""Experiment 10: Bayesian optimization for DPO hyperparameters.

Validates that disteval.bayesian_optimization can find hyperparameters that
outperform the default (alpha=0.4, dpo_bonus=1.5, k=5) on the training
simulation objective. Also checks that the GP-based surrogate is well-calibrated
by comparing the optimized parameters to the known-best grid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

import numpy as np

from disteval.bayesian_optimization import (
    BayesianOptimizer,
    SearchSpace,
    optimize_dpo_hyperparameters,
)
from disteval.records import EpisodeRecord, RecordStore
from disteval.right_tail import right_tail_analysis
from disteval.training_sim import (
    apply_training_effect,
    select_disteval_right_tail,
)

SEED = 42
OUTPUT_DIR = Path(__file__).parent / "results"


def make_synthetic_agent(n_tasks: int = 9, n_attempts: int = 5, seed: int = SEED) -> RecordStore:
    """Create a synthetic agent with RECOVERABLE tasks amenable to DPO bonus."""
    rng = np.random.default_rng(seed)
    store = RecordStore()
    task_kinds = ["recoverable"] * 5 + ["stuck"] * 3 + ["solid"]
    rng.shuffle(task_kinds)
    for task_id, kind in enumerate(task_kinds, start=1):
        if kind == "solid":
            scores = [1.0] * n_attempts
        elif kind == "recoverable":
            q_star = float(rng.uniform(0.9, 1.0))
            q_bar = float(rng.uniform(0.2, 0.3))
            n_reinforce = max(2, n_attempts // 2)
            n_contrast = n_attempts - n_reinforce
            reinforce = [float(np.clip(rng.normal(q_star, 0.02), 0.9 * q_star, 1.0)) for _ in range(n_reinforce)]
            contrast = [float(np.clip(rng.normal(q_bar, 0.05), 0.0, 0.9 * q_star)) for _ in range(n_contrast)]
            scores = reinforce + contrast
            rng.shuffle(scores)
        else:
            scores = [0.0] * n_attempts
        for attempt, score in enumerate(scores):
            store.add(EpisodeRecord(
                run_id="synthetic", model="synthetic_agent", task=f"task_{task_id}",
                episode=attempt, score=float(score), success=float(score) >= 0.99,
            ))
    return store


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    store = make_synthetic_agent()
    records = store._records
    report = right_tail_analysis(store, model_name="synthetic_agent")

    # Evaluate default hyperparameters
    selected_default = select_disteval_right_tail(records, report, k=5)
    new_scores_default = apply_training_effect(
        records, selected_default, report,
        alpha=0.4, strategy="disteval_right_tail", dpo_bonus=1.5,
    )
    default_gain = float(np.mean(new_scores_default) - np.mean([r.score for r in records]))

    # Bayesian optimization with a custom UCB-based optimizer for broader exploration.
    from disteval.bayesian_optimization import BayesianOptimizer

    space = [
        SearchSpace("alpha", 0.1, 0.8, log_scale=False),
        SearchSpace("dpo_bonus", 0.5, 3.0, log_scale=False),
        SearchSpace("k", 1, 10, integer=True),
    ]
    bo = BayesianOptimizer(space=space, acquisition="ucb", n_init=8, seed=SEED)
    baseline = float(np.mean([r.score for r in records]))

    def objective(x: np.ndarray) -> float:
        alpha, dpo_bonus, k = float(x[0]), float(x[1]), int(round(x[2]))
        k = max(1, min(k, len(records) // 2))
        sel = select_disteval_right_tail(records, report, k)
        new_scores = apply_training_effect(
            records, sel, report,
            alpha=alpha, strategy="disteval_right_tail", dpo_bonus=dpo_bonus,
        )
        return float(np.mean(new_scores) - baseline)

    result = bo.optimize(objective, n_iter=40, maximize=True)
    best = result["best_params"]
    best_gain = result["best_value"]

    # Grid search reference for calibration check
    grid_gains = []
    grid_best = None
    grid_best_gain = -float("inf")
    for alpha in np.linspace(0.1, 0.8, 8):
        for beta in np.linspace(0.5, 3.0, 6):
            for k in range(1, 11):
                sel = select_disteval_right_tail(records, report, k=k)
                new_scores = apply_training_effect(
                    records, sel, report,
                    alpha=float(alpha), strategy="disteval_right_tail", dpo_bonus=float(beta),
                )
                gain = float(np.mean(new_scores) - np.mean([r.score for r in records]))
                grid_gains.append(gain)
                if gain > grid_best_gain:
                    grid_best_gain = gain
                    grid_best = {"alpha": float(alpha), "dpo_bonus": float(beta), "k": int(k)}

    rows = [
        {"method": "default", "alpha": 0.4, "dpo_bonus": 1.5, "k": 5, "gain": default_gain},
        {"method": "bo", "alpha": best["alpha"], "dpo_bonus": best["dpo_bonus"], "k": best["k"], "gain": best_gain},
        {"method": "grid_best", "alpha": grid_best["alpha"], "dpo_bonus": grid_best["dpo_bonus"], "k": grid_best["k"], "gain": grid_best_gain},
    ]
    df = __import__("pandas").DataFrame(rows)
    print("Experiment 10 — Bayesian optimization for DPO hyperparameters")
    print("=" * 70)
    print(df.to_string(index=False))
    print()
    print(f"BO gain / default gain: {best_gain / max(default_gain, 1e-9):.2f}")
    print(f"BO gain / grid-best gain: {best_gain / max(grid_best_gain, 1e-9):.2f}")
    print(f"Grid-best: {grid_best}")
    print(f"BO-best:   {best}")

    (OUTPUT_DIR / "summary.json").write_text(json.dumps({
        "default_gain": default_gain,
        "bo_gain": best_gain,
        "grid_best_gain": grid_best_gain,
        "bo_params": best,
        "grid_best_params": grid_best,
    }, indent=2))


if __name__ == "__main__":
    main()
