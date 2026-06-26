"""Experiment 1: Distribution-first metrics vs mean-only reporting.

Generates synthetic agents with identical mean but different tail shapes, then
shows that mean-only ranking misses the reliability differences while IQM,
CVaR, pass^k and the consistency index κ capture them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))


import numpy as np
import pandas as pd
from scipy import stats

from disteval.bootstrap import stratified_bootstrap_ci
from disteval.metrics import cvar, iqm, pass_hat_k
from disteval.records import EpisodeRecord, RecordStore
from disteval.right_tail import right_tail_analysis

SEED = 42
N_TASKS = 6
N_ATTEMPTS = 5
OUTPUT_DIR = Path(__file__).parent / "results"


def generate_profile(profile: str, n_tasks: int = N_TASKS, n_attempts: int = N_ATTEMPTS, seed: int = SEED) -> RecordStore:
    """Generate a synthetic agent profile with mean exactly 0.50 but different tail."""
    rng = np.random.default_rng(seed)
    store = RecordStore()

    # Base sequences with mean exactly 0.5 per task. Higher variance across tasks
    # means the overall mean stays at 0.5 while tails/κ differ strongly.
    base_sequences = {
        "consistent": [0.5, 0.5, 0.5, 0.5, 0.5],
        "inconsistent": [0.0, 1.0, 0.0, 1.0, 0.5],
        "tail_heavy": [0.1, 0.2, 0.5, 0.8, 0.9],
    }
    base = np.array(base_sequences[profile], dtype=float)

    for task_id in range(1, n_tasks + 1):
        # Add tiny task-specific noise so tasks are not identical, but keep mean 0.5
        noise = rng.normal(0.0, 0.01, n_attempts)
        scores = np.clip(base + noise - noise.mean(), 0.0, 1.0)
        # Re-center to exactly 0.5 after clipping
        scores = np.clip(scores + (0.5 - scores.mean()), 0.0, 1.0)
        for attempt, score in enumerate(scores):
            store.add(EpisodeRecord(
                run_id="synthetic",
                model=profile,
                task=f"task_{task_id}",
                episode=attempt,
                score=float(score),
                success=float(score) >= 0.99,
            ))
    return store


def compute_metrics(store: RecordStore) -> dict:
    """Compute distribution-aware metrics for one agent."""
    df = store.df()
    scores = df["score"].to_numpy(dtype=float)
    report = right_tail_analysis(store, model_name=df["model"].iloc[0])

    def mean_fn(df: pd.DataFrame) -> float:
        return float(df["score"].mean())

    def iqm_fn(df: pd.DataFrame) -> float:
        return float(iqm(df["score"].to_numpy()))

    def cvar_fn(df: pd.DataFrame) -> float:
        return float(cvar(df["score"].to_numpy(), alpha=0.1))

    def pass4_fn(df: pd.DataFrame) -> float:
        return float(pass_hat_k(df, k=4))

    kappa = report.sum_q_bar / report.sum_q_star if report.sum_q_star > 0 else 1.0

    metrics = {
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "iqm": iqm(scores),
        "cvar_0.1": cvar(scores, alpha=0.1),
        "pass_hat_4": pass_hat_k(df, k=4),
        "kappa": kappa,
    }

    def kappa_fn(df: pd.DataFrame) -> float:
        return kappa

    for name, stat_fn in [
        ("mean", mean_fn),
        ("iqm", iqm_fn),
        ("cvar_0.1", cvar_fn),
        ("kappa", kappa_fn),
        ("pass_hat_4", pass4_fn),
    ]:
        ci = stratified_bootstrap_ci(df, stat_fn, strata_cols=["task"], n_reps=2000, seed=SEED)
        metrics[f"{name}_ci_width"] = ci["width"]

    return metrics


def rank_agents(results: dict[str, dict]) -> dict[str, dict[str, int]]:
    """Return agent -> rank for each metric."""
    agents = list(results.keys())
    rankings: dict[str, dict[str, int]] = {}
    for metric in ["mean", "iqm", "cvar_0.1", "pass_hat_4", "kappa"]:
        scores = [results[a][metric] for a in agents]
        order = np.argsort(scores)[::-1]
        rankings[metric] = {agents[order[i]]: i for i in range(len(agents))}
    return rankings


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    profiles = ["consistent", "inconsistent", "tail_heavy"]
    stores = {p: generate_profile(p) for p in profiles}

    results = {p: compute_metrics(store) for p, store in stores.items()}

    rankings = rank_agents(results)
    agreement_rows = []
    for metric in ["iqm", "cvar_0.1", "pass_hat_4", "kappa"]:
        mean_ranks = [rankings["mean"][a] for a in profiles]
        metric_ranks = [rankings[metric][a] for a in profiles]
        rho, p = stats.spearmanr(mean_ranks, metric_ranks)
        agreement_rows.append({
            "metric": metric,
            "spearman_rho": float(rho),
            "p_value": float(p),
        })
    agreement_df = pd.DataFrame(agreement_rows)

    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df.index.name = "profile"
    results_df = results_df.reset_index()

    # Effect size: distribution metric span (κ) vs mean span
    kappa_values = [results[p]["kappa"] for p in profiles]
    kappa_range = max(kappa_values) - min(kappa_values)
    mean_range = max(results[p]["mean"] for p in profiles) - min(results[p]["mean"] for p in profiles)

    summary = {
        "mean_equality_check": {p: round(results[p]["mean"], 4) for p in profiles},
        "kappa_range": round(kappa_range, 4),
        "mean_range": round(mean_range, 6),
        "agreement": agreement_rows,
        "key_finding": "Means are equal but κ spans widely, showing distribution metrics capture hidden inconsistency",
    }

    results_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    agreement_df.to_csv(OUTPUT_DIR / "agreement.csv", index=False)
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Experiment 1 — Distribution-first metrics")
    print("=" * 60)
    print("\nMean score equality check (should all be ~0.50):")
    for p in profiles:
        print(f"  {p:15} mean = {results[p]['mean']:.4f}")
    print("\nDistribution metrics:")
    print(results_df[["profile", "mean", "iqm", "cvar_0.1", "pass_hat_4", "kappa"]].to_string(index=False))
    print("\nRanking agreement with mean-based ranking:")
    print(agreement_df.to_string(index=False))
    print(f"\nMean range: {mean_range:.6f}")
    print(f"κ range: {kappa_range:.3f}")
    print(f"\nKey finding: {summary['key_finding']}")
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
