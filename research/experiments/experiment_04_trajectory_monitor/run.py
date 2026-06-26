"""Experiment 4: Trajectory monitor predictive accuracy.

Trains a simple logistic-regression structural predictor on synthetic
trajectories and compares it to majority-class, first-step, and random
baselines via leave-one-out cross-validation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))


import numpy as np
import pandas as pd

from disteval.trajectory_monitor import TrajectoryFeaturizer, TrajectoryFeatures

SEED = 42
N_TRAJECTORIES = 200
OUTPUT_DIR = Path(__file__).parent / "results"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out


def generate_synthetic_trajectories(n: int = N_TRAJECTORIES, seed: int = SEED) -> tuple[list, list[float]]:
    """Generate synthetic trajectories with tool-call steps and binary outcomes."""
    rng = np.random.default_rng(seed)
    trajectories = []
    scores = []
    for _ in range(n):
        n_steps = int(rng.integers(5, 30))
        tools = rng.choice(["read_file", "write_file", "run_shell_command", "search_tool"], size=n_steps)
        steps = [{"tool_calls": [{"function_name": t}]} for t in tools]
        first_write = next((i for i, t in enumerate(tools) if t == "write_file"), n_steps)
        search_ratio = (tools == "search_tool").sum() / n_steps
        score = 1.0 if first_write < 5 and search_ratio < 0.2 else 0.0
        trajectories.append(steps)
        scores.append(score)
    return trajectories, scores


def featurize(trajectories: list) -> np.ndarray:
    """Extract simple structural features from trajectories."""
    featurizer = TrajectoryFeaturizer()
    rows = []
    for steps in trajectories:
        feat: TrajectoryFeatures = featurizer.featurize(steps)
        rows.append([
            np.log1p(feat.first_write_pos),
            np.log1p(feat.first_exec_pos),
            feat.n_exec,
            feat.n_search,
            feat.search_ratio,
            feat.n_writes,
            feat.n_reads,
            float(feat.write_before_read),
            feat.tool_diversity,
            feat.n_steps,
        ])
    return np.array(rows, dtype=float)


def fit_logistic(X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_iter: int = 100) -> tuple[np.ndarray, float]:
    """Simple gradient-descent logistic regression."""
    n_samples, n_features = X.shape
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xs = (X - mean) / std
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    for _ in range(n_iter):
        z = Xs @ weights + bias
        p = _sigmoid(z)
        error = p - y
        weights -= lr * (Xs.T @ error) / n_samples
        bias -= lr * error.mean()
    return weights, bias, mean, std


def predict_logistic(x: np.ndarray, weights: np.ndarray, bias: float, mean: np.ndarray, std: np.ndarray) -> float:
    xs = (x - mean) / std
    return float(_sigmoid(np.array([xs @ weights + bias]))[0])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    trajectories, scores = generate_synthetic_trajectories()
    X = featurize(trajectories)
    y = np.array(scores, dtype=float)

    n = len(y)
    monitor_acc = []
    majority_acc = []
    first_step_acc = []
    random_acc = []

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[[i]], y[[i]]

        weights, bias, mean, std = fit_logistic(X_train, y_train)
        pred = 1.0 if predict_logistic(X_test[0], weights, bias, mean, std) >= 0.5 else 0.0
        monitor_acc.append(float(pred == y_test[0]))

        majority = 1.0 if y_train.mean() >= 0.5 else 0.0
        majority_acc.append(float(majority == y_test[0]))

        steps = trajectories[i]
        if steps and steps[0].get("tool_calls"):
            first_tool = steps[0]["tool_calls"][0].get("function_name", "")
            first_pred = 1.0 if first_tool in {"write_file", "run_shell_command"} else 0.0
        else:
            first_pred = 0.0
        first_step_acc.append(float(first_pred == y_test[0]))

        random_acc.append(float((1.0 if rng.random() < 0.5 else 0.0) == y_test[0]))

    results = pd.DataFrame({
        "method": ["monitor", "majority", "first_step", "random"],
        "loo_accuracy": [
            np.mean(monitor_acc),
            np.mean(majority_acc),
            np.mean(first_step_acc),
            np.mean(random_acc),
        ],
    })
    results.to_csv(OUTPUT_DIR / "results.csv", index=False)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({
            "n_trajectories": n,
            "monitor_accuracy": float(np.mean(monitor_acc)),
            "majority_accuracy": float(np.mean(majority_acc)),
            "first_step_accuracy": float(np.mean(first_step_acc)),
            "random_accuracy": float(np.mean(random_acc)),
        }, f, indent=2)

    print("Experiment 4 — Trajectory monitor predictive accuracy")
    print("=" * 60)
    print(results.to_string(index=False))
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
