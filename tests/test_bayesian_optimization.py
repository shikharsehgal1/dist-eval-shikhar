"""Tests for disteval.bayesian_optimization."""
from __future__ import annotations

import numpy as np

from disteval.bayesian_optimization import BayesianOptimizer, SearchSpace, ThompsonSamplingScheduler


def test_bayesian_optimizer_finds_quadratic_maximum():
    """BO should find the maximum of a simple 1D quadratic."""

    def objective(x: np.ndarray) -> float:
        # Parabola with maximum at x=0.75
        return -((float(x[0]) - 0.75) ** 2)

    space = [SearchSpace("x", 0.0, 1.0)]
    optimizer = BayesianOptimizer(space=space, acquisition="ei", n_init=3, seed=42)
    result = optimizer.optimize(objective, n_iter=15, maximize=True)

    assert len(optimizer.history) == 15
    assert result["best_value"] >= -0.05
    assert 0.65 <= result["best_params"]["x"] <= 0.85


def test_bayesian_optimizer_minimizes():
    """BO should minimize when asked."""
    def objective(x: np.ndarray) -> float:
        return (float(x[0]) - 0.2) ** 2

    space = [SearchSpace("x", 0.0, 1.0)]
    optimizer = BayesianOptimizer(space=space, acquisition="ucb", n_init=3, seed=42)
    result = optimizer.optimize(objective, n_iter=15, maximize=False)

    assert result["best_value"] <= 0.05
    assert 0.1 <= result["best_params"]["x"] <= 0.3


def test_bayesian_optimizer_with_integer_dimension():
    """Integer dimensions should be rounded correctly."""
    def objective(x: np.ndarray) -> float:
        # Best integer near 7
        return -abs(int(round(x[0])) - 7)

    space = [SearchSpace("k", 1, 10, integer=True)]
    optimizer = BayesianOptimizer(space=space, acquisition="ei", n_init=2, seed=42)
    result = optimizer.optimize(objective, n_iter=10, maximize=True)

    assert result["best_params"]["k"] == int(round(result["best_params"]["k"]))
    assert 6 <= result["best_params"]["k"] <= 8


def test_bayesian_optimizer_history_consistency():
    """History should contain the same number of entries as iterations."""
    def objective(x: np.ndarray) -> float:
        return float(x[0])

    space = [SearchSpace("x", 0.0, 1.0)]
    optimizer = BayesianOptimizer(space=space, acquisition="ei", n_init=2, seed=42)
    result = optimizer.optimize(objective, n_iter=8, maximize=True)

    assert len(result["history"]) == 8
    for entry in result["history"]:
        assert "x" in entry["params"]
        assert isinstance(entry["value"], float)


def test_thompson_sampling_updates_posterior():
    """After observing a positive reward, posterior mean should shift toward feature direction."""
    scheduler = ThompsonSamplingScheduler(feature_dim=3, lambda_prior=1.0, sigma_noise=0.1, seed=42)
    x = np.array([1.0, 0.0, 0.0])
    scheduler.update(x, reward=1.0)
    assert scheduler.mu[0] > 0.0


def test_thompson_sampling_selects_highest_reward_task():
    """With a deterministic posterior mean, select the task with highest dot product."""
    scheduler = ThompsonSamplingScheduler(feature_dim=2, lambda_prior=1e-6, sigma_noise=1e-6, seed=42)
    scheduler.mu = np.array([1.0, -1.0])
    scheduler.Sigma = np.eye(2) * 1e-6
    scheduler.Sigma_inv = np.eye(2) * 1e6
    task_features = {
        "a": np.array([0.5, 0.5]),
        "b": np.array([0.9, 0.1]),
    }
    selected = scheduler.select(task_features)
    assert selected == "b"
