"""Item Response Theory (IRT) for efficient, discriminating benchmarks.

A benchmark score treats every task as equally informative. IRT does not: it
fits, for each task, a *difficulty* (where on the ability scale it starts to
fail agents) and a *discrimination* (how sharply it separates strong from weak
agents), plus an *ability* for each agent. Two payoffs:

  1. Diagnostics — which tasks actually discriminate between top models (high
     `a`) versus which are noise or ceiling (near-zero `a`, or saturated `b`).
  2. Adaptive / efficient eval — `select_items` picks the most *informative*
     subset (Fisher information), so you can match a full-bank ability estimate
     with far fewer tasks. Complements bootstrap.confidence_sequence (when to
     stop) with which items to run.

The model is the 2-parameter logistic (2PL):

    P(correct | theta, a, b) = sigmoid(a * (theta - b))

fit by joint (respondent + item) penalized maximum likelihood. Joint MLE is
simple and adequate for the modest matrices produced by multi-run agent evals;
the priors (theta ~ N(0,1), b ~ N(0, prior_sd), log a ~ N(0, prior_sd)) keep
degenerate items (all-pass / all-fail) finite. Abilities are standardized to
mean 0, sd 1 after fitting for identifiability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


__all__ = [
    "fit_2pl",
    "item_information",
    "total_information",
    "select_items",
    "responses_from_frame",
]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def responses_from_frame(df: pd.DataFrame) -> tuple[np.ndarray, list, list]:
    """Build a (respondent x item) 0/1 response matrix from long-format records.

    Expects columns ``model`` (respondent), ``task`` (item), and ``success``.
    When a (model, task) pair has multiple episodes, their mean success is
    thresholded at 0.5. Returns ``(matrix, models, tasks)`` where ``matrix`` is
    NaN where a model never attempted a task.
    """
    for col in ("model", "task", "success"):
        if col not in df.columns:
            raise ValueError("DataFrame must have 'model', 'task' and 'success' columns")
    pivot = df.pivot_table(index="model", columns="task", values="success", aggfunc="mean")
    models = list(pivot.index)
    tasks = list(pivot.columns)
    mat = np.where(np.isnan(pivot.to_numpy(dtype=float)), np.nan, (pivot.to_numpy(dtype=float) >= 0.5).astype(float))
    return mat, models, tasks


def fit_2pl(
    responses: np.ndarray,
    model: str = "2pl",
    prior_sd: float = 2.0,
    max_iter: int = 500,
) -> dict:
    """Fit a 2PL (or 1PL/Rasch) IRT model by penalized joint maximum likelihood.

    responses: a (n_respondents x n_items) array of 0/1 (NaN = not attempted).
    model: "2pl" (fit per-item discrimination) or "1pl"/Rasch (all discriminations
        constrained equal). Note: because abilities are standardized to sd 1 after
        fitting, the reported 1PL discrimination is that common value on the
        standardized ability scale, not literally 1.0 (P(correct) is unchanged).

    Returns a dict with ``difficulty`` (b, per item), ``discrimination`` (a, per
    item), ``ability`` (theta, per respondent, standardized), ``n_respondents``,
    ``n_items`` and the fitted negative log-likelihood ``nll``.
    """
    X = np.asarray(responses, dtype=float)
    if X.ndim != 2:
        raise ValueError("responses must be a 2D (respondents x items) array")
    R, n_items = X.shape
    if R < 2 or n_items < 1:
        raise ValueError("need at least 2 respondents and 1 item")
    observed = ~np.isnan(X)
    two_pl = model.lower() == "2pl"
    if model.lower() not in ("2pl", "1pl"):
        raise ValueError(f"model must be '2pl' or '1pl', got {model!r}")

    # Parameter vector: [theta (R), b (n_items), log_a (n_items, only if 2PL)].
    def unpack(p):
        theta = p[:R]
        b = p[R:R + n_items]
        log_a = p[R + n_items:] if two_pl else np.zeros(n_items)
        a = np.exp(log_a)
        return theta, a, b

    def neg_log_post(p):
        theta, a, b = unpack(p)
        z = a[None, :] * (theta[:, None] - b[None, :])
        pr = _sigmoid(z)
        pr = np.clip(pr, 1e-9, 1 - 1e-9)
        ll = np.where(observed, X * np.log(pr) + (1 - X) * np.log(1 - pr), 0.0).sum()
        # Gaussian priors for identifiability / degenerate-item stability.
        pen = 0.5 * np.sum(theta**2)
        pen += 0.5 * np.sum(b**2) / prior_sd**2
        if two_pl:
            pen += 0.5 * np.sum(np.log(a) ** 2) / prior_sd**2
        return -(ll) + pen

    n_params = R + n_items + (n_items if two_pl else 0)
    x0 = np.zeros(n_params)
    # Warm start: difficulty from item pass-rate, ability from respondent pass-rate.
    item_rate = np.nanmean(np.where(observed, X, np.nan), axis=0)
    resp_rate = np.nanmean(np.where(observed, X, np.nan), axis=1)
    item_rate = np.clip(np.nan_to_num(item_rate, nan=0.5), 0.02, 0.98)
    resp_rate = np.clip(np.nan_to_num(resp_rate, nan=0.5), 0.02, 0.98)
    x0[:R] = np.log(resp_rate / (1 - resp_rate))
    x0[R:R + n_items] = -np.log(item_rate / (1 - item_rate))

    res = minimize(neg_log_post, x0, method="L-BFGS-B", options={"maxiter": max_iter})
    theta, a, b = unpack(res.x)

    # Standardize abilities to mean 0, sd 1; rescale item params so P is unchanged:
    #   a*(theta - b) = (a*s) * (theta' - (b - m)/s),  theta' = (theta - m)/s
    m, s = float(theta.mean()), float(theta.std())
    if s > 0:
        theta = (theta - m) / s
        a = a * s
        b = (b - m) / s

    return {
        "difficulty": b,
        "discrimination": a,
        "ability": theta,
        "n_respondents": R,
        "n_items": n_items,
        "model": "2pl" if two_pl else "1pl",
        "nll": float(res.fun),
        "converged": bool(res.success),
    }


def item_information(a: np.ndarray, b: np.ndarray, theta: float) -> np.ndarray:
    """Fisher information each item provides at ability ``theta``.

    For the 2PL, I_i(theta) = a_i^2 * P_i(theta) * (1 - P_i(theta)) — maximized
    when the item's difficulty matches the ability (P = 0.5) and larger for more
    discriminating items. This is the quantity adaptive testing maximizes.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = _sigmoid(a * (theta - b))
    return a**2 * p * (1 - p)


def total_information(a: np.ndarray, b: np.ndarray, theta_points: np.ndarray) -> np.ndarray:
    """Total item information summed over a set of ability points (per item)."""
    theta_points = np.atleast_1d(np.asarray(theta_points, dtype=float))
    return np.sum([item_information(a, b, t) for t in theta_points], axis=0)


def select_items(a: np.ndarray, b: np.ndarray, n: int, theta_points: np.ndarray | None = None) -> np.ndarray:
    """Indices of the ``n`` most informative items over ``theta_points``.

    Defaults to evaluating information across a standard-normal ability grid, so
    the chosen subset best estimates ability across the population you expect to
    test. Returns indices sorted by descending total information.
    """
    a = np.asarray(a, dtype=float)
    if theta_points is None:
        theta_points = np.linspace(-2.0, 2.0, 9)
    info = total_information(a, b, theta_points)
    n = int(np.clip(n, 0, a.size))
    return np.argsort(info)[::-1][:n]
