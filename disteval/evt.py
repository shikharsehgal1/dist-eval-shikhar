"""Extreme value theory for the tail of an agent's outcome distribution.

disteval's whole thesis is tail risk, yet its tail estimates are purely
empirical: with 8 runs, `cvar(x, 0.01)` is just `min(x)` and any quantile below
the smallest order statistic is undefined. Extreme value theory gives a
principled *parametric tail model* that can extrapolate a VaR/CVaR beyond the
observed sample — the direct answer to "empirical CVaR@0.01 with 8 runs is
meaningless."

Peaks-over-threshold: by the Pickands-Balkema-de Haan theorem, exceedances over
a high threshold converge to a Generalized Pareto Distribution (GPD). We fit the
GPD to the tail and read off closed-form VaR / CVaR (expected shortfall), which
extrapolate past the data (McNeil, Frey & Embrechts, *Quantitative Risk
Management*, §7.2).

HONESTY, loudly: classical EVT is asymptotic (many exceedances). Agent evals
have tens of runs, so with < ~15 exceedances point estimates are unstable — the
functions here set a `low_confidence` flag and you should ALWAYS pair a fit with
`bootstrap.confidence_sequence`-style uncertainty or `evt_bootstrap_ci` here.
For bounded [0,1] *scores* the upper tail is light, so heavy-tail tools (Hill)
are meant for *unbounded* metrics (latency, cost, steps-to-solve).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


__all__ = [
    "fit_gpd_pot",
    "evt_var",
    "evt_cvar",
    "hill_estimator",
    "mean_excess",
    "evt_bootstrap_ci",
]

MIN_EXCEEDANCES = 15  # below this, EVT extrapolation is flagged low-confidence


def fit_gpd_pot(
    x: np.ndarray,
    tail: str = "lower",
    threshold: float | None = None,
    tail_fraction: float = 0.25,
    min_exceedances: int = MIN_EXCEEDANCES,
) -> dict:
    """Fit a Generalized Pareto Distribution to the tail via peaks-over-threshold.

    tail="lower" models the risk tail of *bad* outcomes (disteval's usual case);
    internally it reflects the data (y = -x) so the standard upper-tail POT
    machinery applies. If ``threshold`` is None the threshold is the
    ``tail_fraction`` quantile of the appropriate tail.

    Returns {xi (shape), beta (scale), u (threshold, on x-scale), n_exceedances,
    n, tail, low_confidence}. ``low_confidence`` is True when there are fewer than
    ``min_exceedances`` exceedances — treat the fit as indicative only.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("need at least 2 finite observations")
    if tail not in ("lower", "upper"):
        raise ValueError("tail must be 'lower' or 'upper'")
    # Work on y whose UPPER tail is the tail of interest.
    y = -x if tail == "lower" else x
    if threshold is None:
        u_y = float(np.quantile(y, 1.0 - tail_fraction))
    else:
        u_y = -threshold if tail == "lower" else threshold
    exceed = y[y > u_y] - u_y
    n_exc = int(exceed.size)
    if n_exc < 2:
        raise ValueError("too few exceedances above the threshold to fit a GPD")
    # scipy genpareto MLE with location fixed at 0 (exceedances are >= 0).
    xi, _loc, beta = stats.genpareto.fit(exceed, floc=0.0)
    return {
        "xi": float(xi),
        "beta": float(beta),
        "u": float(-u_y if tail == "lower" else u_y),
        "u_y": float(u_y),
        "n_exceedances": n_exc,
        "n": int(y.size),
        "tail": tail,
        "low_confidence": bool(n_exc < min_exceedances),
    }


def evt_var(fit: dict, alpha: float = 0.1) -> float:
    """EVT Value-at-Risk: the alpha-tail quantile from a fitted GPD.

    For a lower-tail fit, returns the score q such that P(score <= q) = alpha
    (the alpha-quantile of the bad tail), extrapolated via the GPD even when
    alpha is smaller than 1/n. ``alpha`` is the tail probability.
    """
    xi, beta = fit["xi"], fit["beta"]
    u_y, n, n_exc = fit["u_y"], fit["n"], fit["n_exceedances"]
    ratio = alpha * n / n_exc  # P(Y > q) target as a fraction of the exceedance prob
    if ratio >= 1:
        import warnings
        warnings.warn(
            f"alpha={alpha} exceeds the exceedance fraction n_exc/n={n_exc / n:.3f}; "
            "the GPD tail model does not extend this far into the body — use a "
            "smaller alpha or a lower threshold. Returning the threshold itself."
        )
        return float(fit["u"])
    if xi == 0:
        q_y = u_y - beta * np.log(ratio)
    else:
        q_y = u_y + (beta / xi) * (ratio ** (-xi) - 1.0)
    return float(-q_y if fit["tail"] == "lower" else q_y)


def evt_cvar(fit: dict, alpha: float = 0.1) -> float:
    """EVT Conditional Value-at-Risk (expected shortfall) from a fitted GPD.

    Mean outcome in the worst-alpha tail, extrapolated via the GPD. Requires
    xi < 1 (otherwise the tail has infinite mean and CVaR is undefined -> nan,
    with a warning).
    """
    xi, beta, u_y = fit["xi"], fit["beta"], fit["u_y"]
    if xi >= 1:
        import warnings
        warnings.warn("GPD shape xi >= 1: infinite-mean tail, CVaR undefined")
        return float("nan")
    var = evt_var(fit, alpha)
    q_y = -var if fit["tail"] == "lower" else var
    # ES on the y (upper) scale: E[Y | Y > q_y] = q_y/(1-xi) + (beta - xi*u_y)/(1-xi)
    es_y = q_y / (1 - xi) + (beta - xi * u_y) / (1 - xi)
    return float(-es_y if fit["tail"] == "lower" else es_y)


def hill_estimator(x: np.ndarray, k: int | None = None) -> float:
    """Hill tail-index estimate gamma = 1/alpha for a heavy (Fréchet) upper tail.

    Uses the top-k order statistics of positive data:
    gamma_hat = mean(log X_(n-i+1) - log X_(n-k)), i=1..k. Meaningful only for
    genuinely heavy, unbounded, positive tails (latency, cost, steps) — NOT for
    bounded [0,1] scores, where it returns nan. Returns the tail index gamma
    (alpha = 1/gamma is the Pareto exponent). ``k`` defaults to floor(sqrt(n)).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3 or np.any(x <= 0):
        return float("nan")
    n = x.size
    if k is None:
        k = max(2, int(np.floor(np.sqrt(n))))
    k = int(min(k, n - 1))
    xs = np.sort(x)[::-1]  # descending
    top = xs[:k + 1]
    return float(np.mean(np.log(top[:k]) - np.log(xs[k])))


def mean_excess(x: np.ndarray, thresholds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean-excess (mean-residual-life) function e(u) = E[X - u | X > u].

    Threshold-selection diagnostic for POT: a region of u where e(u) is roughly
    linear indicates the GPD approximation holds there (the slope is xi/(1-xi)).
    Returns (thresholds, mean_excess_values); thresholds with no exceedances give
    nan. This is an eyeball diagnostic and is hard to read at small n.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if thresholds is None:
        thresholds = np.quantile(x, np.linspace(0.5, 0.95, 20))
    thresholds = np.asarray(thresholds, dtype=float)
    out = np.array([
        float((x[x > u] - u).mean()) if np.any(x > u) else float("nan")
        for u in thresholds
    ])
    return thresholds, out


def evt_bootstrap_ci(
    x: np.ndarray,
    estimand: str = "cvar",
    alpha: float = 0.1,
    ci: float = 0.95,
    n_reps: int = 1000,
    seed: int = 0,
    **fit_kwargs,
) -> dict:
    """Bootstrap CI for an EVT tail estimand — the small-sample honesty guard.

    At small n an EVT point estimate is a coin flip; this resamples the data,
    re-fits the GPD, and returns a percentile CI so the (often large) uncertainty
    is visible rather than hidden behind a falsely precise number. ``estimand`` is
    "var" or "cvar". Returns {point, lo, hi, width, ci, low_confidence}.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    fn = evt_cvar if estimand == "cvar" else evt_var
    base_fit = fit_gpd_pot(x, **fit_kwargs)
    point = fn(base_fit, alpha)
    boot = []
    for _ in range(n_reps):
        xb = rng.choice(x, size=x.size, replace=True)
        try:
            boot.append(fn(fit_gpd_pot(xb, **fit_kwargs), alpha))
        except (ValueError, RuntimeError):
            continue
    boot = np.array([b for b in boot if np.isfinite(b)], dtype=float)
    if boot.size < 2:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "width": float("nan"), "ci": ci, "low_confidence": True}
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "width": float(hi - lo), "ci": ci,
            "low_confidence": bool(base_fit["low_confidence"])}
