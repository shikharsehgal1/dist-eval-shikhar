"""How reliability decays with task horizon.

The question
------------
Long-horizon agent work is the setting this repository is aimed at, and the
central empirical regularity there is that success falls off with horizon. If
each of ``L`` steps must go right with probability ``s``, then
``P(success) = s^L``, i.e. ``log P = L * log s`` -- a straight line in log-success
against horizon. Fitting that line gives a per-step reliability with a direct
reading, and comparing the fitted lines across agents says something a pass rate
cannot: *which agent degrades more slowly as tasks get longer*.

Two parameterisations are fit and compared, because the constant-per-step model
is an idealisation:

``geometric``   ``logit P(success) = a + b * L``          (b < 0)
``log_horizon`` ``logit P(success) = a + b * log(1 + L)``

The first says each additional step costs a fixed amount of log-odds; the second
says cost grows sublinearly, which is what you see when agents verify and recover.
Whichever fits better on held-out tasks is reported, along with both, so the
choice is data-driven rather than assumed.

Horizon measures
----------------
``L`` is not uniquely defined. Any of these is a defensible complexity proxy and
they can disagree sharply: number of steps, number of tool calls, number of
distinct tools, number of state transitions, number of distinct targets touched,
context length. :func:`horizon_features` computes them all;
:func:`fit_reliability_curve` takes whichever you name, and the fit reports which
one it used. Fitting against observed step count has an important caveat that is
stated on the result: a *failed* run may be short because it gave up, so step
count is partly an outcome, not only a covariate. Prefer a task-intrinsic
horizon (declared task complexity, number of required subtasks) when available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy import stats
from scipy.special import expit

from ..trajectory.events import EventType, Trajectory

__all__ = [
    "HORIZON_MEASURES",
    "fit_common_form",
    "horizon_features",
    "ScalingFit",
    "fit_reliability_curve",
    "compare_scaling",
]

HORIZON_MEASURES = (
    "n_steps",
    "n_tool_calls",
    "n_unique_tools",
    "n_state_transitions",
    "n_unique_targets",
    "n_retrievals",
    "context_chars",
)


def horizon_features(t: Trajectory) -> dict[str, float]:
    """Every horizon/complexity proxy for one run. See ``HORIZON_MEASURES``."""
    return {
        "n_steps": float(len(t.events)),
        "n_tool_calls": float(sum(1 for e in t.events if e.tool_name)),
        "n_unique_tools": float(len({e.tool_name for e in t.events if e.tool_name})),
        "n_state_transitions": float(
            sum(1 for e in t.events if e.event_type == EventType.STATE_CHANGE)
        ),
        "n_unique_targets": float(len({e.target for e in t.events if e.target})),
        "n_retrievals": float(
            sum(1 for e in t.events if e.event_type == EventType.RETRIEVAL)
        ),
        "context_chars": float(sum(len(e.observation or "") for e in t.events)),
    }


@dataclass
class ScalingFit:
    """A fitted reliability-vs-horizon curve."""

    model: str                 # agent name
    form: str                  # "geometric" | "log_horizon"
    measure: str               # which horizon proxy
    intercept: float
    slope: float
    slope_se: float
    slope_ci: tuple[float, float]
    n: int
    log_likelihood: float
    aic: float
    pseudo_r2: float
    caveat: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def per_step_reliability(self) -> float:
        """Implied per-unit-horizon success factor, only for the geometric form.

        ``exp(slope)`` is the odds multiplier per additional unit of horizon.
        Reported as NaN for the log-horizon form, where no such constant exists.
        """
        return float(np.exp(self.slope)) if self.form == "geometric" else float("nan")

    def predict(self, horizon) -> np.ndarray:
        L = np.asarray(horizon, dtype=float)
        x = L if self.form == "geometric" else np.log1p(L)
        return expit(self.intercept + self.slope * x)

    def half_life(self, base_horizon: float = 0.0) -> float:
        """Additional horizon that halves the odds of success. NaN if not decaying."""
        if self.slope >= 0:
            return float("nan")
        if self.form == "geometric":
            return float(np.log(0.5) / self.slope)
        start = np.log1p(base_horizon)
        return float(np.expm1(start + np.log(0.5) / self.slope) - base_horizon)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "form": self.form,
            "measure": self.measure,
            "intercept": self.intercept,
            "slope": self.slope,
            "slope_se": self.slope_se,
            "slope_ci_lo": self.slope_ci[0],
            "slope_ci_hi": self.slope_ci[1],
            "per_step_reliability": self.per_step_reliability,
            "n": self.n,
            "aic": self.aic,
            "pseudo_r2": self.pseudo_r2,
            "significant_decay": bool(self.slope_ci[1] < 0),
            "caveat": self.caveat,
        }


def _fit_logistic(x: np.ndarray, y: np.ndarray, max_iter: int = 100) -> dict:
    """IRLS logistic regression with an intercept. Returns coefs, SEs, loglik."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(max_iter):
        eta = X @ beta
        p = expit(eta)
        w = np.clip(p * (1 - p), 1e-9, None)
        z = eta + (y - p) / w
        WX = X * w[:, None]
        try:
            new = np.linalg.solve(X.T @ WX, WX.T @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(new - beta)) < 1e-9:
            beta = new
            break
        beta = new
    eta = X @ beta
    p = np.clip(expit(eta), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    w = np.clip(p * (1 - p), 1e-9, None)
    try:
        cov = np.linalg.inv(X.T @ (X * w[:, None]))
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.array([np.nan, np.nan])
    p0 = float(np.clip(y.mean(), 1e-12, 1 - 1e-12))
    ll0 = float(np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0)))
    return {
        "beta": beta,
        "se": se,
        "loglik": ll,
        "pseudo_r2": float(1.0 - ll / ll0) if ll0 != 0 else float("nan"),
    }


def fit_reliability_curve(
    horizons: Sequence[float],
    successes: Sequence[float],
    *,
    form: str = "auto",
    measure: str = "n_steps",
    model: str = "",
    level: float = 0.95,
    caveat: str = "",
) -> ScalingFit:
    """Fit ``P(success | L)`` and report the decay rate with an interval.

    ``form="auto"`` fits both parameterisations and returns the lower-AIC one.
    """
    L = np.asarray(horizons, dtype=float)
    y = np.asarray(successes, dtype=float)
    if L.size != y.size:
        raise ValueError("horizons and successes must be the same length")
    if L.size < 4:
        raise ValueError("need at least 4 runs to fit a scaling curve")
    if len(set(y.tolist())) < 2:
        raise ValueError("all runs have the same outcome; the slope is unidentified")

    forms = ("geometric", "log_horizon") if form == "auto" else (form,)
    best = None
    for f in forms:
        x = L if f == "geometric" else np.log1p(L)
        r = _fit_logistic(x, y)
        aic = 2 * 2 - 2 * r["loglik"]
        if best is None or aic < best[0]:
            best = (aic, f, r)

    aic, chosen, r = best
    z = stats.norm.ppf(1 - (1 - level) / 2)
    slope, se = float(r["beta"][1]), float(r["se"][1])
    default_caveat = (
        "horizon measured from the observed run: a failed run may be short "
        "because it gave up, so this covariate is partly an outcome. Prefer a "
        "task-intrinsic complexity measure where one is available."
        if measure in ("n_steps", "n_tool_calls", "context_chars")
        else ""
    )
    return ScalingFit(
        model=model,
        form=chosen,
        measure=measure,
        intercept=float(r["beta"][0]),
        slope=slope,
        slope_se=se,
        slope_ci=(slope - z * se, slope + z * se),
        n=int(L.size),
        log_likelihood=r["loglik"],
        aic=float(aic),
        pseudo_r2=r["pseudo_r2"],
        caveat=caveat or default_caveat,
        meta={"forms_tried": list(forms)},
    )


def compare_scaling(fits: Sequence[ScalingFit]) -> "object":
    """Rank agents by how slowly they degrade with horizon.

    The decisive column is ``slope``: the agent with the least negative slope
    holds up best as tasks lengthen, which is a different -- and for long-horizon
    deployment more important -- question than who has the highest pass rate.
    Only fits sharing a ``measure`` and ``form`` are comparable, and mixing them
    raises.
    """
    import pandas as pd

    if not fits:
        return pd.DataFrame()
    measures = {f.measure for f in fits}
    forms = {f.form for f in fits}
    if len(measures) > 1 or len(forms) > 1:
        raise ValueError(
            f"fits are not comparable: measures={measures}, forms={forms}"
        )
    return (
        pd.DataFrame([f.to_dict() for f in fits])
        .sort_values("slope", ascending=False)
        .reset_index(drop=True)
    )


def fit_common_form(
    data: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    measure: str = "n_steps",
    level: float = 0.95,
) -> list[ScalingFit]:
    """Fit one shared functional form across several agents, so they compare.

    ``form="auto"`` per agent is the right default in isolation but will happily
    select ``geometric`` for one agent and ``log_horizon`` for another, at which
    point their slopes are in different units and :func:`compare_scaling`
    (correctly) refuses to rank them. This helper picks the single form
    minimising *summed* AIC across all agents and refits everyone with it.

    ``data`` maps agent name to ``(horizons, successes)``.
    """
    scored: dict[str, float] = {}
    per_form: dict[str, list[ScalingFit]] = {}
    for f in ("geometric", "log_horizon"):
        fits = [
            fit_reliability_curve(L, y, form=f, measure=measure, model=name, level=level)
            for name, (L, y) in data.items()
        ]
        per_form[f] = fits
        scored[f] = float(sum(x.aic for x in fits))
    best = min(scored, key=scored.get)
    for fit in per_form[best]:
        fit.meta["form_selected_by"] = "summed AIC across agents"
        fit.meta["aic_by_form"] = dict(scored)
    return per_form[best]
