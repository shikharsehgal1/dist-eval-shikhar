"""Sequential reliability: when during a run does it go wrong, and does it recover?

Why final success rate is not enough
------------------------------------
Two policies with identical pass rates can have completely different failure
dynamics. One fails early and cheaply; the other runs 200 steps, corrupts state
at step 30, and spends the rest of the run producing a confidently wrong
artifact. For long-horizon agents these are different problems with different
fixes, and a scalar success rate cannot distinguish them.

So each run is treated as a survival process over its step clock:

    tau = (s_0, a_0, s_1, a_1, ..., s_T)
    h_j = P(irrecoverable failure at step j | survived to step j)

and the standard estimators apply.

What "event" and "censoring" mean here
--------------------------------------
* The **event** is the first *irrecoverable* failure -- an error the run never
  came back from. A run that errors at step 5 and finishes successfully was not
  irrecoverable at step 5, and counting it as an event would make early steps
  look far more dangerous than they are. Determining irrecoverability requires
  knowing the run's outcome, so this is necessarily a retrospective labelling;
  :func:`run_survival_record` states the rule it uses and it is overridable.
* A successful run is **right-censored** at its final step: it survived at least
  that long and we have no failure time for it. Kaplan-Meier handles this
  correctly, which is the whole reason for using it rather than a histogram of
  failure steps -- the histogram silently conditions on failing.

The step clock is a weak clock
------------------------------
Step index is only comparable across runs to the extent that runs do comparable
things per step. When alignment is available, pass ``clock="aligned"`` to use
each event's aligned column instead of its raw index. This is stated in the
metric registry as an assumption of the hazard estimate and it is a real
limitation, not a formality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from ..trajectory.events import EventType, Trajectory

__all__ = [
    "SurvivalRecord",
    "SurvivalCurve",
    "run_survival_record",
    "kaplan_meier",
    "nelson_aalen",
    "hazard_by_step",
    "time_to_first_error",
    "recovery_probability",
    "survival_summary",
]


@dataclass(frozen=True)
class SurvivalRecord:
    """One run reduced to (time, event) for survival estimation."""

    trajectory_id: str
    task: str
    model: str
    time: int          # step index of the event, or of censoring
    event: bool        # True = irrecoverable failure observed; False = censored
    n_steps: int
    first_error_step: Optional[int] = None
    n_errors: int = 0
    recovered_from_error: bool = False

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "task": self.task,
            "model": self.model,
            "time": self.time,
            "event": self.event,
            "n_steps": self.n_steps,
            "first_error_step": self.first_error_step,
            "n_errors": self.n_errors,
            "recovered_from_error": self.recovered_from_error,
        }


def run_survival_record(
    t: Trajectory,
    *,
    irrecoverable: Optional[Callable[[Trajectory], Optional[int]]] = None,
) -> SurvivalRecord:
    """Reduce a trajectory to a survival record.

    Default irrecoverability rule, stated explicitly so it can be argued with:

    * A **successful** run is censored at its last step. It never failed
      irrecoverably by definition.
    * A **failed** run's event time is its *last* error step -- the point after
      which it never got back on track. Using the *first* error would treat a
      run that errored early, recovered, and failed for an unrelated reason at
      the end as though the early error were fatal.
    * A failed run with **no** error events is treated as failing at its final
      step: the failure is in the artifact, not in any single action.

    Override with ``irrecoverable``, a callable returning the failure step or
    ``None`` for censored.
    """
    n = len(t.events)
    err_steps = [
        e.index for e in t.events if e.ok is False or e.event_type == EventType.ERROR
    ]
    first_err = err_steps[0] if err_steps else None

    if irrecoverable is not None:
        step = irrecoverable(t)
        event = step is not None
        time = step if event else max(n - 1, 0)
    elif t.success:
        event, time = False, max(n - 1, 0)
    else:
        event = True
        time = err_steps[-1] if err_steps else max(n - 1, 0)

    return SurvivalRecord(
        trajectory_id=t.trajectory_id,
        task=t.task,
        model=t.model,
        time=int(time),
        event=bool(event),
        n_steps=n,
        first_error_step=first_err,
        n_errors=len(err_steps),
        recovered_from_error=bool(first_err is not None and t.success),
    )


@dataclass
class SurvivalCurve:
    """A Kaplan-Meier survival curve with Greenwood standard errors."""

    times: np.ndarray
    survival: np.ndarray
    se: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    cumulative_hazard: np.ndarray
    n_runs: int
    n_events: int

    def at(self, step: int) -> float:
        """Estimated probability of surviving past ``step``."""
        if self.times.size == 0:
            return 1.0
        idx = np.searchsorted(self.times, step, side="right") - 1
        return 1.0 if idx < 0 else float(self.survival[idx])

    def ci(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise Greenwood interval, clipped to [0, 1]."""
        from scipy import stats

        z = stats.norm.ppf(1 - (1 - level) / 2)
        lo = np.clip(self.survival - z * self.se, 0.0, 1.0)
        hi = np.clip(self.survival + z * self.se, 0.0, 1.0)
        return lo, hi

    def median_survival(self) -> Optional[float]:
        """First step at which survival drops to or below 0.5. None if it never does."""
        below = np.where(self.survival <= 0.5)[0]
        return float(self.times[below[0]]) if below.size else None

    def to_frame(self):
        import pandas as pd

        lo, hi = self.ci()
        return pd.DataFrame(
            {
                "step": self.times,
                "survival": self.survival,
                "se": self.se,
                "ci_lo": lo,
                "ci_hi": hi,
                "at_risk": self.at_risk,
                "events": self.events,
                "cumulative_hazard": self.cumulative_hazard,
            }
        )


def kaplan_meier(records: Sequence[SurvivalRecord]) -> SurvivalCurve:
    """Kaplan-Meier survival estimate with Greenwood variance.

    Handles right-censoring correctly, which is the point: a histogram of failure
    steps silently conditions on the run having failed and will report that late
    steps are safe simply because few runs get there.
    """
    if not records:
        empty = np.array([])
        return SurvivalCurve(empty, empty, empty, empty, empty, empty, 0, 0)

    times = np.array([r.time for r in records], dtype=float)
    events = np.array([r.event for r in records], dtype=bool)
    uniq = np.unique(times[events]) if events.any() else np.array([])

    n = len(records)
    surv, se_, risk_, ev_, cumhaz = [], [], [], [], []
    s = 1.0
    var_acc = 0.0
    H = 0.0
    for t in uniq:
        at_risk = int(np.sum(times >= t))
        d = int(np.sum((times == t) & events))
        if at_risk == 0:
            continue
        s *= 1.0 - d / at_risk
        H += d / at_risk
        if at_risk > d:
            var_acc += d / (at_risk * (at_risk - d))
        surv.append(s)
        se_.append(s * np.sqrt(var_acc))
        risk_.append(at_risk)
        ev_.append(d)
        cumhaz.append(H)

    return SurvivalCurve(
        times=np.asarray(uniq, dtype=float),
        survival=np.asarray(surv, dtype=float),
        se=np.asarray(se_, dtype=float),
        at_risk=np.asarray(risk_, dtype=int),
        events=np.asarray(ev_, dtype=int),
        cumulative_hazard=np.asarray(cumhaz, dtype=float),
        n_runs=n,
        n_events=int(events.sum()),
    )


def nelson_aalen(records: Sequence[SurvivalRecord]) -> np.ndarray:
    """Nelson-Aalen cumulative hazard estimate (returned by :func:`kaplan_meier` too)."""
    return kaplan_meier(records).cumulative_hazard


def hazard_by_step(
    records: Sequence[SurvivalRecord], n_bins: Optional[int] = None
) -> "object":
    """Discrete hazard ``h_j = d_j / n_j`` per step, optionally binned.

    Binning matters in practice: at 200 steps and 8 runs, every per-step risk set
    is tiny and the raw hazard is pure noise. ``n_bins`` pools adjacent steps into
    equal-width bins over the observed range.
    """
    import pandas as pd

    if not records:
        return pd.DataFrame(columns=["step", "at_risk", "events", "hazard"])
    times = np.array([r.time for r in records], dtype=float)
    events = np.array([r.event for r in records], dtype=bool)
    max_t = int(times.max())

    if n_bins:
        edges = np.linspace(0, max_t + 1, n_bins + 1)
        labels = 0.5 * (edges[:-1] + edges[1:])
    else:
        edges = np.arange(0, max_t + 2)
        labels = edges[:-1]

    rows = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        at_risk = int(np.sum(times >= lo))
        d = int(np.sum((times >= lo) & (times < hi) & events))
        rows.append(
            {
                "step": float(lab),
                "at_risk": at_risk,
                "events": d,
                "hazard": (d / at_risk) if at_risk else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def time_to_first_error(trajectories: Sequence[Trajectory]) -> dict:
    """Distribution of the step at which the first error appears.

    Reported separately from the survival curve because "when does anything go
    wrong" and "when does it go irrecoverably wrong" are different questions and
    the gap between them is exactly the agent's recovery capacity.
    """
    steps = [
        r.first_error_step
        for r in (run_survival_record(t) for t in trajectories)
        if r.first_error_step is not None
    ]
    n_clean = len(trajectories) - len(steps)
    if not steps:
        return {
            "n": 0, "n_error_free": n_clean, "mean": float("nan"),
            "median": float("nan"), "p10": float("nan"), "p90": float("nan"),
        }
    arr = np.asarray(steps, dtype=float)
    return {
        "n": int(arr.size),
        "n_error_free": int(n_clean),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.1)),
        "p90": float(np.quantile(arr, 0.9)),
    }


def recovery_probability(trajectories: Sequence[Trajectory]) -> dict:
    """``P(run succeeds | an error occurred)`` -- the agent's recovery capacity.

    Compared against ``P(run succeeds | no error occurred)``, the difference is
    how much an error actually costs. A small difference means errors are
    routine and handled; a large one means a single error is effectively fatal.
    """
    with_err, without_err = [], []
    for t in trajectories:
        (with_err if t.n_errors() > 0 else without_err).append(bool(t.success))
    p_err = float(np.mean(with_err)) if with_err else float("nan")
    p_clean = float(np.mean(without_err)) if without_err else float("nan")
    return {
        "n_with_error": len(with_err),
        "n_without_error": len(without_err),
        "p_success_given_error": p_err,
        "p_success_given_no_error": p_clean,
        "error_cost": (p_clean - p_err) if (with_err and without_err) else float("nan"),
    }


def survival_summary(trajectories: Sequence[Trajectory], n_bins: int = 10) -> dict:
    """One-shot sequential-reliability summary for a set of runs."""
    records = [run_survival_record(t) for t in trajectories]
    curve = kaplan_meier(records)
    return {
        "n_runs": len(records),
        "n_irrecoverable": int(sum(r.event for r in records)),
        "median_survival_step": curve.median_survival(),
        "final_survival": float(curve.survival[-1]) if curve.survival.size else 1.0,
        "cumulative_hazard": float(curve.cumulative_hazard[-1])
        if curve.cumulative_hazard.size else 0.0,
        "time_to_first_error": time_to_first_error(trajectories),
        "recovery": recovery_probability(trajectories),
        "hazard_table": hazard_by_step(records, n_bins=n_bins).to_dict("records"),
    }
