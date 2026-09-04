"""Combining several recoverability signals -- and testing whether that helps.

The premise under test
----------------------
    Tasks where an agent has demonstrated meaningful capability but remains
    unreliable may provide more sample-efficient post-training data than random
    tasks or than simply the hardest tasks.

That is a hypothesis. This module does not assume a combined score beats its
components; it makes the comparison possible by computing every candidate on the
same tasks and exposing them all to the selection experiments.

The candidate signals
---------------------
``posterior_gap`` / ``expected_headroom`` / ``evidence_weighted``
    Posterior-only, from :mod:`disteval.reliability.classify`. These need only
    repeated-run outcomes -- no trajectories -- so they are the ones that apply
    to any benchmark.

``neighbourhood`` (D)
    ``1 / (1 + normalized_distance)`` from the trajectory embedding: failed runs
    that live near successful ones score high. Encodes "the gradient has a short
    way to travel".

``failure_concentration`` (H)
    ``1 - normalized failure entropy``: one repeated defect rather than chaos.

``intervention`` (I)
    ``1 / (1 + mean_normalized_cost)`` from the counterfactual edit script: a
    failure one retarget from success scores high.

``uncertainty_penalty`` (U)
    ``1 - posterior_sd / max_sd``: how much of the score is actually supported by
    evidence rather than by the prior. Used multiplicatively in the
    uncertainty-aware variant so a confident middling task outranks a wild guess.

Combination
-----------
``rho_t = f(C_t, R_t, U_t, D_t, H_t, I_t)``

Two combiners are provided, and neither is asserted to be right:

``weighted``
    A convex combination of the normalised signals with configurable weights.
    Transparent and requires no fitting. The default weights lean on the
    posterior signals because they are the only ones available on every
    benchmark, and are declared in :class:`RecoverabilityWeights` rather than
    buried as magic numbers.

``learned``
    Logistic regression of the signals on an observed training-benefit outcome.
    Only usable once you *have* outcomes -- i.e. after running the selection
    experiment once -- and it is therefore explicitly a second-pass tool. It
    reports cross-validated performance, and a model that does not beat the
    single best component out-of-fold is reported as such rather than adopted.

Missing signals
---------------
Trajectory-derived signals are unavailable for tasks with no successes, no
failures, or no trajectory logs. They are propagated as NaN, and the combiner
renormalises over the signals that are present rather than imputing zero --
imputing zero would systematically demote exactly the tasks with sparse data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from .classify import TaskDiagnosis

__all__ = [
    "RecoverabilitySignals",
    "RecoverabilityWeights",
    "SIGNAL_NAMES",
    "combine_weighted",
    "score_tasks",
    "LearnedRecoverability",
    "compare_signals",
    "incremental_validity",
]

SIGNAL_NAMES = (
    "posterior_gap",
    "expected_headroom",
    "evidence_weighted",
    "neighbourhood",
    "failure_concentration",
    "intervention",
    "uncertainty",
)


@dataclass
class RecoverabilitySignals:
    """Every candidate recoverability signal for one task, on a common [0,1] scale.

    NaN means "not measurable here", which is different from zero and is handled
    as such by the combiner.
    """

    task: str
    model: str
    # posterior-only
    posterior_gap: float = float("nan")
    expected_headroom: float = float("nan")
    evidence_weighted: float = float("nan")
    # trajectory-derived
    neighbourhood: float = float("nan")
    failure_concentration: float = float("nan")
    intervention: float = float("nan")
    # evidence weight
    uncertainty: float = float("nan")
    # provenance
    n_runs: int = 0
    n_success: float = 0.0
    n_failure: float = 0.0
    label: str = ""
    extras: dict = field(default_factory=dict)

    def vector(self, names: Sequence[str] = SIGNAL_NAMES) -> np.ndarray:
        return np.array([getattr(self, n, float("nan")) for n in names], dtype=float)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "extras"}
        d.update(self.extras)
        return d


@dataclass(frozen=True)
class RecoverabilityWeights:
    """Weights for the ``weighted`` combiner. Declared, not buried.

    The defaults lean on ``expected_headroom`` because it is the only signal
    available on every benchmark (it needs outcomes, not trajectories) and it is
    the one with the clearest decision-theoretic reading. The trajectory signals
    split the remainder evenly: there is no evidence yet that any of them
    deserves more, and pretending otherwise would be fitting on intuition.
    """

    expected_headroom: float = 0.40
    posterior_gap: float = 0.00
    evidence_weighted: float = 0.00
    neighbourhood: float = 0.20
    failure_concentration: float = 0.20
    intervention: float = 0.20
    #: When True, multiply the combined score by the uncertainty weight, so a
    #: well-evidenced middling task outranks a barely-observed extreme one.
    uncertainty_aware: bool = True

    def as_dict(self) -> dict[str, float]:
        return {
            "expected_headroom": self.expected_headroom,
            "posterior_gap": self.posterior_gap,
            "evidence_weighted": self.evidence_weighted,
            "neighbourhood": self.neighbourhood,
            "failure_concentration": self.failure_concentration,
            "intervention": self.intervention,
        }


def combine_weighted(
    sig: RecoverabilitySignals, weights: Optional[RecoverabilityWeights] = None
) -> float:
    """Convex combination over the signals that are actually available.

    Weights are renormalised over the non-NaN signals. Imputing zero for a
    missing signal would systematically demote tasks with sparse trajectory
    data, which is the opposite of what an uncertainty-aware method should do.
    """
    w = weights or RecoverabilityWeights()
    num, den = 0.0, 0.0
    for name, wt in w.as_dict().items():
        if wt <= 0:
            continue
        v = getattr(sig, name, float("nan"))
        if np.isfinite(v):
            num += wt * float(v)
            den += wt
    if den <= 0:
        return float("nan")
    score = num / den
    if w.uncertainty_aware and np.isfinite(sig.uncertainty):
        score *= float(sig.uncertainty)
    return float(score)


def _uncertainty_weight(posterior_sd: float) -> float:
    """Map posterior sd to an evidence weight in [0, 1].

    The maximum possible sd of a distribution on [0, 1] is 0.5 (all mass at the
    endpoints), so ``1 - 2*sd`` maps "no information" to 0 and "pinned down" to 1.
    """
    if not np.isfinite(posterior_sd):
        return float("nan")
    return float(np.clip(1.0 - 2.0 * posterior_sd, 0.0, 1.0))


def score_tasks(
    diagnoses: Sequence[TaskDiagnosis],
    *,
    neighbourhood: Optional[dict[str, float]] = None,
    failure_concentration: Optional[dict[str, float]] = None,
    intervention: Optional[dict[str, float]] = None,
    weights: Optional[RecoverabilityWeights] = None,
) -> list[RecoverabilitySignals]:
    """Assemble every signal per task and compute the combined score.

    The trajectory-derived dicts are keyed by task id and are all optional; a
    posterior-only run of the pipeline is a first-class use case, not a
    degraded one.
    """
    neighbourhood = neighbourhood or {}
    failure_concentration = failure_concentration or {}
    intervention = intervention or {}
    out: list[RecoverabilitySignals] = []
    for d in diagnoses:
        sig = RecoverabilitySignals(
            task=d.task,
            model=d.model,
            posterior_gap=d.recoverability_gap,
            expected_headroom=d.recoverability_headroom,
            evidence_weighted=d.recoverability_evidence,
            neighbourhood=neighbourhood.get(d.task, float("nan")),
            failure_concentration=failure_concentration.get(d.task, float("nan")),
            intervention=intervention.get(d.task, float("nan")),
            uncertainty=_uncertainty_weight(d.posterior_sd),
            n_runs=d.n_runs,
            n_success=d.n_success,
            n_failure=max(d.n_runs - d.n_success, 0.0),
            label=d.label,
        )
        sig.extras["combined"] = combine_weighted(sig, weights)
        out.append(sig)
    return out


class LearnedRecoverability:
    """Fit the combination to observed training benefit, honestly.

    This is a *second-pass* tool: it requires an outcome per task (did training
    on this task actually help?), which only exists after running the selection
    experiment. Fitting it and then evaluating on the same tasks would be
    circular, so :meth:`fit` reports cross-validated performance and
    :meth:`beats_best_component` compares that against the best single signal
    out of fold. If the combination does not beat its best component, that is the
    result and it is reported, not hidden.
    """

    def __init__(self, signal_names: Sequence[str] = SIGNAL_NAMES, seed: int = 0):
        self.signal_names = list(signal_names)
        self.seed = seed
        self.model = None
        self.cv_score_: float = float("nan")
        self.component_scores_: dict[str, float] = {}
        self.impute_: Optional[np.ndarray] = None

    def _design(self, signals: Sequence[RecoverabilitySignals]) -> np.ndarray:
        X = np.vstack([s.vector(self.signal_names) for s in signals])
        if self.impute_ is None:
            self.impute_ = np.nanmedian(X, axis=0)
            self.impute_ = np.where(np.isfinite(self.impute_), self.impute_, 0.0)
        idx = ~np.isfinite(X)
        X[idx] = np.take(self.impute_, np.where(idx)[1])
        return X

    def fit(
        self, signals: Sequence[RecoverabilitySignals], benefit: Sequence[float]
    ) -> dict:
        """Fit and cross-validate against a per-task training-benefit outcome."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        X = self._design(list(signals))
        y = (np.asarray(benefit, dtype=float) > np.nanmedian(benefit)).astype(int)
        if len(set(y.tolist())) < 2:
            return {"cv_auc": float("nan"), "note": "benefit outcome has one class"}
        n_splits = int(min(5, np.bincount(y).min()))
        if n_splits < 2:
            return {"cv_auc": float("nan"), "note": "too few tasks in the minority class"}
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        clf = LogisticRegression(max_iter=2000)
        self.cv_score_ = float(
            cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()
        )
        self.model = clf.fit(X, y)

        from sklearn.metrics import roc_auc_score

        for i, name in enumerate(self.signal_names):
            col = X[:, i]
            try:
                self.component_scores_[name] = float(roc_auc_score(y, col))
            except ValueError:
                self.component_scores_[name] = float("nan")
        return {
            "cv_auc": self.cv_score_,
            "component_auc": dict(self.component_scores_),
            "best_component": self.best_component(),
            "beats_best_component": self.beats_best_component(),
            "coefficients": dict(zip(self.signal_names, self.model.coef_[0].tolist())),
            "n_tasks": int(X.shape[0]),
            "n_splits": n_splits,
        }

    def best_component(self) -> Optional[str]:
        valid = {k: v for k, v in self.component_scores_.items() if np.isfinite(v)}
        return max(valid, key=valid.get) if valid else None

    def beats_best_component(self, margin: float = 0.02) -> bool:
        """Whether the combination is worth its extra complexity."""
        b = self.best_component()
        if b is None or not np.isfinite(self.cv_score_):
            return False
        return bool(self.cv_score_ > self.component_scores_[b] + margin)

    def predict(self, signals: Sequence[RecoverabilitySignals]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() must be called before predict()")
        return self.model.predict_proba(self._design(list(signals)))[:, 1]


def compare_signals(
    signals: Sequence[RecoverabilitySignals],
    benefit: Sequence[float],
    names: Sequence[str] = SIGNAL_NAMES,
) -> "object":
    """Rank-correlate every signal against observed training benefit.

    Spearman rather than Pearson because only the *ranking* matters -- the
    selection methods take a top-k, so a monotone transform of a signal is the
    same selector. Returns a DataFrame sorted by correlation, which is the
    headline table for "does recoverability actually predict training value".
    """
    import pandas as pd
    from scipy import stats

    b = np.asarray(benefit, dtype=float)
    rows = []
    for name in names:
        v = np.array([getattr(s, name, float("nan")) for s in signals], dtype=float)
        ok = np.isfinite(v) & np.isfinite(b)
        if ok.sum() < 3 or np.all(v[ok] == v[ok][0]):
            rows.append({"signal": name, "n": int(ok.sum()),
                         "spearman": float("nan"), "p_value": float("nan")})
            continue
        rho, p = stats.spearmanr(v[ok], b[ok])
        rows.append({"signal": name, "n": int(ok.sum()),
                     "spearman": float(rho), "p_value": float(p)})
    combined = np.array(
        [s.extras.get("combined", float("nan")) for s in signals], dtype=float
    )
    ok = np.isfinite(combined) & np.isfinite(b)
    if ok.sum() >= 3 and not np.all(combined[ok] == combined[ok][0]):
        rho, p = stats.spearmanr(combined[ok], b[ok])
        rows.append({"signal": "combined", "n": int(ok.sum()),
                     "spearman": float(rho), "p_value": float(p)})
    return pd.DataFrame(rows).sort_values("spearman", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Incremental validity                                                        #
# --------------------------------------------------------------------------- #
def incremental_validity(
    features: Mapping[str, Mapping[str, float]],
    benefit: Mapping[str, float],
    *,
    base: Sequence[str] = ("difficulty", "learning_progress"),
    added: Sequence[str] = ("gap", "failure_concentration", "neighbourhood",
                            "intervention"),
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
) -> dict:
    """Does ``added`` predict training benefit *beyond* ``base``?

    This is the test the central hypothesis actually asks. "Structure correlates
    with benefit" is not the claim -- difficulty correlates with benefit too, and
    structure correlates with difficulty. The claim is **incremental validity**:
    that knowing how a task fails improves prediction over and above knowing how
    hard it is and how fast it is currently improving.

    Method: repeated k-fold cross-validated ridge regression of ``benefit`` on the
    base features alone and on base + added, compared by out-of-fold :math:`R^2`.
    Cross-validated rather than in-sample because adding features can only ever
    increase in-sample :math:`R^2`, so an in-sample comparison would be
    guaranteed to "support" the hypothesis and would mean nothing. Repeated
    because a single k-fold split of a few hundred tasks is noisy enough to flip
    the sign.

    Returns the two scores, the difference, a paired bootstrap interval on the
    difference across repeats, and a verdict that is only positive when the
    interval excludes zero.

    A negative or zero result is a real finding and is reported as such.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import RepeatedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tasks = sorted(set(features) & set(benefit))
    if len(tasks) < 3 * n_splits:
        return {
            "n_tasks": len(tasks), "verdict": "insufficient data",
            "note": f"{len(tasks)} tasks is too few for {n_splits}-fold "
                    "cross-validation to say anything",
        }

    def _design(names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
        cols, used = [], []
        for n in names:
            v = np.array([float(features[t].get(n, np.nan)) for t in tasks])
            if np.all(~np.isfinite(v)):
                continue
            med = np.nanmedian(v[np.isfinite(v)]) if np.any(np.isfinite(v)) else 0.0
            v = np.where(np.isfinite(v), v, med)
            if np.std(v) < 1e-12:
                continue
            cols.append(v)
            used.append(n)
        if not cols:
            return np.zeros((len(tasks), 0)), []
        return np.column_stack(cols), used

    y = np.array([float(benefit[t]) for t in tasks])
    X_base, base_used = _design(base)
    X_add, add_used = _design(list(base) + list(added))

    if X_base.shape[1] == 0 or X_add.shape[1] <= X_base.shape[1]:
        return {
            "n_tasks": len(tasks), "verdict": "not testable",
            "base_features": base_used, "added_features": [],
            "note": "the added features were constant or absent, so there is "
                    "nothing to test for incremental validity",
        }

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    model = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))  # noqa: E731
    s_base = cross_val_score(model(), X_base, y, cv=cv, scoring="r2")
    s_add = cross_val_score(model(), X_add, y, cv=cv, scoring="r2")

    diff = s_add - s_base
    rng = np.random.default_rng(seed)
    boots = np.array([
        diff[rng.integers(0, diff.size, diff.size)].mean() for _ in range(4000)
    ])
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    supported = lo > 0.0

    return {
        "n_tasks": len(tasks),
        "base_features": base_used,
        "added_features": [f for f in add_used if f not in base_used],
        "r2_base": float(s_base.mean()),
        "r2_base_sd": float(s_base.std()),
        "r2_with_added": float(s_add.mean()),
        "r2_with_added_sd": float(s_add.std()),
        "delta_r2": float(diff.mean()),
        "delta_ci_lo": lo,
        "delta_ci_hi": hi,
        "n_folds": int(s_base.size),
        "supported": bool(supported),
        "verdict": (
            "within-task failure structure adds predictive value beyond the base "
            "features on this data"
            if supported else
            "no evidence that within-task failure structure adds predictive value "
            "beyond the base features on this data"
        ),
        "note": (
            "out-of-fold R^2; an in-sample comparison would rise mechanically with "
            "any added feature and could not test this"
        ),
    }
