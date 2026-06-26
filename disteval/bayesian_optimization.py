"""disteval.bayesian_optimization — sample-efficient optimization for disteval.

Provides a lightweight Bayesian optimizer built on scikit-learn's GaussianProcessRegressor.
It can be used to tune DPO hyperparameters, curriculum ordering, recursion boundaries, and
any other objective that is expensive to evaluate but can be queried cheaply enough for a
few dozen iterations.

Mathematical overview
─────────────────────
Given an unknown objective f: X → R, we maintain a Gaussian process surrogate

    f(x) ~ GP(μ(x), k(x, x')),

with a Matern kernel by default. At each iteration we observe y_i = f(x_i) + ε_i and update
the posterior μ_t(x) and σ_t(x). The next point is chosen by maximizing an acquisition
function over X:

  * Expected Improvement (EI):
      EI(x) = E[max(0, f(x) − f*)]
            = (μ_t(x) − f*) Φ(Z) + σ_t(x) φ(Z),
      where Z = (μ_t(x) − f*) / σ_t(x), and f* is the best observed value.

  * Upper Confidence Bound (UCB):
      UCB(x) = μ_t(x) + β σ_t(x),
      where β controls exploration-exploitation trade-off.

Mixed discrete/continuous search spaces are handled by rounding integer-valued variables
inside the objective wrapper.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import numpy as np

from .records import EpisodeRecord
from .right_tail import RightTailReport
from scipy import stats
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

# Silence sklearn convergence warnings during small-sample BO
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class SearchSpace:
    """Named search-space dimensions with optional integer rounding."""

    name: str
    lower: float
    upper: float
    integer: bool = False
    log_scale: bool = False

    def encode(self, value: float) -> float:
        """Map a raw value to the internal encoding used by the optimizer."""
        if self.log_scale:
            return float(np.log(value))
        return float(value)

    def decode(self, encoded: float) -> float:
        """Map an encoded value back to the raw domain, clipping to bounds."""
        raw = float(np.exp(encoded)) if self.log_scale else float(encoded)
        raw = float(np.clip(raw, self.lower, self.upper))
        if self.integer:
            raw = int(np.round(raw))
        return raw


@dataclass
class BayesianOptimizer:
    """Generic Bayesian optimizer for scalar objectives on bounded spaces.

    Parameters
    ----------
    space
        List of search-space dimensions.
    acquisition
        Acquisition function: "ei" (Expected Improvement) or "ucb".
    beta
        Exploration parameter for UCB (ignored for EI).
    n_init
        Number of random Latin-hypercube samples before the surrogate guides search.
    noise_std
        Expected observation noise, used as a small prior for numerical stability.
    seed
        Random seed for reproducibility.
    """

    space: list[SearchSpace]
    acquisition: Literal["ei", "ucb"] = "ei"
    beta: float = 2.0
    n_init: int = 5
    noise_std: float = 1e-5
    seed: int = 42
    history: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._best_y: float = -float("inf")
        self._best_x: Optional[np.ndarray] = None
        self._build_gp()

    def _build_gp(self) -> None:
        # Matern 5/2 is a good default for moderately smooth objectives.
        kernel = Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 10.0))
        if self.noise_std > 0:
            kernel = kernel + WhiteKernel(noise_level=self.noise_std**2, noise_level_bounds=(1e-10, 1.0))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=int(self.rng.integers(0, 2**31)),
        )

    def _encode(self, x: np.ndarray) -> np.ndarray:
        return np.array([dim.encode(float(v)) for dim, v in zip(self.space, x)])

    def _decode(self, encoded: np.ndarray) -> np.ndarray:
        return np.array([dim.decode(float(v)) for dim, v in zip(self.space, encoded)])

    def _random_sample(self) -> np.ndarray:
        """Latin-hypercube-style random sample in encoded space."""
        encoded = []
        for dim in self.space:
            lo = dim.encode(dim.lower)
            hi = dim.encode(dim.upper)
            encoded.append(self.rng.uniform(lo, hi))
        return np.array(encoded)

    def _acquisition(self, encoded: np.ndarray) -> float:
        """Evaluate acquisition function in encoded space."""
        if not self._X:
            return 0.0
        X = np.atleast_2d(encoded)
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = float(mu[0])
        sigma = float(sigma[0])
        if self.acquisition == "ucb":
            return mu + self.beta * sigma
        # Expected Improvement
        if sigma < 1e-9:
            return 0.0
        imp = mu - self._best_y
        z = imp / sigma
        ei = imp * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        return float(ei)

    def _neg_acquisition(self, encoded: np.ndarray) -> float:
        return -self._acquisition(encoded)

    def _propose(self) -> np.ndarray:
        """Propose the next encoded point by optimizing the acquisition function."""
        # Random restarts to avoid local optima in the acquisition surface.
        best_x = None
        best_acq = -float("inf")
        for _ in range(10):
            x0 = self._random_sample()
            bounds = [
                (dim.encode(dim.lower), dim.encode(dim.upper)) for dim in self.space
            ]
            result = minimize(
                self._neg_acquisition,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 100},
            )
            if result.success and -result.fun > best_acq:
                best_acq = -result.fun
                best_x = result.x
        if best_x is None:
            best_x = self._random_sample()
        return best_x

    def _tell(self, x: np.ndarray, y: float) -> None:
        """Record an observation."""
        encoded = self._encode(x)
        self._X.append(encoded)
        self._y.append(float(y))
        if y > self._best_y:
            self._best_y = float(y)
            self._best_x = np.array(x)
        self.history.append({
            "x": [float(v) for v in x],
            "y": float(y),
            "encoded": [float(v) for v in encoded],
        })
        if len(self._X) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(np.vstack(self._X), np.array(self._y))

    def _ask(self) -> np.ndarray:
        """Return the next point to evaluate."""
        if len(self._X) < self.n_init:
            return self._decode(self._random_sample())
        return self._decode(self._propose())

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        n_iter: int,
        maximize: bool = True,
    ) -> dict:
        """Run the Bayesian optimization loop.

        Parameters
        ----------
        objective
            Function that takes a decoded parameter vector and returns a scalar.
        n_iter
            Total number of function evaluations (including initialization).
        maximize
            If True, maximize the objective; otherwise minimize it.

        Returns
        -------
        dict with best parameters, best value, history, and number of iterations.
        """
        sign = 1.0 if maximize else -1.0
        for _ in range(n_iter):
            x = self._ask()
            y = sign * objective(x)
            self._tell(x, y)

        best_idx = int(np.argmax(self._y))
        best_x = self._decode(self._X[best_idx])
        best_y = sign * self._y[best_idx]
        return {
            "best_params": {
                dim.name: float(best_x[i]) for i, dim in enumerate(self.space)
            },
            "best_value": float(best_y),
            "n_iter": n_iter,
            "history": [
                {
                    "params": {
                        dim.name: float(self._decode(np.array(h["encoded"]))[i])
                        for i, dim in enumerate(self.space)
                    },
                    "value": sign * h["y"],
                }
                for h in self.history
            ],
        }

    def get_best(self) -> dict:
        """Return the best parameters seen so far without further optimization."""
        if not self._X:
            raise RuntimeError("No observations have been recorded yet.")
        best_idx = int(np.argmax(self._y))
        best_x = self._decode(self._X[best_idx])
        return {
            "best_params": {
                dim.name: float(best_x[i]) for i, dim in enumerate(self.space)
            },
            "best_value": float(self._y[best_idx]),
        }



# ── Thompson Sampling task scheduler (contextual bandit) ─────────────────────


class ThompsonSamplingScheduler:
    """Online task scheduler using Thompson Sampling for linear payoffs.

    Maintains a Gaussian posterior over feature weights θ. At each cycle a sample
    θ̃ is drawn and the task with the highest predicted reward x_i^T θ̃ is selected.

    Posterior update (Gaussian likelihood, Gaussian prior):

        Σ_t^{-1} = Σ_{t-1}^{-1} + x_t x_t^T / σ^2
        μ_t      = Σ_t (Σ_{t-1}^{-1} μ_{t-1} + x_t r_t / σ^2)

    This is a contextual bandit (Chu et al. 2011; Agrawal & Goyal 2013).
    """

    def __init__(
        self,
        feature_dim: int = 5,
        lambda_prior: float = 1.0,
        sigma_noise: float = 0.05,
        seed: int = 42,
    ):
        self.d = feature_dim
        self.sigma2 = sigma_noise**2
        self.mu = np.zeros(feature_dim)
        self.Sigma_inv = (1.0 / lambda_prior) * np.eye(feature_dim)
        self.Sigma = lambda_prior * np.eye(feature_dim)
        self.rng = np.random.default_rng(seed)

    def update(self, x: np.ndarray, reward: float) -> None:
        """Observe a training outcome (feature vector, reward) and update the posterior."""
        x = np.asarray(x, dtype=float).reshape(self.d)
        self.Sigma_inv += np.outer(x, x) / self.sigma2
        self.Sigma = np.linalg.inv(self.Sigma_inv)
        self.mu = self.Sigma @ (self.Sigma_inv @ self.mu + x * reward / self.sigma2)

    def select(self, task_features: dict[str, np.ndarray]) -> str:
        """Sample θ̃ ~ N(μ, Σ) and return the task with highest predicted reward."""
        theta_tilde = self.rng.multivariate_normal(self.mu, self.Sigma)
        best_task = None
        best_reward = -float("inf")
        for task, x in task_features.items():
            reward = float(np.dot(np.asarray(x, dtype=float), theta_tilde))
            if reward > best_reward:
                best_reward = reward
                best_task = task
        return best_task  # type: ignore[return-value]

    def rank(self, tasks: list[Any], feature_fn: Optional[Callable[[Any], np.ndarray]] = None) -> list[Any]:
        """Rank tasks by repeatedly sampling θ̃ and selecting the highest-scoring task.

        Uses the current posterior without updating it, so repeated calls can differ
        due to sampling. This is intentional: it encodes posterior uncertainty into
        the ordering.
        """
        if feature_fn is None:
            def feature_fn(task):
                gap = float(getattr(task, "gap", 0.0))
                consistency = float(getattr(task, "consistency", 0.0))
                difficulty = str(getattr(task, "difficulty", "medium")).lower()
                diff_map = {"easy": 0.0, "medium": 0.5, "hard": 1.0}
                pairs = float(len(getattr(task, "training_pairs", []))) / 10.0
                return np.array([gap, 1.0 - consistency, diff_map.get(difficulty, 0.5), pairs])

        remaining = list(tasks)
        ordered = []
        while remaining:
            features = {getattr(t, "task", str(id(t))): feature_fn(t) for t in remaining}
            chosen_id = self.select(features)
            chosen = next(t for t in remaining if getattr(t, "task", str(id(t))) == chosen_id)
            remaining.remove(chosen)
            ordered.append(chosen)
        return ordered


# Optional: typed helpers for common disteval optimization problems.


def optimize_dpo_hyperparameters(
    report: RightTailReport,
    records: list[EpisodeRecord],
    n_iter: int = 20,
    seed: int = 42,
) -> dict:
    """Optimize DPO-style training hyperparameters with a fast surrogate.

    Parameters
    ----------
    report
        RightTailReport from `right_tail_analysis`.
    records
        Episode records used to compute baseline and apply the training effect.
    n_iter
        Number of BO evaluations.
    seed
        Random seed.

    Returns
    -------
    dict with best alpha, beta, k, and predicted gain.
    """
    from .training_sim import apply_training_effect, select_disteval_right_tail

    baseline = float(np.mean([r.score for r in records]))

    def objective(x: np.ndarray) -> float:
        alpha, dpo_bonus, k = float(x[0]), float(x[1]), int(round(x[2]))
        k = max(1, min(k, len(records) // 2))
        selected = select_disteval_right_tail(records, report, k)
        new_scores = apply_training_effect(
            records,
            selected,
            report,
            alpha=alpha,
            strategy="disteval_right_tail",
            dpo_bonus=dpo_bonus,
        )
        return float(np.mean(new_scores) - baseline)

    space = [
        SearchSpace("alpha", 0.1, 0.8, log_scale=False),
        SearchSpace("dpo_bonus", 0.5, 3.0, log_scale=False),
        SearchSpace("k", 1, 10, integer=True),
    ]

    optimizer = BayesianOptimizer(
        space=space,
        acquisition="ei",
        n_init=5,
        seed=seed,
    )
    return optimizer.optimize(objective, n_iter=n_iter, maximize=True)


# ── Curriculum scheduling helper (uses BO surrogate as a gain model) ─────────


class BayesianCurriculumScheduler:
    """Learn a task-priority model from historical cycle gains.

    Each task is featurized by (gap, 1 - consistency, difficulty). A GP is fit on
    historical (features, observed_gain) pairs. The predicted mean is then used to
    re-rank the current curriculum.
    """

    def __init__(
        self,
        seed: int = 42,
        difficulty_map: Optional[dict[str, int]] = None,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.difficulty_map = difficulty_map or {"easy": 0, "medium": 1, "hard": 2}
        self.gp = GaussianProcessRegressor(
            kernel=Matern(length_scale=1.0, nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=int(self.rng.integers(0, 2**31)),
        )
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._fitted = False

    def _featurize(self, task: Any) -> np.ndarray:
        """Feature vector from a TaskOutcomeProfile or similar object."""
        gap = float(getattr(task, "gap", 0.0))
        consistency = float(getattr(task, "consistency", 0.0))
        difficulty = str(getattr(task, "difficulty", "medium")).lower()
        diff_enc = self.difficulty_map.get(difficulty, 1)
        return np.array([gap, 1.0 - consistency, diff_enc])

    def observe(self, task: Any, gain: float) -> None:
        """Record one historical observation: training this task yielded `gain`."""
        self._X.append(self._featurize(task))
        self._y.append(float(gain))
        self._fitted = False

    def fit(self) -> None:
        """Fit the GP on observed historical gains."""
        if len(self._X) < 2:
            return
        X = np.vstack(self._X)
        y = np.array(self._y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(X, y)
        self._fitted = True

    def predict_gain(self, task: Any) -> tuple[float, float]:
        """Return (mean, std) predicted gain for a task."""
        x = np.atleast_2d(self._featurize(task))
        if not self._fitted:
            # Fallback to the heuristic priority score when no history exists.
            gap = float(getattr(task, "gap", 0.0))
            consistency = float(getattr(task, "consistency", 0.0))
            return gap * (1.0 - consistency), 0.0
        mu, sigma = self.gp.predict(x, return_std=True)
        return float(mu[0]), float(sigma[0])

    def rank(self, tasks: list[Any]) -> list[Any]:
        """Rank tasks by predicted gain (descending)."""
        self.fit()
        scored = [(self.predict_gain(t)[0], t) for t in tasks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored]
