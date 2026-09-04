"""A simulator with known ground truth, for validating the methodology itself.

Why this exists
---------------
Every estimator in this repository makes a claim about a latent quantity that is
never observed on real data. Whether the Beta-Binomial posterior actually recovers
``p_t``, whether the observed max is as biased as claimed, whether adaptive
sampling saves anything, whether the classifier mislabels tasks -- none of these
can be answered by running on a benchmark, because the truth is unavailable
there. They can be answered exactly here.

The world model
---------------
Tasks are generated with known latent properties::

    q_t   ~ Beta            capability: P(the agent finds the right approach)
    r_t   ~ Beta            execution reliability: P(finishes | right approach)
    p_t   = q_t * r_t       the latent success probability actually observed
    g_t                     TRUE training benefit: reliability gain from training
                            on this task

Runs are then simulated as::

    A ~ Bernoulli(q_t)                      milestone reached
    Y ~ Bernoulli(r_t) if A else 0          success

which is exactly the generative process the identifiable decomposition in
:mod:`disteval.reliability.decomposition` assumes -- so the simulation can check
whether that estimator recovers ``q`` and ``r``, and (by breaking the necessity
assumption on request) what happens when it should not.

Failure modes and trajectories are generated too, with a per-task
``failure_concentration`` controlling whether a task fails the same way every
time or differently each time, so the failure-entropy signal can be validated
against a known truth rather than assumed.

The honesty constraint
----------------------
``benefit_coupling`` controls how ``g_t`` relates to recoverability, and it is
the single most important knob here:

``"strong"``       g_t is largely determined by true recoverability
``"weak"``         g_t is partly determined by it, mostly noise
``"none"``         g_t is independent of it
``"adversarial"``  g_t is *anti*-correlated: the hardest tasks are the ones
                   worth training on, and recoverability selection should lose

A simulator that only implements ``"strong"`` would make the method win by
construction and prove nothing. The validation suite runs all four and reports
the method's behaviour in each; the ``"none"`` and ``"adversarial"`` worlds are
where a selection method is supposed to fail, and a method that appears to win
there is broken, not brilliant.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

import numpy as np

from ..diagnosis.taxonomy import TAXONOMY
from ..trajectory.events import EventType, Trajectory, TrajectoryEvent

__all__ = [
    "TaskTruth",
    "WorldConfig",
    "SimulatedWorld",
    "BENEFIT_COUPLINGS",
]

BENEFIT_COUPLINGS = ("strong", "weak", "none", "adversarial")


@dataclass
class TaskTruth:
    """Everything true about a simulated task. Never shown to any estimator."""

    task: str
    domain: str
    q: float                       # capability
    r: float                       # execution reliability
    p: float                       # q * r
    benefit: float                 # g_t: true reliability gain from training here
    failure_modes: list[str]
    failure_probs: list[float]
    failure_entropy: float
    horizon: int
    true_label: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorldConfig:
    """Generator settings. Every default is declared, none is magic."""

    n_tasks: int = 120
    n_domains: int = 4
    #: Beta parameters for the capability and execution-reliability priors.
    capability_prior: tuple[float, float] = (2.0, 1.2)
    execution_prior: tuple[float, float] = (2.5, 1.5)
    #: Between-domain spread of the capability logit.
    domain_effect_sd: float = 0.8
    #: How g_t relates to true recoverability. See the module docstring.
    benefit_coupling: Literal["strong", "weak", "none", "adversarial"] = "weak"
    #: Noise added to g_t regardless of coupling.
    benefit_noise: float = 0.15
    #: Number of distinct failure modes a task can exhibit.
    n_failure_modes: int = 6
    #: Beta(a, a) controlling how concentrated each task's failure distribution is.
    failure_concentration_a: float = 0.7
    horizon_range: tuple[int, int] = (6, 40)
    #: Classification thresholds used to define the TRUE label from p.
    tau_cap: float = 0.15
    tau_rel: float = 0.90
    tau_stuck: float = 0.10
    #: Break the decomposition's necessity assumption at this rate, to test what
    #: happens when the milestone is mis-specified.
    necessity_violation_rate: float = 0.0
    seed: int = 0


def _true_label(p: float, cfg: WorldConfig) -> str:
    if p >= cfg.tau_rel:
        return "SOLID"
    if p < cfg.tau_stuck:
        return "STUCK"
    if p > cfg.tau_cap:
        return "RECOVERABLE"
    return "MARGINAL"


class SimulatedWorld:
    """Generates tasks with known truth, and runs/trajectories from them."""

    def __init__(self, config: Optional[WorldConfig] = None):
        self.config = config or WorldConfig()
        if self.config.benefit_coupling not in BENEFIT_COUPLINGS:
            raise ValueError(
                f"benefit_coupling must be one of {BENEFIT_COUPLINGS}"
            )
        self._rng = np.random.default_rng(self.config.seed)
        self.truth: dict[str, TaskTruth] = {}
        self._generate()

    # -- generation ---------------------------------------------------------
    def _generate(self) -> None:
        cfg = self.config
        rng = self._rng
        modes = sorted(TAXONOMY)[: cfg.n_failure_modes]
        domain_effect = rng.normal(0, cfg.domain_effect_sd, cfg.n_domains)

        raw_recov = []
        entries = []
        for i in range(cfg.n_tasks):
            d = i % cfg.n_domains
            q = float(np.clip(
                rng.beta(*cfg.capability_prior) * float(np.exp(domain_effect[d]) /
                                                        (1 + np.exp(domain_effect[d])) * 2),
                0.001, 0.999,
            ))
            r = float(np.clip(rng.beta(*cfg.execution_prior), 0.001, 0.999))
            p = q * r

            # True recoverability: capable but unreliable. Defined on the LATENT
            # quantities, which is what the estimators are trying to recover.
            true_recov = float(
                (p > cfg.tau_cap) * np.clip(cfg.tau_rel - p, 0.0, None)
                / (cfg.tau_rel - cfg.tau_cap)
            )
            raw_recov.append(true_recov)

            # Failure-mode distribution, with a per-task concentration.
            conc = float(rng.beta(cfg.failure_concentration_a, cfg.failure_concentration_a))
            alpha = np.full(len(modes), 0.05 + 3.0 * (1.0 - conc))
            alpha[rng.integers(len(modes))] += 8.0 * conc
            probs = rng.dirichlet(alpha)
            ent = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))

            entries.append(
                {
                    "task": f"task_{i:04d}",
                    "domain": f"domain_{d}",
                    "q": q, "r": r, "p": p,
                    "failure_modes": list(modes),
                    "failure_probs": probs.tolist(),
                    "failure_entropy": ent,
                    "horizon": int(rng.integers(*cfg.horizon_range)),
                    "true_label": _true_label(p, cfg),
                }
            )

        benefit = self._make_benefit(np.asarray(raw_recov), entries, rng)
        for e, g in zip(entries, benefit):
            t = TaskTruth(benefit=float(g), **e)
            self.truth[t.task] = t

    def _make_benefit(self, recov: np.ndarray, entries, rng) -> np.ndarray:
        """Generate g_t under the configured coupling. See the module docstring."""
        cfg = self.config
        noise = rng.normal(0, cfg.benefit_noise, recov.size)
        concentration = 1.0 - np.array(
            [e["failure_entropy"] for e in entries]
        ) / max(np.log(cfg.n_failure_modes), 1e-9)

        if cfg.benefit_coupling == "strong":
            g = 0.75 * recov + 0.25 * concentration
        elif cfg.benefit_coupling == "weak":
            g = 0.35 * recov + 0.15 * concentration + 0.5 * rng.random(recov.size)
        elif cfg.benefit_coupling == "none":
            g = rng.random(recov.size)
        else:  # adversarial: the least recoverable tasks are the valuable ones
            g = 1.0 - recov
        return np.clip(g + noise, 0.0, None)

    # -- sampling -----------------------------------------------------------
    def run(self, task: str, rng: Optional[np.random.Generator] = None) -> dict:
        """Simulate one run: milestone, outcome, failure mode, horizon, cost."""
        rng = rng or self._rng
        t = self.truth[task]
        reached = bool(rng.random() < t.q)
        success = bool(reached and rng.random() < t.r)
        if (
            not reached
            and self.config.necessity_violation_rate > 0
            and rng.random() < self.config.necessity_violation_rate
        ):
            success = True  # deliberately breaks the decomposition's assumption
        mode = None
        if not success:
            mode = str(rng.choice(t.failure_modes, p=t.failure_probs))
        n_steps = max(2, int(rng.poisson(t.horizon)))
        return {
            "task": task,
            "domain": t.domain,
            "milestone": reached,
            "success": success,
            "score": 1.0 if success else 0.0,
            "failure_mode": mode,
            "n_steps": n_steps,
            "cost_usd": float(0.02 * n_steps * rng.gamma(4, 0.25)),
        }

    def scores(self, task: str, n: int, rng=None) -> list[float]:
        """``n`` simulated scores for a task. The estimators' only input."""
        rng = rng or self._rng
        return [self.run(task, rng)["score"] for _ in range(n)]

    def trajectory(
        self, task: str, run_index: int = 0, rng: Optional[np.random.Generator] = None
    ) -> Trajectory:
        """Simulate a full trajectory whose structure reflects the run's outcome.

        Successful runs follow a canonical path. Failed runs follow that path to a
        divergence point determined by the sampled failure mode's pipeline stage,
        then deviate -- so trajectory-derived signals (divergence location,
        intervention distance, embedding neighbourhood) have something real to
        recover rather than noise dressed as structure.
        """
        rng = rng or self._rng
        out = self.run(task, rng)
        t = self.truth[task]
        n = out["n_steps"]
        canonical = ["plan", "search", "read", "compute", "verify", "write"]

        events: list[TrajectoryEvent] = []
        divergence_at = n
        if not out["success"]:
            from ..diagnosis.taxonomy import stage_of

            stage = stage_of(out["failure_mode"] or "")
            frac = np.clip((stage + 1) / 11.0, 0.1, 0.9)
            divergence_at = max(1, int(n * frac))

        for i in range(n):
            tool = canonical[i % len(canonical)]
            diverged = i >= divergence_at
            target = f"{task}/doc_{i % 3}" if not diverged else f"{task}/wrong_{i % 3}"
            events.append(
                TrajectoryEvent(
                    index=i,
                    event_type=(
                        EventType.ERROR if (diverged and i == divergence_at)
                        else EventType.VERIFICATION if tool == "verify"
                        else EventType.RETRIEVAL if tool in ("search", "read")
                        else EventType.TOOL_CALL
                    ),
                    tool_name=tool,
                    tool_args={"step": i, "mode": "alt" if diverged else "std"},
                    target=target,
                    ok=(not diverged) or (i != divergence_at),
                    observation=f"{tool} result {i}",
                    state_features={"progress": (i + 1) / n * (0.4 if diverged else 1.0)},
                    cost={"usd": out["cost_usd"] / n},
                )
            )
        traj = Trajectory(
            trajectory_id=f"{task}#{run_index}",
            task=task,
            model="simulated-agent",
            events=events,
            score=out["score"],
            success=out["success"],
            episode=run_index,
            domain=t.domain,
            cost={"usd": out["cost_usd"]},
            meta={"failure_mode": out["failure_mode"], "milestone": out["milestone"]},
        )
        traj.rubric_scores = {"milestone": float(out["milestone"])}
        return traj

    # -- convenience --------------------------------------------------------
    def tasks(self) -> list[str]:
        return list(self.truth)

    def true_frame(self):
        import pandas as pd

        return pd.DataFrame([t.to_dict() for t in self.truth.values()])

    def true_benefit(self) -> dict[str, float]:
        return {k: v.benefit for k, v in self.truth.items()}

    def true_recoverability(self) -> dict[str, float]:
        cfg = self.config
        return {
            k: float(
                (v.p > cfg.tau_cap) * np.clip(cfg.tau_rel - v.p, 0.0, None)
                / (cfg.tau_rel - cfg.tau_cap)
            )
            for k, v in self.truth.items()
        }
