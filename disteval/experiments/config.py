"""Declarative experiment configuration.

Every experiment in this repository is defined by a YAML file, not by arguments
scattered across a call site. The reason is reproducibility with teeth: the exact
config is hashed into the experiment id and written into the result directory, so
a result can always be traced to the settings that produced it, and two configs
that differ anywhere produce different ids.

Example::

    experiment:
      name: recoverability_vs_baselines
      seed: 42

    evaluation:
      runs_per_task: 8
      adaptive_sampling: true
      allocation_policy: greedy_voi
      budget_per_task: 8

    reliability:
      model: hierarchical
      backend: laplace
      credible_level: 0.95
      tau_cap: 0.15
      tau_rel: 0.90
      confidence: 0.80

    selection:
      method: recoverability
      n_tasks: 40
      n_pairs: 100

    training:
      backend: simulated

    split:
      strategy: domain_holdout
      test_fraction: 0.3

Unknown keys raise rather than being ignored: a typo in a config that silently
does nothing is the most expensive kind of bug in an experimental framework.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

__all__ = [
    "ExperimentMeta",
    "EvaluationConfig",
    "ReliabilityConfig",
    "SelectionConfig",
    "TrainingConfig",
    "SplitConfig",
    "ReportConfig",
    "ExperimentConfig",
    "load_config",
    "save_config",
]


def _strict_from_dict(cls, data: dict, path: str = ""):
    """Build a flat dataclass from a mapping, raising on unknown keys.

    ``from __future__ import annotations`` makes ``field.type`` a string, so no
    nested-dataclass resolution is attempted here; the config is deliberately two
    levels deep and :meth:`ExperimentConfig.from_dict` dispatches sections by an
    explicit map instead of by introspecting annotations.
    """
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        where = f" in section {path!r}" if path else ""
        raise ValueError(
            f"unknown configuration key(s) {sorted(unknown)}{where}; "
            f"valid keys are {sorted(known)}"
        )
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ExperimentMeta:
    """Identity and reproducibility settings."""

    name: str = "experiment"
    description: str = ""
    seed: int = 42
    #: Number of independent repetitions. Anything below 3 cannot support a
    #: variance estimate, and the pipeline warns when it is set lower.
    n_seeds: int = 5
    tags: list[str] = field(default_factory=list)


@dataclass
class EvaluationConfig:
    """Phase A: how the agent is evaluated."""

    runs_per_task: int = 8
    adaptive_sampling: bool = False
    allocation_policy: str = "uniform"
    #: Total budget expressed per task; the adaptive loop spends it globally.
    budget_per_task: Optional[int] = None
    warmup_runs: int = 2
    stopping: bool = True
    max_runs_per_task: int = 24
    min_runs_per_task: int = 3
    success_threshold: float = 1.0
    #: Metadata key identifying correlated run clusters (seed, environment...).
    cluster_key: Optional[str] = None


@dataclass
class ReliabilityConfig:
    """Estimation settings."""

    #: "independent" (per-task Beta-Binomial) or "hierarchical" (partial pooling).
    model: str = "hierarchical"
    backend: str = "laplace"     # or "pg_gibbs"
    prior: str = "jeffreys"      # "jeffreys" | "uniform" | "empirical_bayes"
    credible_level: float = 0.95
    tau_cap: float = 0.15
    tau_rel: float = 0.90
    tau_stuck: float = 0.10
    confidence: float = 0.80
    recoverability_estimator: str = "headroom"
    use_trajectory_signals: bool = True
    n_mcmc_samples: int = 500


@dataclass
class SelectionConfig:
    """Phase B: which tasks and how many pairs."""

    #: A single strategy name; "all" for the six curriculum strategies (the
    #: usual research run); "baselines" for uniform/difficulty/uncertainty/
    #: learning_progress only; "extended" to add the legacy ablation arms.
    method: str = "all"
    n_tasks: int = 40
    n_pairs: int = 100
    max_pairs_per_task: int = 4
    min_margin: float = 0.5
    match_environment: bool = True
    restrict_to_recoverable: bool = False
    #: For the gap selectors: require posterior evidence that the task is ever
    #: fully solved. None disables the gate. See criterion.GapProfile.
    require_joint_capability: Optional[float] = None
    #: Weights for gap_plus_structure: score = G_t * (gap_weight + structure_weight * S_t).
    gap_weight: float = 0.5
    structure_weight: float = 0.5
    #: Dataset sizes for the sample-efficiency learning curve.
    dataset_sizes: list[int] = field(default_factory=lambda: [25, 50, 100, 250])


@dataclass
class TrainingConfig:
    """Phase C: how (or whether) the selected data is trained on."""

    #: "none" exports only; "simulated" uses the analytic response model;
    #: "trl"/"axolotl" hand off to an external trainer.
    backend: str = "simulated"
    base_model: str = ""
    learning_rate: float = 5e-6
    n_epochs: int = 1
    beta: float = 0.1
    lora: bool = True
    output_dir: str = ""


@dataclass
class SplitConfig:
    """Phase D: how held-out evaluation is constructed."""

    #: "task" | "domain_holdout" | "environment" | "tool_composition" | "rubric_type"
    strategy: str = "domain_holdout"
    test_fraction: float = 0.3
    #: Named held-out groups, when you want a specific split rather than a random one.
    holdout_groups: list[str] = field(default_factory=list)
    stratify: bool = True


@dataclass
class ReportConfig:
    """What to emit."""

    figures: bool = True
    figure_format: str = "pdf"
    html: bool = True
    markdown: bool = True
    parquet: bool = True


@dataclass
class ExperimentConfig:
    """The whole experiment. Hashable, serialisable, and strictly validated."""

    experiment: ExperimentMeta = field(default_factory=ExperimentMeta)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        """Stable 12-hex-char hash of the entire config.

        Order-independent (keys are sorted) so a reordered YAML file gives the
        same id, but sensitive to every value -- which is the point.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.blake2s(blob.encode(), digest_size=6).hexdigest()

    @property
    def experiment_id(self) -> str:
        """Human-readable, collision-resistant run id."""
        return f"{self.experiment.name}-{self.fingerprint}"

    def validate(self) -> list[str]:
        """Return a list of warnings. Hard errors raise; soft issues are surfaced."""
        warns = []
        if self.experiment.n_seeds < 3:
            warns.append(
                f"n_seeds={self.experiment.n_seeds}: fewer than 3 repetitions cannot "
                "support a variance estimate, so differences between strategies will "
                "not be distinguishable from noise"
            )
        if self.evaluation.runs_per_task < self.reliability_thresholds_min_runs():
            warns.append(
                f"runs_per_task={self.evaluation.runs_per_task} is below the "
                f"classifier's min_runs; every task will be UNCERTAIN"
            )
        if not (0 <= self.reliability.tau_stuck <= self.reliability.tau_cap
                < self.reliability.tau_rel <= 1):
            raise ValueError("thresholds must satisfy 0 <= tau_stuck <= tau_cap < tau_rel <= 1")
        if self.selection.n_pairs < 10:
            warns.append("n_pairs < 10: the training signal will be dominated by noise")
        if self.split.strategy == "task" :
            warns.append(
                "split.strategy='task' holds out individual tasks but keeps their "
                "domains in training, so improvements may reflect domain-specific "
                "adaptation rather than transferable policy improvement; "
                "'domain_holdout' is the stronger test"
            )
        return warns

    def reliability_thresholds_min_runs(self) -> int:
        return self.evaluation.min_runs_per_task

    def thresholds(self):
        """Build the classifier thresholds this config implies."""
        from ..reliability.classify import ReliabilityThresholds

        return ReliabilityThresholds(
            tau_cap=self.reliability.tau_cap,
            tau_rel=self.reliability.tau_rel,
            tau_stuck=self.reliability.tau_stuck,
            confidence=self.reliability.confidence,
            min_runs=self.evaluation.min_runs_per_task,
            credible_level=self.reliability.credible_level,
        )

    #: Explicit section -> dataclass map. See :func:`_strict_from_dict`.
    SECTIONS = {
        "experiment": ExperimentMeta,
        "evaluation": EvaluationConfig,
        "reliability": ReliabilityConfig,
        "selection": SelectionConfig,
        "training": TrainingConfig,
        "split": SplitConfig,
        "report": ReportConfig,
    }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        unknown = set(data) - set(cls.SECTIONS)
        if unknown:
            raise ValueError(
                f"unknown top-level configuration section(s) {sorted(unknown)}; "
                f"valid sections are {sorted(cls.SECTIONS)}"
            )
        kwargs = {}
        for name, typ in cls.SECTIONS.items():
            if data.get(name) is not None:
                kwargs[name] = _strict_from_dict(typ, data[name], name)
        return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and strictly validate a YAML (or JSON) experiment config."""
    import yaml

    p = Path(path)
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} does not contain a mapping at the top level")
    return ExperimentConfig.from_dict(data)


def save_config(config: ExperimentConfig, path: str | Path) -> Path:
    """Write the config to YAML. Called automatically into every result directory."""
    import yaml

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=True, default_flow_style=False)
    return p
