"""disteval -- distributional evaluation and reliability-aware post-training for agents.

The question this package exists to answer is not "how good is this agent" but
**"where has it demonstrated a capability it cannot reproduce, and is that a
useful place to train?"**

Layers, roughly in dependency order:

``records``, ``loaders``
    Per-episode storage and the generic ``load_runs`` / ``load_tasks`` /
    ``load_trajectories`` entry points. Structurally compatible with the data a
    long-horizon professional-agent benchmark produces.

``metrics``, ``metrics_spec``
    Distribution-aware aggregates (IQM, lower-tail CVaR, pass@k vs pass^k) and a
    registry declaring every metric's definition, assumptions and edge cases.

``reliability``
    The statistical core: Beta-Binomial and hierarchical posteriors over latent
    task performance, the SOLID/RECOVERABLE/STUCK/UNCERTAIN taxonomy defined on
    those posteriors, recoverability estimators, per-criterion rubric reliability,
    the identifiable capability/execution split, IID diagnostics, horizon scaling,
    cost-aware metrics, and paired model comparison.

``trajectory``
    Canonical event logs, alignment (Needleman-Wunsch / DTW), divergence
    analysis, embeddings, and counterfactual intervention distance.

``diagnosis``
    Failure taxonomy, causality graphs, failure entropy, survival analysis.

``active``
    Value of information, adaptive allocation, sequential stopping.

``selection``
    Which tasks to train on, and the matched preference pairs to build from them.

``experiments``
    Config, tracking, splits, the A-D pipeline, and ablation sweeps.

``taskgen``
    Metamorphic eval-task generation: variants of seed tasks whose verifier
    transfers by construction, aimed at measured criterion weaknesses.

``sim``
    A ground-truth simulator and the validation suite for the estimators
    themselves.

``plots``, ``research_report``
    Figures and the end-to-end Markdown/HTML report.

Legacy modules (``right_tail``, ``self_engine``, ``training_sim``, ...) are
retained and still work; ``right_tail`` in particular is superseded by
``reliability`` and says so.
"""
__version__ = "0.2.0"

from . import bootstrap, compare, failure, metrics, metrics_spec, repeat, right_tail
from . import trajectory_monitor, trajectory_memory
from . import self_engine
from . import recursion_engine, environment_generator, environment_registry, distributed_eval
from . import logging, training_harness, agent_harness
from . import bayesian_optimization, curriculum_optimizer
from . import irt, ppi, shrinkage, evt, best_arm
from . import adapters

# Reliability-analysis layers.
from . import active, diagnosis, experiments, loaders, plots, reliability
from . import research_report, selection, sim, taskgen, trajectory

from .loaders import TaskSpec, load_runs, load_tasks, load_trajectories
from .records import EpisodeRecord, RecordStore
from .reliability import (
    RECOVERABLE,
    SOLID,
    STUCK,
    UNCERTAIN,
    ReliabilityThresholds,
    TaskDiagnosis,
    TaskPosterior,
    diagnose,
    fit_hierarchical,
    posterior_from_scores,
    rank_by_recoverability,
)
from .research_report import generate_report

__all__ = [
    "__version__",
    # storage and loading
    "EpisodeRecord", "RecordStore", "TaskSpec",
    "load_runs", "load_tasks", "load_trajectories",
    # headline API
    "posterior_from_scores", "fit_hierarchical", "diagnose",
    "rank_by_recoverability", "generate_report",
    "TaskPosterior", "TaskDiagnosis", "ReliabilityThresholds",
    "SOLID", "RECOVERABLE", "STUCK", "UNCERTAIN",
    # subpackages
    "reliability", "trajectory", "diagnosis", "active", "selection",
    "experiments", "sim", "taskgen", "plots", "research_report", "loaders",
    "metrics", "metrics_spec", "adapters",
    # pre-existing modules
    "bootstrap", "compare", "failure", "repeat", "right_tail",
    "trajectory_monitor", "trajectory_memory", "self_engine",
    "recursion_engine", "environment_generator", "environment_registry",
    "distributed_eval", "logging", "training_harness", "agent_harness",
    "bayesian_optimization", "curriculum_optimizer",
    "irt", "ppi", "shrinkage", "evt", "best_arm",
]
