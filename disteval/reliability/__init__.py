"""Latent reliability estimation: posteriors, hierarchical pooling, classification.

This subpackage is the statistical core of the framework. Everything downstream
(classification, recoverability, data selection, active evaluation) is a
functional of the posteriors produced here.
"""
from .posterior import (
    JEFFREYS_PRIOR,
    UNIFORM_PRIOR,
    BetaPrior,
    TaskPosterior,
    binary_posterior,
    continuous_posterior,
    dispersion_ratio,
    posterior_from_scores,
)

from .classify import (
    CATEGORIES,
    RECOVERABLE,
    SOLID,
    STUCK,
    UNCERTAIN,
    ReliabilityThresholds,
    TaskDiagnosis,
    diagnose,
    diagnose_many,
    evidence_weighted_gap,
    expected_headroom,
    posterior_gap,
    rank_by_recoverability,
)
from .decomposition import (
    Decomposition,
    decompose,
    milestone_from_events,
    milestone_from_rubric,
    milestone_from_score,
)
from .rubric import (
    CriterionProfile,
    RubricProfile,
    criterion_posteriors,
    profile_rubric,
    root_criterion_profile,
    rubric_matrix,
)
from .recoverability import (
    SIGNAL_NAMES,
    LearnedRecoverability,
    RecoverabilitySignals,
    RecoverabilityWeights,
    combine_weighted,
    compare_signals,
    score_tasks,
)
from .hierarchical import (
    HierarchicalFit,
    HierarchicalSpec,
    LogitNormalPosterior,
    fit_hierarchical,
)

__all__ = [
    "Decomposition",
    "decompose",
    "milestone_from_rubric",
    "milestone_from_events",
    "milestone_from_score",
    "CriterionProfile",
    "RubricProfile",
    "criterion_posteriors",
    "profile_rubric",
    "root_criterion_profile",
    "rubric_matrix",
    "RecoverabilitySignals",
    "RecoverabilityWeights",
    "SIGNAL_NAMES",
    "combine_weighted",
    "score_tasks",
    "LearnedRecoverability",
    "compare_signals",
    "ReliabilityThresholds",
    "TaskDiagnosis",
    "diagnose",
    "diagnose_many",
    "rank_by_recoverability",
    "posterior_gap",
    "expected_headroom",
    "evidence_weighted_gap",
    "SOLID",
    "RECOVERABLE",
    "STUCK",
    "UNCERTAIN",
    "CATEGORIES",
    "HierarchicalSpec",
    "HierarchicalFit",
    "LogitNormalPosterior",
    "fit_hierarchical",
    "BetaPrior",
    "TaskPosterior",
    "JEFFREYS_PRIOR",
    "UNIFORM_PRIOR",
    "binary_posterior",
    "continuous_posterior",
    "posterior_from_scores",
    "dispersion_ratio",
]
