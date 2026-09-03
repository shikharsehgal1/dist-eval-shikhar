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
from .hierarchical import (
    HierarchicalFit,
    HierarchicalSpec,
    LogitNormalPosterior,
    fit_hierarchical,
)

__all__ = [
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
