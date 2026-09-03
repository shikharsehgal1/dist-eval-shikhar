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

from .hierarchical import (
    HierarchicalFit,
    HierarchicalSpec,
    LogitNormalPosterior,
    fit_hierarchical,
)

__all__ = [
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
