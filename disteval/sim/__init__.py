"""Simulation with known ground truth: validating the estimators themselves."""
from .studies import (
    adaptive_saving_study,
    classification_study,
    decomposition_study,
    max_bias_study,
    posterior_recovery_study,
    run_all_studies,
    runs_needed_study,
    selection_study,
    shrinkage_study,
)
from .world import BENEFIT_COUPLINGS, SimulatedWorld, TaskTruth, WorldConfig

__all__ = [
    "SimulatedWorld", "WorldConfig", "TaskTruth", "BENEFIT_COUPLINGS",
    "max_bias_study", "posterior_recovery_study", "shrinkage_study",
    "classification_study", "runs_needed_study", "adaptive_saving_study",
    "decomposition_study", "selection_study", "run_all_studies",
]
