"""Failure diagnosis: taxonomy, causality graphs, failure entropy, survival analysis."""
from .causality import (
    CausalEdge,
    CriterionGraph,
    FailureGraph,
    aggregate_graphs,
    attribute_rubric_failures,
    build_failure_graph,
)
from .entropy import (
    FailureDistribution,
    dominant_mode_share,
    entropy_ci,
    failure_distribution,
    normalized_entropy,
)
from .survival import (
    SurvivalCurve,
    SurvivalRecord,
    hazard_by_step,
    kaplan_meier,
    recovery_probability,
    run_survival_record,
    survival_summary,
    time_to_first_error,
)
from .taxonomy import (
    TAXONOMY,
    FailureAttributor,
    FailureLabel,
    FailureMode,
    classify_events,
    classify_failure,
    register_mode,
)

__all__ = [
    "FailureMode", "TAXONOMY", "register_mode", "FailureLabel",
    "classify_failure", "classify_events", "FailureAttributor",
    "CausalEdge", "FailureGraph", "CriterionGraph", "build_failure_graph",
    "attribute_rubric_failures", "aggregate_graphs",
    "FailureDistribution", "failure_distribution", "normalized_entropy",
    "entropy_ci", "dominant_mode_share",
    "SurvivalRecord", "SurvivalCurve", "run_survival_record", "kaplan_meier",
    "hazard_by_step", "time_to_first_error", "recovery_probability",
    "survival_summary",
]
