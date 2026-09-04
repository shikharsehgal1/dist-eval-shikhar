"""Active evaluation: value of information, adaptive allocation, sequential stopping."""
from .allocation import (
    VALUE_CRITERIA,
    AllocationPolicy,
    AllocationResult,
    GreedyValuePolicy,
    StratifiedUniform,
    TaskState,
    ThompsonAllocation,
    UniformAllocation,
    make_policy,
)
from .evaluator import AdaptiveEvaluator, EvaluationTrace, compare_allocation
from .stopping import (
    AllOf,
    AnyOf,
    ConfidentLabel,
    IntervalWidth,
    MaxRuns,
    MinRuns,
    NoValue,
    StoppingLedger,
    StoppingRule,
    default_stopping,
)
from .value import (
    bald,
    expected_entropy_reduction,
    expected_runs_to_decide,
    flip_probability,
    label_of,
    rank_instability,
)

__all__ = [
    "TaskState", "AllocationPolicy", "AllocationResult", "UniformAllocation",
    "GreedyValuePolicy", "ThompsonAllocation", "StratifiedUniform",
    "VALUE_CRITERIA", "make_policy",
    "StoppingRule", "ConfidentLabel", "IntervalWidth", "MaxRuns", "MinRuns",
    "NoValue", "AnyOf", "AllOf", "StoppingLedger", "default_stopping",
    "expected_entropy_reduction", "bald", "flip_probability",
    "expected_runs_to_decide", "rank_instability", "label_of",
    "AdaptiveEvaluator", "EvaluationTrace", "compare_allocation",
]
