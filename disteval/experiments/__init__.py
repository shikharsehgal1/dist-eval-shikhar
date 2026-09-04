"""Reproducible experiments: config, tracking, splits, pipeline, sweeps."""
from .config import (
    EvaluationConfig,
    ExperimentConfig,
    ExperimentMeta,
    ReliabilityConfig,
    ReportConfig,
    SelectionConfig,
    SplitConfig,
    TrainingConfig,
    load_config,
    save_config,
)
from .pipeline import (
    ExperimentResult,
    PhaseAResult,
    StrategyOutcome,
    run_experiment,
    run_phase_a,
)
from .splits import SPLIT_STRATEGIES, Split, make_split
from .sweep import SweepSpec, aggregate_sweep, expand_sweep, load_sweep, run_sweep
from .tracking import ExperimentRun, environment_metadata
from .training import (
    BACKENDS,
    AxolotlBackend,
    ExportOnlyBackend,
    SimulatedBackend,
    TrainerBackend,
    TrainingResult,
    TRLBackend,
    make_backend,
)

__all__ = [
    "ExperimentConfig", "ExperimentMeta", "EvaluationConfig", "ReliabilityConfig",
    "SelectionConfig", "TrainingConfig", "SplitConfig", "ReportConfig",
    "load_config", "save_config",
    "ExperimentRun", "environment_metadata",
    "Split", "make_split", "SPLIT_STRATEGIES",
    "TrainerBackend", "TrainingResult", "ExportOnlyBackend", "SimulatedBackend",
    "TRLBackend", "AxolotlBackend", "BACKENDS", "make_backend",
    "PhaseAResult", "StrategyOutcome", "ExperimentResult", "run_experiment",
    "run_phase_a",
    "SweepSpec", "load_sweep", "expand_sweep", "run_sweep", "aggregate_sweep",
]
