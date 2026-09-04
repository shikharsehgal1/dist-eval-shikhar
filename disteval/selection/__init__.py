"""Training-data selection: which tasks, and which trajectory pairs from them."""
from .pairs import (
    PairConfig,
    PreferenceDataset,
    PreferencePair,
    build_dataset,
    build_pairs_for_task,
    to_dpo_jsonl,
    to_ranking_jsonl,
)
from .selectors import (
    SELECTORS,
    DataSelector,
    HardestSelector,
    HighestVarianceSelector,
    LowestMeanSelector,
    OracleSelector,
    RandomSelector,
    RecoverabilitySelector,
    SelectionResult,
    SuccessFailureSelector,
    UncertaintyAwareSelector,
    make_selector,
)

__all__ = [
    "SelectionResult", "DataSelector", "RandomSelector", "HardestSelector",
    "LowestMeanSelector", "HighestVarianceSelector", "SuccessFailureSelector",
    "RecoverabilitySelector", "UncertaintyAwareSelector", "OracleSelector",
    "SELECTORS", "make_selector",
    "PairConfig", "PreferencePair", "PreferenceDataset",
    "build_pairs_for_task", "build_dataset", "to_dpo_jsonl", "to_ranking_jsonl",
]
