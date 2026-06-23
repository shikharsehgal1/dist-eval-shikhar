"""disteval.training_harness — bridge self-improvement curricula to DPO trainers.

The SelfEngine produces a ``SelfImprovementPlan``: a ranked list of RECOVERABLE
tasks with (reinforce, contrast) trajectory pairs. This module defines the
contract for converting that plan into a trained policy and ships a small set
of reference implementations.

Design principles:
- disteval stays an evaluation framework; it does not implement gradient steps.
- The harness is a thin adapter layer that knows how to read the plan format
  and call an external trainer (TRL, Axolotl, etc.).
- Heavy dependencies are optional imports so the core package stays light.

A minimal trainer must implement ``DPOTrainerBase.train(curriculum, output_dir)
and return a dict of {task: improved_score}.
"""
from __future__ import annotations

import importlib.util
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DPOExample:
    """One DPO-style (chosen, rejected) example."""
    task: str
    chosen_trajectory_path: str
    rejected_trajectory_path: str
    chosen_score: float
    rejected_score: float


class DPOTrainerBase(ABC):
    """Abstract base for any trainer that consumes a disteval curriculum."""

    @abstractmethod
    def train(self, curriculum: Any, output_dir: str) -> dict[str, float]:
        """Train on ``curriculum`` and return a map of task -> improved score.

        ``curriculum`` is expected to be a ``SelfImprovementPlan`` or a plain
        dict with the same structure (see ``self_engine.SelfImprovementPlan``).
        """
        ...

    def _plan_to_dpo_examples(self, curriculum: Any) -> list[DPOExample]:
        """Extract DPO examples from a plan/dict."""
        items = getattr(curriculum, "curriculum", None)
        if items is None:
            items = curriculum.get("curriculum", [])
        examples: list[DPOExample] = []
        for item in items:
            pairs = getattr(item, "training_pairs", None)
            if pairs is None:
                pairs = item.get("training_pairs", [])
            for pair in pairs:
                chosen = getattr(pair, "reinforce_traj_path", None)
                rejected = getattr(pair, "contrast_traj_path", None)
                if chosen is None:
                    chosen = pair.get("reinforce_traj_path")
                    rejected = pair.get("contrast_traj_path")
                if chosen and rejected:
                    examples.append(
                        DPOExample(
                            task=getattr(item, "task", item.get("task", "")),
                            chosen_trajectory_path=chosen,
                            rejected_trajectory_path=rejected,
                            chosen_score=getattr(
                                pair, "reinforce_score", pair.get("reinforce_score", 0.0)
                            ),
                            rejected_score=getattr(
                                pair, "contrast_score", pair.get("contrast_score", 0.0)
                            ),
                        )
                    )
        return examples

    def _write_dataset(self, examples: list[DPOExample], path: str) -> None:
        """Write a JSONL DPO dataset of trajectory references."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(
                    json.dumps(
                        {
                            "task": ex.task,
                            "chosen": ex.chosen_trajectory_path,
                            "rejected": ex.rejected_trajectory_path,
                            "chosen_score": ex.chosen_score,
                            "rejected_score": ex.rejected_score,
                        }
                    )
                    + "\n"
                )


class NoOpTrainer(DPOTrainerBase):
    """Trainer that does nothing but validate the harness contract.

    Useful for integration tests and CI where heavy ML dependencies are not
    installed.
    """

    def __init__(self, improved_score: float = 0.99):
        self.improved_score = improved_score

    def train(self, curriculum: Any, output_dir: str) -> dict[str, float]:
        """Return the configured score for every task in the curriculum."""
        os.makedirs(output_dir, exist_ok=True)
        examples = self._plan_to_dpo_examples(curriculum)
        self._write_dataset(examples, os.path.join(output_dir, "dpo_dataset.jsonl"))
        items = getattr(curriculum, "curriculum", None)
        if items is None:
            items = curriculum.get("curriculum", [])
        return {item.task if hasattr(item, "task") else item.get("task"): self.improved_score for item in items}


class SimulatedTrainer(DPOTrainerBase):
    """Trainer that simulates improvement based on the curriculum itself.

    Returns a score that is the current q_star plus a fraction of the predicted
gain, capped at 1.0. Useful for end-to-end pipeline tests without real
    training.
    """

    def train(self, curriculum: Any, output_dir: str) -> dict[str, float]:
        """Simulate improved scores from the plan's predicted gains."""
        os.makedirs(output_dir, exist_ok=True)
        examples = self._plan_to_dpo_examples(curriculum)
        self._write_dataset(examples, os.path.join(output_dir, "dpo_dataset.jsonl"))
        items = getattr(curriculum, "curriculum", None)
        if items is None:
            items = curriculum.get("curriculum", [])
        improved: dict[str, float] = {}
        for item in items:
            task = item.task if hasattr(item, "task") else item.get("task")
            q_star = item.current_q_star if hasattr(item, "current_q_star") else item.get("current_q_star", 0.0)
            gain = item.predicted_gain if hasattr(item, "predicted_gain") else item.get("predicted_gain")
            if gain is None:
                gain = 0.05
            improved[task] = min(1.0, q_star + gain)
        return improved


class TRLReferenceTrainer(DPOTrainerBase):
    """Reference adapter for HuggingFace TRL's ``DPOTrainer``.

    This class demonstrates the exact contract. TRL is an optional dependency;
    if it is not installed, ``__init__`` raises ImportError with a helpful
    message.
    """

    def __init__(
        self,
        model_name: str,
        learning_rate: float = 5e-7,
        beta: float = 0.1,
        num_train_epochs: int = 1,
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.beta = beta
        self.num_train_epochs = num_train_epochs

    def train(self, curriculum: Any, output_dir: str) -> dict[str, float]:
        """Build a DPO dataset from the curriculum and delegate to TRL.

        This is a reference skeleton. A production implementation must load the
        trajectories, convert them into a ``datasets.Dataset`` with
        ``chosen`` / ``rejected`` conversation fields, instantiate the reference
        model + tokenizer, and run ``DPOTrainer.train()``.
        """
        if importlib.util.find_spec("trl") is None:
            raise ImportError(
                "TRLReferenceTrainer requires `trl` to be installed. "
                "Install with: pip install trl"
            )

        os.makedirs(output_dir, exist_ok=True)
        examples = self._plan_to_dpo_examples(curriculum)
        self._write_dataset(examples, os.path.join(output_dir, "dpo_dataset.jsonl"))

        # Reference skeleton: return a not-implemented score map.
        # A real implementation would instantiate the trainer here.
        return {
            ex.task: 1.0
            for ex in examples
        }


class AxolotlReferenceTrainer(DPOTrainerBase):
    """Reference adapter for Axolotl YAML-based DPO training.

    Axolotl is configured via YAML and launched with ``axolotl.cli.train``.
    This adapter writes the DPO dataset and a generated config file, then
    returns a stub result map. TRL is not required.
    """

    def __init__(self, base_model: str, template: str = "llama3"):
        self.base_model = base_model
        self.template = template

    def train(self, curriculum: Any, output_dir: str) -> dict[str, float]:
        os.makedirs(output_dir, exist_ok=True)
        examples = self._plan_to_dpo_examples(curriculum)
        dataset_path = os.path.join(output_dir, "dpo_dataset.jsonl")
        self._write_dataset(examples, dataset_path)

        # Write a minimal Axolotl YAML config file for reference.
        config_path = os.path.join(output_dir, "axolotl_config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(
                f"""base_model: {self.base_model}
rl: dpo

datasets:
  - path: {Path(dataset_path).resolve()}
    type: json

output_dir: {output_dir}
sequence_len: 4096
num_epochs: 1
micro_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 5e-7
"""
            )

        return {ex.task: 1.0 for ex in examples}


def run_training(
    curriculum: Any,
    trainer: DPOTrainerBase,
    output_dir: str,
) -> dict[str, float]:
    """Convenience entrypoint: train with the given trainer and return scores."""
    os.makedirs(output_dir, exist_ok=True)
    return trainer.train(curriculum, output_dir)
