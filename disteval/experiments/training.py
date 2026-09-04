"""Trainer backends: from selected pairs to an updated policy.

This repository is an evaluation and diagnosis framework; it does not implement
gradient steps. What it must do is make the training step *pluggable* and make
the simulated one honest enough to be worth running.

Backends
--------
``ExportOnlyBackend``
    Writes the DPO/ranking datasets and stops. The right answer when the actual
    training happens elsewhere, which is the common case.

``SimulatedBackend``
    Applies an analytic response model so the full loop can be exercised and the
    selection experiment can be run thousands of times without a GPU.

    **The critical design constraint**: the improvement a task receives is
    determined by that task's *true* latent training benefit ``g_t`` -- supplied
    from the simulator's ground truth -- and never by the score the selector used
    to pick it. If the response model read the recoverability score, then
    "recoverability selection improves the policy" would be true by construction
    and the experiment would be circular. Transfer to *other* tasks is governed
    by an explicit, configurable ``transfer`` kernel over domains, so that
    held-out performance is not simply a copy of training performance.

``TRLBackend`` / ``AxolotlBackend``
    Thin adapters over the existing :mod:`disteval.training_harness` reference
    trainers, preserved for backwards compatibility. They write a dataset and a
    config; they do not silently claim to have trained anything, and the existing
    warning about placeholder scores is retained.

Every backend returns a :class:`TrainingResult` recording what it actually did,
including whether any real optimisation happened, so a report can never present
a simulated number as a measured one.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from ..selection.pairs import PreferenceDataset, to_dpo_jsonl, to_ranking_jsonl

__all__ = [
    "TrainingResult",
    "TrainerBackend",
    "ExportOnlyBackend",
    "SimulatedBackend",
    "TRLBackend",
    "AxolotlBackend",
    "BACKENDS",
    "make_backend",
]


@dataclass
class TrainingResult:
    """What a backend did, and what it is and is not claiming."""

    backend: str
    n_pairs: int
    #: Per-task multiplicative change applied to the latent success probability.
    #: Empty for backends that did not train.
    task_effects: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    #: False for export-only and reference backends. Reports MUST surface this.
    is_real_training: bool = False
    #: True when the numbers come from an analytic model rather than measurement.
    is_simulated: bool = False
    note: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "n_pairs": self.n_pairs,
            "n_tasks_affected": len(self.task_effects),
            "is_real_training": self.is_real_training,
            "is_simulated": self.is_simulated,
            "note": self.note,
            "artifacts": dict(self.artifacts),
        }


class TrainerBackend(ABC):
    """Plugin interface for Phase C."""

    name = "backend"

    @abstractmethod
    def train(self, dataset: PreferenceDataset, output_dir: str, **kwargs) -> TrainingResult:
        ...

    def _export(self, dataset: PreferenceDataset, output_dir: str) -> dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dpo = out / "preference_pairs.jsonl"
        rank = out / "trajectory_ranking.jsonl"
        to_dpo_jsonl(dataset, str(dpo))
        to_ranking_jsonl(dataset, str(rank))
        return {"dpo_dataset": str(dpo), "ranking_dataset": str(rank)}


class ExportOnlyBackend(TrainerBackend):
    """Writes datasets and stops. Honest about doing nothing else."""

    name = "none"

    def train(self, dataset, output_dir, **kwargs) -> TrainingResult:
        return TrainingResult(
            backend=self.name,
            n_pairs=len(dataset),
            artifacts=self._export(dataset, output_dir),
            is_real_training=False,
            note="datasets exported; no training was performed",
        )


class SimulatedBackend(TrainerBackend):
    """Analytic response model driven by ground-truth training benefit.

    The model, stated explicitly so it can be argued with:

        delta_logit(t) = gain * g_t * saturate(pairs_t) + transfer_term(t)
        saturate(m)    = m / (m + half_pairs)

    * ``g_t`` is the task's **true** latent benefit, supplied by the caller from
      simulator ground truth. It is never the selector's score. This is what
      keeps the selection experiment non-circular.
    * ``saturate`` gives diminishing returns in the number of pairs from a task,
      which is why ``max_pairs_per_task`` matters and why a strategy that piles
      every pair onto one task does worse than one that spreads them.
    * ``transfer_term`` gives untrained tasks in a trained domain a fraction
      ``transfer_within_domain`` of the mean effect, and tasks in other domains
      ``transfer_cross_domain``. Both default well below 1, so held-out
      performance is a genuinely weaker signal than training performance rather
      than a copy of it.
    * ``regression`` optionally subtracts a small amount from tasks that received
      no data, modelling the interference that real preference training exhibits.
      Off by default; turn it on to test whether a selection method's advantage
      survives it.
    """

    name = "simulated"

    def __init__(
        self,
        true_benefit: Mapping[str, float],
        domains: Optional[Mapping[str, str]] = None,
        *,
        gain: float = 1.5,
        half_pairs: float = 3.0,
        transfer_within_domain: float = 0.35,
        transfer_cross_domain: float = 0.05,
        regression: float = 0.0,
        noise: float = 0.05,
        seed: int = 0,
    ):
        self.true_benefit = dict(true_benefit)
        self.domains = dict(domains or {})
        self.gain = gain
        self.half_pairs = half_pairs
        self.transfer_within = transfer_within_domain
        self.transfer_cross = transfer_cross_domain
        self.regression = regression
        self.noise = noise
        self.seed = seed

    def train(self, dataset, output_dir, *, all_tasks: Optional[Sequence[str]] = None,
              **kwargs) -> TrainingResult:
        rng = np.random.default_rng(self.seed)
        pairs_per_task: dict[str, int] = {}
        for p in dataset.pairs:
            pairs_per_task[p.task_id] = pairs_per_task.get(p.task_id, 0) + 1

        direct: dict[str, float] = {}
        for task, m in pairs_per_task.items():
            g = float(self.true_benefit.get(task, 0.0))
            direct[task] = self.gain * g * (m / (m + self.half_pairs))

        trained_domains: dict[str, list[float]] = {}
        for task, eff in direct.items():
            trained_domains.setdefault(self.domains.get(task, "_all"), []).append(eff)
        domain_mean = {d: float(np.mean(v)) for d, v in trained_domains.items()}
        overall_mean = float(np.mean(list(direct.values()))) if direct else 0.0

        universe = list(all_tasks) if all_tasks else list(
            set(self.true_benefit) | set(direct)
        )
        effects: dict[str, float] = {}
        for task in universe:
            if task in direct:
                e = direct[task]
            else:
                d = self.domains.get(task, "_all")
                if d in domain_mean:
                    e = self.transfer_within * domain_mean[d]
                else:
                    e = self.transfer_cross * overall_mean
                e -= self.regression
            effects[task] = float(e + rng.normal(0, self.noise))

        return TrainingResult(
            backend=self.name,
            n_pairs=len(dataset),
            task_effects=effects,
            artifacts=self._export(dataset, output_dir),
            is_real_training=False,
            is_simulated=True,
            note=(
                "effects come from an analytic response model driven by the "
                "simulator's TRUE per-task benefit, not by the selector's score; "
                "these are not measured training results"
            ),
            meta={
                "gain": self.gain,
                "half_pairs": self.half_pairs,
                "transfer_within_domain": self.transfer_within,
                "transfer_cross_domain": self.transfer_cross,
                "regression": self.regression,
                "n_tasks_directly_trained": len(direct),
            },
        )


class _ReferenceBackend(TrainerBackend):
    """Adapter over the legacy :mod:`disteval.training_harness` reference trainers."""

    trainer_cls_name = ""

    def __init__(self, base_model: str = "model", **kwargs):
        self.base_model = base_model
        self.kwargs = kwargs

    def train(self, dataset, output_dir, **kwargs) -> TrainingResult:
        from .. import training_harness

        cls = getattr(training_harness, self.trainer_cls_name)
        trainer = cls(self.base_model)
        artifacts = self._export(dataset, output_dir)
        curriculum = {
            "curriculum": [
                {
                    "task": p.task_id,
                    "training_pairs": [
                        {
                            "chosen_trajectory_path": p.metadata.get("chosen_id", ""),
                            "rejected_trajectory_path": p.metadata.get("rejected_id", ""),
                            "chosen_score": p.chosen_score,
                            "rejected_score": p.rejected_score,
                        }
                    ],
                }
                for p in dataset.pairs
            ]
        }
        scores = trainer.train(curriculum, output_dir)
        return TrainingResult(
            backend=self.name,
            n_pairs=len(dataset),
            task_effects={},
            artifacts=artifacts,
            is_real_training=False,
            note=(
                f"{self.trainer_cls_name} is a reference skeleton: it wrote a "
                "dataset and a trainer config but ran no optimisation. The scores "
                "it returns are placeholders and must not be reported as results."
            ),
            meta={"placeholder_scores": scores},
        )


class TRLBackend(_ReferenceBackend):
    name = "trl"
    trainer_cls_name = "TRLReferenceTrainer"


class AxolotlBackend(_ReferenceBackend):
    name = "axolotl"
    trainer_cls_name = "AxolotlReferenceTrainer"


BACKENDS = {
    "none": ExportOnlyBackend,
    "simulated": SimulatedBackend,
    "trl": TRLBackend,
    "axolotl": AxolotlBackend,
}


def make_backend(name: str, **kwargs) -> TrainerBackend:
    if name not in BACKENDS:
        raise ValueError(f"unknown training backend {name!r}; have {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)
