"""Generating eval tasks by metamorphic variation of seed tasks.

What this engine does
---------------------
Given seed tasks whose grading you already trust, it produces variants by
applying transformations whose effect on the correct answer is known
(:mod:`disteval.taskgen.relations`), so the seed's verifier transfers. It aims
generation using what the framework has already measured: which rubric criteria
are unstable, which failure modes recur, and which tasks are near the learnable
frontier.

What it deliberately does not do
--------------------------------
It does not invent tasks from nothing. A task whose answer nobody knows is not an
eval task, and grading a generated task with the generator's own answer measures
agreement with the generator. Free-form synthesis needs an independent oracle --
for instance requiring several models to agree, as in the PROPEL setup -- and
that is a different, weaker guarantee than an inherited verifier.

It also does not train a generator. Learning a task-proposal policy against a
difficulty signal (Wolf et al., 2026) is a reasonable thing to want and is out of
scope here: it needs model internals and training infrastructure, and a gestural
version would be a fake.

The circularity guard
---------------------
The failure mode that makes self-generated evals untrustworthy is the evaluated
model shaping its own test distribution. Two structural defences:

1. **The transformations are structural**, not model-authored, so the variant
   distribution does not depend on the evaluated model's preferences. Where a
   model-backed relation *is* used, every affected task is flagged
   ``model_generated`` and :mod:`disteval.taskgen.validate` excludes those from
   headline metrics unless explicitly confirmed.
2. **Generator identity is recorded** on every task. When the generator and the
   evaluated agent share a model family, :func:`GenerationReport.warnings` says
   so, because that is a confound a reader must be told about rather than one the
   framework can remove.

Targeting
---------
:meth:`TaskGenerator.targets_from_gaps` turns criterion-level gap profiles into
generation targets: criteria with evidence of capability but not of reliability,
ranked by how much of the task's gap they carry. Relations that declare they
stress those criteria are then preferred. The point is that a variant probing a
criterion the agent is *already* known to be shaky on is far more informative
than a variant probing one it always satisfies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from .relations import RELATIONS, MetamorphicRelation, RelationKind, relations_stressing

__all__ = [
    "SeedTask",
    "GeneratedTask",
    "GenerationTarget",
    "GenerationReport",
    "TaskGenerator",
]


@dataclass
class SeedTask:
    """A task whose grading is already trusted, used as a generation source."""

    task_id: str
    payload: dict
    #: How the seed is graded. Inherited by every variant; without it the seed
    #: cannot be used, because the variant would not be gradeable either.
    verifier: Optional[str] = None
    domain: str = ""
    rubric_criteria: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("seed task needs a task_id")
        if not isinstance(self.payload, dict):
            raise TypeError("seed payload must be a dict")


@dataclass
class GeneratedTask:
    """A variant, with everything needed to grade it and to audit its origin."""

    task_id: str
    seed_id: str
    relation: str
    relation_kind: str
    payload: dict
    #: The seed's verifier, plus how to apply it under this relation.
    verifier: Optional[str]
    verifier_note: str
    evidence: dict
    domain: str = ""
    #: Criteria or failure modes this variant was generated to probe.
    targets: list[str] = field(default_factory=list)
    model_generated: bool = False
    preserves_difficulty: bool = True
    generator: str = ""
    validation: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return bool(self.validation.get("valid", False))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GenerationTarget:
    """A measured weakness to aim generation at."""

    task: str
    criterion: str
    #: Share of the task's capability--reliability gap carried by this criterion.
    gap_share: float
    capability: float
    reliability: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationReport:
    """A batch of variants plus the accounting needed to trust or discount it."""

    tasks: list[GeneratedTask]
    requested: int
    seeds_used: list[str]
    relations_used: dict[str, int] = field(default_factory=dict)
    targets: list[GenerationTarget] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    generator: str = ""
    evaluated_model: str = ""
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tasks)

    @property
    def valid_tasks(self) -> list[GeneratedTask]:
        return [t for t in self.tasks if t.is_valid]

    def summary(self) -> dict:
        return {
            "requested": self.requested,
            "generated": len(self.tasks),
            "valid": len(self.valid_tasks),
            "rejected": len(self.rejected),
            "n_seeds": len(self.seeds_used),
            "n_relations": len(self.relations_used),
            "relations_used": dict(self.relations_used),
            "model_generated": sum(1 for t in self.tasks if t.model_generated),
            "generator": self.generator,
            "warnings": list(self.warnings),
        }

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame([
            {
                "task_id": t.task_id, "seed_id": t.seed_id, "relation": t.relation,
                "kind": t.relation_kind, "domain": t.domain,
                "targets": ",".join(t.targets), "valid": t.is_valid,
                "model_generated": t.model_generated,
                "preserves_difficulty": t.preserves_difficulty,
            }
            for t in self.tasks
        ])


class TaskGenerator:
    """Produces metamorphic variants of seed tasks, aimed at measured weaknesses."""

    def __init__(
        self,
        seeds: Sequence[SeedTask],
        *,
        relations: Optional[Sequence[MetamorphicRelation]] = None,
        generator_id: str = "structural",
        evaluated_model: str = "",
        require_verifier: bool = True,
        seed: int = 0,
    ):
        seeds = list(seeds)
        if not seeds:
            raise ValueError("no seed tasks supplied")
        missing = [s.task_id for s in seeds if require_verifier and not s.verifier]
        if missing:
            raise ValueError(
                f"seed task(s) {missing[:5]} have no verifier. A variant inherits "
                "its seed's verifier; without one the generated task cannot be "
                "graded and is not an eval task. Pass require_verifier=False only "
                "for a dry run."
            )
        self.seeds = {s.task_id: s for s in seeds}
        self.relations = list(relations) if relations is not None else list(RELATIONS.values())
        self.generator_id = generator_id
        self.evaluated_model = evaluated_model
        self._rng = np.random.default_rng(seed)

    # -- targeting ----------------------------------------------------------
    @staticmethod
    def targets_from_gaps(
        gap_profiles: Iterable,
        *,
        min_gap_share: float = 0.15,
        max_per_task: int = 3,
    ) -> list[GenerationTarget]:
        """Turn criterion-level gap profiles into generation targets.

        Selects criteria carrying a meaningful share of their task's
        capability--reliability gap: the agent has shown it can satisfy them but
        does not do so dependably. Those are the criteria where a variant is most
        likely to tell you something a rerun of the seed would not.
        """
        out: list[GenerationTarget] = []
        for prof in gap_profiles:
            crits = sorted(prof.criteria, key=lambda c: -c.gap)
            total = sum(max(c.gap, 0.0) for c in crits)
            if total <= 1e-12:
                continue
            for c in crits[:max_per_task]:
                share = c.gap / total
                if share >= min_gap_share:
                    out.append(GenerationTarget(
                        task=prof.task, criterion=c.criterion, gap_share=float(share),
                        capability=float(c.capability), reliability=float(c.reliability),
                    ))
        return sorted(out, key=lambda t: -t.gap_share)

    # -- generation ---------------------------------------------------------
    def _pick_relations(self, targets: Sequence[str]) -> list[MetamorphicRelation]:
        if not targets:
            return list(self.relations)
        preferred = [r for r in relations_stressing(*targets) if r in self.relations]
        # Keep the rest available but ranked lower: restricting entirely to
        # target-matching relations collapses diversity, which is the failure mode
        # that makes a generated task set stop being informative.
        rest = [r for r in self.relations if r not in preferred]
        return preferred + rest

    def generate(
        self,
        n: int,
        *,
        targets: Optional[Sequence[GenerationTarget]] = None,
        seed_ids: Optional[Sequence[str]] = None,
        max_per_seed: int = 4,
        max_per_relation: Optional[int] = None,
        target_bias: float = 0.7,
    ) -> GenerationReport:
        """Generate up to ``n`` variants.

        ``max_per_seed`` and ``max_per_relation`` bound how concentrated the batch
        may become. Both default to bounded values because an unconstrained
        generator will happily emit ``n`` variants of one seed under one relation,
        which measures one thing ``n`` times rather than ``n`` things.

        ``target_bias`` is the probability of drawing from a targeted seed when
        targets are supplied; the remainder is drawn broadly, so the batch keeps
        coverage of tasks the estimates currently think are fine.
        """
        targets = list(targets or [])
        by_task: dict[str, list[str]] = {}
        for t in targets:
            by_task.setdefault(t.task, []).append(t.criterion)

        pool = [s for s in (seed_ids or list(self.seeds)) if s in self.seeds]
        if not pool:
            raise ValueError("no usable seed ids")
        targeted = [s for s in pool if s in by_task]

        out: list[GeneratedTask] = []
        rejected: list[dict] = []
        per_seed: dict[str, int] = {}
        per_rel: dict[str, int] = {}
        attempts = 0
        max_attempts = max(n * 12, 60)

        while len(out) < n and attempts < max_attempts:
            attempts += 1
            use_targeted = targeted and self._rng.random() < target_bias
            src = targeted if use_targeted else pool
            sid = str(src[int(self._rng.integers(0, len(src)))])
            if per_seed.get(sid, 0) >= max_per_seed:
                continue

            seed = self.seeds[sid]
            crit = by_task.get(sid, [])
            rels = self._pick_relations(crit)
            if not rels:
                continue
            # Sample with a mild preference for the first (target-matching) block.
            weights = np.array([2.0 if i < max(len(relations_stressing(*crit)), 1) else 1.0
                                for i in range(len(rels))]) if crit else np.ones(len(rels))
            rel = rels[int(self._rng.choice(len(rels), p=weights / weights.sum()))]
            if max_per_relation is not None and per_rel.get(rel.name, 0) >= max_per_relation:
                continue

            try:
                payload, evidence = rel.apply(seed.payload, self._rng)
            except (ValueError, KeyError, TypeError) as exc:
                rejected.append({
                    "seed_id": sid, "relation": rel.name,
                    "reason": f"transform not applicable: {exc}",
                })
                continue

            note = {
                RelationKind.INVARIANT:
                    "apply the seed's verifier unchanged: this relation does not "
                    "alter the correct answer",
                RelationKind.EQUIVARIANT:
                    "invert the relation's answer map, then apply the seed's "
                    "verifier",
                RelationKind.REFUTED:
                    "the task is now unsatisfiable: the agent must decline, and "
                    "producing a confident answer is a failure",
            }[rel.kind]

            gid = f"{sid}::{rel.name}::{len(out):03d}"
            out.append(GeneratedTask(
                task_id=gid, seed_id=sid, relation=rel.name, relation_kind=rel.kind,
                payload=payload, verifier=seed.verifier, verifier_note=note,
                evidence=evidence, domain=seed.domain, targets=list(crit),
                model_generated=bool(payload.get("_model_generated", False)),
                preserves_difficulty=rel.preserves_difficulty,
                generator=self.generator_id,
            ))
            per_seed[sid] = per_seed.get(sid, 0) + 1
            per_rel[rel.name] = per_rel.get(rel.name, 0) + 1

        warnings: list[str] = []
        if len(out) < n:
            warnings.append(
                f"generated {len(out)} of {n} requested variants in {attempts} "
                "attempts; the per-seed and per-relation caps, or the seeds' "
                "declared fields, limited what was producible. The shortfall is "
                "reported rather than made up by relaxing the caps."
            )
        if self.evaluated_model and self.generator_id:
            g, m = self.generator_id.lower(), self.evaluated_model.lower()
            fam = [t for t in ("gpt", "claude", "llama", "qwen", "mistral", "gemini")
                   if t in g and t in m]
            if fam:
                warnings.append(
                    f"the generator ({self.generator_id}) and the evaluated model "
                    f"({self.evaluated_model}) share a model family ({fam[0]}). The "
                    "generated task distribution is then not independent of the "
                    "model under test, and results should be read with that "
                    "confound stated."
                )
        n_model = sum(1 for t in out if t.model_generated)
        if n_model:
            warnings.append(
                f"{n_model} variant(s) were produced by a model-backed relation. "
                "Those carry a weaker guarantee than structural ones -- the model "
                "may have changed the task's meaning and broken the inherited "
                "verifier -- and are excluded from headline metrics until confirmed."
            )

        return GenerationReport(
            tasks=out, requested=n, seeds_used=sorted(per_seed),
            relations_used=per_rel, targets=targets, rejected=rejected,
            generator=self.generator_id, evaluated_model=self.evaluated_model,
            warnings=warnings,
        )
