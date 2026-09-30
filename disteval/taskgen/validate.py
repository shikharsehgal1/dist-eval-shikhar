"""Validity gates and diversity measurement for generated task batches.

A generated task is worth evaluating on only if it is *gradeable*, *distinct*
from its seed, and *not a near-duplicate* of its siblings. This module enforces
all three, and reports the batch-level concentration that says whether a set of
variants measures many things or one thing many times.

The gates
---------
``gradeable``
    The task carries a verifier and a relation whose kind says how to apply it.
    Without this it is not an eval task. Non-negotiable.

``distinct_from_seed``
    The transformation actually changed something. A no-op variant inflates the
    apparent size of an eval set while adding nothing, and -- worse -- it looks
    like agreement when the agent trivially reproduces its seed answer.

``well_formed``
    The payload still satisfies the structural invariants the seed declared: it
    did not lose every file, empty its instruction, or drop an input a
    non-``refuted`` relation was supposed to preserve.

``confirmed_if_model_generated``
    A variant produced by a model-backed relation is held out of headline
    metrics unless explicitly confirmed, because the model may have changed the
    task's meaning and silently broken the inherited verifier.

``solvable`` (optional)
    When a solver callback is supplied, the variant must be solved at least once
    in ``n_probe`` attempts. This is the expensive gate and it is off by default.
    Note what it costs and what it buys: it catches variants that are broken
    rather than merely hard, but it also **removes exactly the hardest variants**,
    which biases the surviving set toward easier tasks. :func:`validate_batch`
    reports how many were dropped this way so the bias is visible rather than
    silent.

Diversity
---------
A generator left unconstrained emits many variants of one seed under one
relation. :func:`batch_diversity` measures this three ways -- distinct-n over the
instruction text, relation and seed concentration (Herfindahl), and the share
held by the single largest group -- because a batch that is concentrated is
measuring one thing repeatedly no matter how many rows it has. The n-gram
measures follow the usual practice in generated-text evaluation; the
concentration measures are what actually matter for a metamorphic batch, where
the text may differ while the probe does not.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from .generator import GeneratedTask, GenerationReport
from .relations import RelationKind

__all__ = [
    "ValidationConfig",
    "validate_task",
    "validate_batch",
    "batch_diversity",
    "GATES",
]

GATES = (
    "gradeable",
    "distinct_from_seed",
    "well_formed",
    "confirmed_if_model_generated",
    "solvable",
)


@dataclass
class ValidationConfig:
    """Which gates to enforce, and how."""

    require_verifier: bool = True
    require_distinct: bool = True
    require_well_formed: bool = True
    #: Model-generated variants are excluded unless their ids appear in
    #: ``confirmed_ids``. Set False only when you have another reason to trust them.
    exclude_unconfirmed_model_generated: bool = True
    confirmed_ids: frozenset = frozenset()
    #: Optional solver probe. ``(task) -> bool`` for a single attempt.
    solver: Optional[Callable[[GeneratedTask], bool]] = None
    n_probe: int = 3
    #: Minimum fields a payload must retain to count as well formed.
    required_payload_keys: tuple[str, ...] = ("instruction",)


def _payload_signature(payload: Mapping) -> str:
    import json

    clean = {k: v for k, v in payload.items() if not str(k).startswith("_")}
    return json.dumps(clean, sort_keys=True, default=str)


def validate_task(
    task: GeneratedTask,
    seed_payload: Optional[Mapping] = None,
    config: Optional[ValidationConfig] = None,
) -> dict:
    """Run every configured gate. Returns the verdict and each gate's result."""
    cfg = config or ValidationConfig()
    results: dict[str, object] = {}
    reasons: list[str] = []

    ok = bool(task.verifier) or not cfg.require_verifier
    results["gradeable"] = ok
    if not ok:
        reasons.append("no verifier: the variant cannot be graded")

    if cfg.require_distinct and seed_payload is not None:
        distinct = _payload_signature(task.payload) != _payload_signature(seed_payload)
        results["distinct_from_seed"] = distinct
        if not distinct:
            reasons.append("payload is identical to the seed: the transform was a no-op")
    else:
        results["distinct_from_seed"] = True

    if cfg.require_well_formed:
        wf = True
        for key in cfg.required_payload_keys:
            v = task.payload.get(key)
            if v is None or (isinstance(v, str) and not v.strip()):
                wf = False
                reasons.append(f"payload lost required field {key!r}")
        files = task.payload.get("files")
        if files is not None and len(files) == 0 and task.relation_kind != RelationKind.REFUTED:
            wf = False
            reasons.append("payload has no files left, but the relation should have "
                           "preserved solvability")
        results["well_formed"] = wf
    else:
        results["well_formed"] = True

    if task.model_generated and cfg.exclude_unconfirmed_model_generated:
        confirmed = task.task_id in cfg.confirmed_ids
        results["confirmed_if_model_generated"] = confirmed
        if not confirmed:
            reasons.append("model-generated and not confirmed: the inherited "
                           "verifier may no longer apply")
    else:
        results["confirmed_if_model_generated"] = True

    if cfg.solver is not None:
        if task.relation_kind == RelationKind.REFUTED:
            # An unsatisfiable variant is *supposed* to be unsolvable; probing it
            # for solvability would reject exactly the tasks this relation exists
            # to create.
            results["solvable"] = True
            results["solvable_note"] = "skipped: refuted relations are unsolvable by design"
        else:
            solved = any(bool(cfg.solver(task)) for _ in range(max(cfg.n_probe, 1)))
            results["solvable"] = solved
            if not solved:
                reasons.append(
                    f"not solved in {cfg.n_probe} probe attempts: may be broken "
                    "rather than merely hard"
                )
    else:
        results["solvable"] = True

    results["valid"] = all(
        bool(results.get(g, True)) for g in GATES if g in results
    )
    results["reasons"] = reasons
    return results


def validate_batch(
    report: GenerationReport,
    seeds: Mapping[str, object],
    config: Optional[ValidationConfig] = None,
) -> dict:
    """Validate every task in a report in place, and summarise what was dropped."""
    cfg = config or ValidationConfig()
    gate_failures: Counter = Counter()
    for t in report.tasks:
        seed = seeds.get(t.seed_id)
        seed_payload = getattr(seed, "payload", None) if seed is not None else None
        t.validation = validate_task(t, seed_payload, cfg)
        if not t.validation["valid"]:
            for g in GATES:
                if g in t.validation and not t.validation[g]:
                    gate_failures[g] += 1

    valid = report.valid_tasks
    out = {
        "n_tasks": len(report.tasks),
        "n_valid": len(valid),
        "n_rejected": len(report.tasks) - len(valid),
        "failures_by_gate": dict(gate_failures),
        "diversity": batch_diversity(valid),
    }
    if cfg.solver is not None and gate_failures.get("solvable"):
        out["solvability_bias_warning"] = (
            f"{gate_failures['solvable']} variant(s) were dropped for failing the "
            "solvability probe. That gate removes variants that are broken, but it "
            "also removes the hardest genuine ones, biasing the surviving set "
            "toward easier tasks. Report the drop rate alongside any difficulty "
            "statistic computed on the survivors."
        )
    return out


def _distinct_n(texts: Sequence[str], n: int = 3) -> float:
    """Fraction of n-grams across the batch that are unique. Higher is more varied."""
    grams: list[tuple] = []
    for t in texts:
        toks = str(t).split()
        grams.extend(tuple(toks[i:i + n]) for i in range(max(len(toks) - n + 1, 0)))
    if not grams:
        return float("nan")
    return float(len(set(grams)) / len(grams))


def _herfindahl(counts: Iterable[int]) -> float:
    """Concentration index: 1/k for k equal groups, 1.0 when one group holds all."""
    c = np.array(list(counts), dtype=float)
    if c.sum() <= 0:
        return float("nan")
    share = c / c.sum()
    return float(np.sum(share ** 2))


def batch_diversity(tasks: Sequence[GeneratedTask]) -> dict:
    """How concentrated is this batch -- does it measure many things or one thing?

    ``relation_concentration`` and ``seed_concentration`` are the ones that matter
    for metamorphic batches: text can differ substantially while every variant
    probes the same invariance of the same seed, and an n-gram measure would call
    that diverse. A concentration near ``1/k`` is even coverage; near 1.0 means
    one group dominates.
    """
    if not tasks:
        return {"n": 0}
    rel = Counter(t.relation for t in tasks)
    seed = Counter(t.seed_id for t in tasks)
    kind = Counter(t.relation_kind for t in tasks)
    texts = [str(t.payload.get("instruction") or "") for t in tasks]
    return {
        "n": len(tasks),
        "n_distinct_relations": len(rel),
        "n_distinct_seeds": len(seed),
        "n_distinct_kinds": len(kind),
        "relation_concentration": _herfindahl(rel.values()),
        "seed_concentration": _herfindahl(seed.values()),
        "largest_relation_share": float(max(rel.values()) / len(tasks)),
        "largest_seed_share": float(max(seed.values()) / len(tasks)),
        "distinct_3": _distinct_n(texts, 3),
        "distinct_2": _distinct_n(texts, 2),
        "note": (
            "relation and seed concentration are the meaningful measures here; "
            "distinct-n can look healthy while every variant probes the same "
            "invariance of the same seed"
        ),
    }
