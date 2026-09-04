"""Optimiser-agnostic export of a selected training set.

The selection question -- *which tasks and which trajectories* -- is separate
from the optimisation question -- *what objective to train with*. Coupling them
would tie the research claim to one algorithm, and DPO in particular is a choice
this framework has no reason to privilege.

So a selected dataset is exported through **views**. A view is a rendering of the
same underlying (task, trajectory, score) data into the shape one family of
trainers expects:

===================== ======================================================
view                  consumed by
===================== ======================================================
``pairwise``          DPO, IPO, SLiC, and other pairwise preference losses
``listwise``          listwise ranking losses, best-of-n reranker training
``weighted_sft``      supervised fine-tuning on successful trajectories,
                      optionally weighted by how much they beat the failures
``scalar_reward``     PPO / GRPO / REINFORCE-style RL, and reward-model fitting
===================== ======================================================

All four are derived from the same selection, so a comparison across objectives
is not confounded by a different data pipeline. A backend declares the views it
needs and receives exactly those.

Compute accounting
------------------
The research claim is about **sample efficiency**, so every view reports what it
would cost to train on: number of examples, total trajectory events, and an
approximate token count. :func:`dataset_cost` aggregates these so held-out gain
can be normalised per unit of training data rather than compared at whatever
volume each strategy happened to yield. Without that normalisation a strategy
that simply produced more pairs would look better for the wrong reason.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .pairs import PreferenceDataset

__all__ = [
    "VIEWS",
    "DatasetCost",
    "dataset_cost",
    "export_view",
    "export_views",
    "to_pairwise",
    "to_listwise",
    "to_weighted_sft",
    "to_scalar_reward",
]

VIEWS = ("pairwise", "listwise", "weighted_sft", "scalar_reward")

#: Rough characters-per-token, used only for the cost estimate. It is a constant
#: because the point is to compare strategies on the same footing, not to predict
#: a bill; the tokeniser-exact number would change every row by the same factor.
_CHARS_PER_TOKEN = 4.0


@dataclass
class DatasetCost:
    """What training on this dataset would cost, for per-unit normalisation."""

    n_examples: int
    n_trajectories: int
    n_events: int
    approx_tokens: int
    n_tasks: int

    def to_dict(self) -> dict:
        return {
            "n_examples": self.n_examples,
            "n_trajectories": self.n_trajectories,
            "n_events": self.n_events,
            "approx_tokens": self.approx_tokens,
            "n_tasks": self.n_tasks,
        }


def _traj_events(payload: Mapping) -> int:
    return int(payload.get("n_events", len(payload.get("events", []) or [])))


def _traj_chars(payload: Mapping) -> int:
    total = 0
    for e in payload.get("events", []) or []:
        total += len(str(e.get("observation") or "")) + len(str(e.get("tool_args") or ""))
    return total


def dataset_cost(dataset: PreferenceDataset) -> DatasetCost:
    """Approximate training cost of a selected dataset.

    Counts both sides of every pair, since a pairwise loss forward-passes both.
    """
    n_events = n_chars = 0
    seen: set[str] = set()
    for p in dataset.pairs:
        for side, key in ((p.chosen, "chosen_id"), (p.rejected, "rejected_id")):
            tid = (side or {}).get("trajectory_id") or p.metadata.get(key)
            if tid is not None:
                seen.add(str(tid))
            n_events += _traj_events(side or {})
            n_chars += _traj_chars(side or {})
    return DatasetCost(
        n_examples=len(dataset.pairs),
        n_trajectories=len(seen),
        n_events=n_events,
        approx_tokens=int(n_chars / _CHARS_PER_TOKEN),
        n_tasks=len({p.task_id for p in dataset.pairs}),
    )


# --------------------------------------------------------------------------- #
# Views                                                                       #
# --------------------------------------------------------------------------- #
def to_pairwise(dataset: PreferenceDataset) -> list[dict]:
    """Chosen/rejected records for pairwise preference losses (DPO, IPO, SLiC)."""
    return [p.to_dict() for p in dataset.pairs]


def to_listwise(dataset: PreferenceDataset) -> list[dict]:
    """One group per task with every distinct trajectory and its score.

    Keeps the *magnitude* of the differences that a pair discards, which listwise
    losses and reward models use.
    """
    groups: dict[str, dict[str, dict]] = {}
    for p in dataset.pairs:
        g = groups.setdefault(p.task_id, {})
        for side, score, key in (
            (p.chosen, p.chosen_score, "chosen_id"),
            (p.rejected, p.rejected_score, "rejected_id"),
        ):
            tid = (side or {}).get("trajectory_id") or p.metadata.get(key)
            if tid is None:
                continue
            g.setdefault(str(tid), {"trajectory_id": str(tid),
                                    "trajectory": side, "score": float(score)})
    out = []
    for task, members in groups.items():
        cands = sorted(members.values(), key=lambda c: -c["score"])
        out.append({"task_id": task, "candidates": cands,
                    "selection_strategy": dataset.strategy})
    return out


def to_weighted_sft(dataset: PreferenceDataset, min_weight: float = 0.0) -> list[dict]:
    """Successful trajectories only, weighted by their margin over the failures.

    The simplest objective that uses this selection at all: fine-tune on what
    worked. The weight is the mean margin over the failed runs it was paired
    against, so a success that beat a near-miss counts for less than one that beat
    a total failure. Included because if plain weighted SFT on the same selection
    matches a preference loss, the preference machinery is not what is doing the
    work -- and that is worth knowing.
    """
    acc: dict[str, dict] = {}
    for p in dataset.pairs:
        tid = (p.chosen or {}).get("trajectory_id") or p.metadata.get("chosen_id")
        if tid is None:
            continue
        rec = acc.setdefault(str(tid), {
            "trajectory_id": str(tid), "task_id": p.task_id,
            "trajectory": p.chosen, "score": p.chosen_score,
            "_margins": [], "selection_strategy": p.selection_strategy,
        })
        rec["_margins"].append(p.margin)
    out = []
    for rec in acc.values():
        w = float(np.mean(rec.pop("_margins")))
        if w < min_weight:
            continue
        rec["weight"] = w
        out.append(rec)
    return out


def to_scalar_reward(dataset: PreferenceDataset) -> list[dict]:
    """Every trajectory with its scalar outcome, for RL or reward-model fitting.

    No preference structure at all: this is the view that lets an RL objective
    consume exactly the same selection, so "which tasks" can be varied
    independently of "which optimiser".
    """
    out: dict[str, dict] = {}
    for p in dataset.pairs:
        for side, score, key in (
            (p.chosen, p.chosen_score, "chosen_id"),
            (p.rejected, p.rejected_score, "rejected_id"),
        ):
            tid = (side or {}).get("trajectory_id") or p.metadata.get(key)
            if tid is None:
                continue
            out[str(tid)] = {
                "trajectory_id": str(tid),
                "task_id": p.task_id,
                "trajectory": side,
                "reward": float(score),
                "selection_strategy": p.selection_strategy,
            }
    return list(out.values())


_RENDERERS = {
    "pairwise": to_pairwise,
    "listwise": to_listwise,
    "weighted_sft": to_weighted_sft,
    "scalar_reward": to_scalar_reward,
}


def export_view(dataset: PreferenceDataset, view: str, path: str | Path) -> int:
    """Write one view as JSONL. Returns the number of records."""
    if view not in _RENDERERS:
        raise ValueError(f"unknown view {view!r}; have {sorted(_RENDERERS)}")
    rows = _RENDERERS[view](dataset)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return len(rows)


def export_views(
    dataset: PreferenceDataset,
    output_dir: str | Path,
    views: Sequence[str] = VIEWS,
) -> dict[str, str]:
    """Write each requested view, plus a manifest recording the selection and cost."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths, counts = {}, {}
    for v in views:
        p = out / f"{v}.jsonl"
        counts[v] = export_view(dataset, v, p)
        paths[v] = str(p)
    manifest = {
        "strategy": dataset.strategy,
        "summary": dataset.summary(),
        "cost": dataset_cost(dataset).to_dict(),
        "views": {v: {"path": paths[v], "n_records": counts[v]} for v in views},
        "selected_tasks": list(dataset.selected_tasks),
        "note": (
            "views are alternative renderings of one selection; a comparison "
            "across training objectives using these is not confounded by a "
            "different data pipeline"
        ),
    }
    mpath = out / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    paths["manifest"] = str(mpath)
    return paths
