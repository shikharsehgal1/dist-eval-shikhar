"""Building preference pairs from matched successful and failed trajectories.

Preference learning from success/failure trajectory pairs is **prior work** (ETO
and the trajectory-preference literature); nothing about the construction here is
claimed as novel. What this repository tests is *which tasks to build pairs from*.

Matching, and why it matters
----------------------------
A pair is only a clean learning signal if the two trajectories differ in ways
attributable to the agent rather than to the task. Pairs are therefore built
**within** a task and, where the data supports it, within the same environment
seed/variant -- so instruction, environment and rubric are held fixed and the
contrast is behavioural. Cross-task pairs are supported but off by default and
flagged, because a "chosen" from an easy task against a "rejected" from a hard
one teaches the model about task difficulty, not about doing the task well.

Pair quality
------------
Not every (success, failure) combination is equally informative, so pairs carry a
``quality`` score and can be filtered:

* **Score margin.** A 1.0-vs-0.9 pair is mostly noise; a 1.0-vs-0.0 pair is a
  clear contrast. Configurable minimum margin.
* **Divergence locality.** When trajectories are available, a pair whose runs
  share a long common prefix and diverge at one identifiable point is a sharper
  signal than two runs that differ everywhere. Computed from the alignment.
* **Diversity.** Taking the ``m`` best pairs from one task often returns ``m``
  near-copies. Pairs are de-duplicated by their divergence signature and capped
  per task, so a fixed pair budget covers more distinct failure modes.

Fair budgets
------------
Every strategy is compared at equal pair counts, not equal task counts, because
a strategy that happens to select tasks with more failures would otherwise get
more training data. :func:`build_dataset` takes a target pair count and reports
what each strategy actually delivered.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from ..trajectory.divergence import analyse_divergence
from ..trajectory.events import Trajectory

__all__ = [
    "PreferencePair",
    "PairConfig",
    "build_pairs_for_task",
    "build_dataset",
    "PreferenceDataset",
    "to_dpo_jsonl",
    "to_ranking_jsonl",
]


@dataclass
class PairConfig:
    """Pair-construction policy. All thresholds explicit."""

    #: Minimum score gap between chosen and rejected.
    min_margin: float = 0.5
    #: Maximum pairs kept per task, to stop one task dominating the dataset.
    max_pairs_per_task: int = 4
    #: Require both trajectories to share an environment/seed key when present.
    match_environment: bool = True
    #: Allow pairs drawn from different tasks. Off by default; see module docs.
    allow_cross_task: bool = False
    #: Drop pairs whose divergence signature duplicates one already kept.
    deduplicate_by_divergence: bool = True
    #: Compute alignment-based quality. Costs O(n*m) per pair; off for huge runs.
    compute_divergence: bool = True
    #: Weight of divergence locality in the quality score.
    locality_weight: float = 0.3


@dataclass
class PreferencePair:
    """One (chosen, rejected) training example with full provenance."""

    task_id: str
    chosen: dict
    rejected: dict
    chosen_score: float
    rejected_score: float
    recoverability_score: float
    selection_strategy: str
    quality: float
    metadata: dict = field(default_factory=dict)

    @property
    def margin(self) -> float:
        return self.chosen_score - self.rejected_score

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_score": self.chosen_score,
            "rejected_score": self.rejected_score,
            "margin": self.margin,
            "recoverability_score": self.recoverability_score,
            "selection_strategy": self.selection_strategy,
            "quality": self.quality,
            "metadata": self.metadata,
        }


def _traj_payload(t: Trajectory, max_events: int = 512) -> dict:
    """Compact, trainer-agnostic serialisation of a trajectory."""
    return {
        "trajectory_id": t.trajectory_id,
        "task": t.task,
        "model": t.model,
        "run_id": t.run_id,
        "episode": t.episode,
        "score": t.score,
        "success": t.success,
        "n_events": len(t.events),
        "events": [
            {
                "index": e.index,
                "event_type": e.event_type,
                "tool_name": e.tool_name,
                "tool_args": e.tool_args,
                "target": e.target,
                "observation": e.observation,
                "ok": e.ok,
            }
            for e in t.events[:max_events]
        ],
        "truncated": len(t.events) > max_events,
    }


def _env_key(t: Trajectory) -> Optional[str]:
    for k in ("environment", "env", "seed", "variant", "world"):
        if k in t.meta:
            return f"{k}={t.meta[k]}"
    return None


def build_pairs_for_task(
    successes: Sequence[Trajectory],
    failures: Sequence[Trajectory],
    *,
    task: str = "",
    recoverability: float = float("nan"),
    strategy: str = "",
    config: Optional[PairConfig] = None,
    extra_metadata: Optional[Mapping] = None,
) -> list[PreferencePair]:
    """All qualifying pairs for one task, ranked by quality, capped per task."""
    cfg = config or PairConfig()
    task = task or (successes[0].task if successes else (failures[0].task if failures else ""))
    candidates: list[PreferencePair] = []

    for s in successes:
        for f in failures:
            margin = float(s.score) - float(f.score)
            if margin < cfg.min_margin:
                continue
            if cfg.match_environment:
                ks, kf = _env_key(s), _env_key(f)
                if ks is not None and kf is not None and ks != kf:
                    continue

            locality, signature, div_meta = float("nan"), None, {}
            if cfg.compute_divergence and len(s.events) and len(f.events):
                try:
                    rep = analyse_divergence(s, f)
                    # Locality: a long shared prefix and a small downstream blast
                    # radius means one identifiable thing went wrong.
                    denom = max(len(f.events), 1)
                    prefix_frac = rep.common_prefix / denom
                    locality = float(
                        np.clip(prefix_frac * (1.0 - rep.consequence_strength), 0.0, 1.0)
                    )
                    if rep.primary is not None:
                        signature = (
                            f"{rep.primary.kind}:{rep.primary.j}:"
                            f"{rep.primary.detail.get('failure_target')}"
                        )
                    div_meta = rep.to_dict()
                except Exception as exc:  # alignment can refuse very long runs
                    div_meta = {"divergence_error": str(exc)}

            quality = margin
            if np.isfinite(locality):
                quality = (1 - cfg.locality_weight) * margin + cfg.locality_weight * locality

            candidates.append(
                PreferencePair(
                    task_id=task,
                    chosen=_traj_payload(s),
                    rejected=_traj_payload(f),
                    chosen_score=float(s.score),
                    rejected_score=float(f.score),
                    recoverability_score=float(recoverability),
                    selection_strategy=strategy,
                    quality=float(quality),
                    metadata={
                        "chosen_id": s.trajectory_id,
                        "rejected_id": f.trajectory_id,
                        "environment_matched": _env_key(s) == _env_key(f),
                        "divergence_signature": signature,
                        "divergence_locality": locality,
                        "domain": s.domain or f.domain,
                        **dict(extra_metadata or {}),
                        **div_meta,
                    },
                )
            )

    candidates.sort(key=lambda p: -p.quality)
    if cfg.deduplicate_by_divergence:
        seen: set = set()
        kept = []
        for p in candidates:
            sig = p.metadata.get("divergence_signature")
            if sig is not None and sig in seen:
                continue
            if sig is not None:
                seen.add(sig)
            kept.append(p)
        candidates = kept
    return candidates[: cfg.max_pairs_per_task]


@dataclass
class PreferenceDataset:
    """A built preference dataset, with the accounting to compare strategies."""

    pairs: list[PreferencePair]
    strategy: str
    requested_pairs: int
    selected_tasks: list[str]
    tasks_with_pairs: list[str]
    note: str = ""
    config: PairConfig = field(default_factory=PairConfig)

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def shortfall(self) -> int:
        return max(self.requested_pairs - len(self.pairs), 0)

    def summary(self) -> dict:
        margins = [p.margin for p in self.pairs]
        loc = [
            p.metadata.get("divergence_locality", float("nan")) for p in self.pairs
        ]
        loc = [x for x in loc if isinstance(x, float) and np.isfinite(x)]
        return {
            "strategy": self.strategy,
            "n_pairs": len(self.pairs),
            "requested_pairs": self.requested_pairs,
            "shortfall": self.shortfall,
            "n_selected_tasks": len(self.selected_tasks),
            "n_tasks_with_pairs": len(self.tasks_with_pairs),
            "mean_margin": float(np.mean(margins)) if margins else float("nan"),
            "mean_locality": float(np.mean(loc)) if loc else float("nan"),
            "pairs_per_task": (
                len(self.pairs) / len(self.tasks_with_pairs)
                if self.tasks_with_pairs else 0.0
            ),
            "note": self.note,
        }

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "task_id": p.task_id,
                    "chosen_id": p.metadata.get("chosen_id"),
                    "rejected_id": p.metadata.get("rejected_id"),
                    "chosen_score": p.chosen_score,
                    "rejected_score": p.rejected_score,
                    "margin": p.margin,
                    "recoverability_score": p.recoverability_score,
                    "quality": p.quality,
                    "locality": p.metadata.get("divergence_locality"),
                    "strategy": p.selection_strategy,
                }
                for p in self.pairs
            ]
        )


def build_dataset(
    selected_tasks: Sequence[str],
    trajectories: Mapping[str, Sequence[Trajectory]],
    *,
    strategy: str,
    target_pairs: int,
    recoverability: Optional[Mapping[str, float]] = None,
    config: Optional[PairConfig] = None,
    seed: int = 0,
) -> PreferenceDataset:
    """Build a preference dataset of ``target_pairs`` from the selected tasks.

    Tasks are drawn in selection order (best first) and each contributes at most
    ``max_pairs_per_task``, so a fixed pair budget spreads across as many tasks as
    the strategy ranked highly rather than being consumed by the first one. When
    the selected tasks cannot supply the target, the shortfall is reported --
    which is itself informative, since a strategy that picks tasks with no usable
    pairs has a real problem.
    """
    cfg = config or PairConfig()
    rec = recoverability or {}
    pairs: list[PreferencePair] = []
    with_pairs: list[str] = []

    for task in selected_tasks:
        if len(pairs) >= target_pairs:
            break
        runs = list(trajectories.get(task, []))
        succ = [t for t in runs if t.success]
        fail = [t for t in runs if not t.success]
        if not succ or not fail:
            continue
        got = build_pairs_for_task(
            succ, fail,
            task=task,
            recoverability=float(rec.get(task, float("nan"))),
            strategy=strategy,
            config=cfg,
        )
        if got:
            with_pairs.append(task)
            pairs.extend(got[: target_pairs - len(pairs)])

    note = ""
    if len(pairs) < target_pairs:
        note = (
            f"{strategy!r} produced {len(pairs)} of {target_pairs} requested pairs "
            f"from {len(with_pairs)} of {len(selected_tasks)} selected tasks; "
            "the shortfall is reported, not backfilled from another strategy"
        )
    return PreferenceDataset(
        pairs=pairs,
        strategy=strategy,
        requested_pairs=target_pairs,
        selected_tasks=list(selected_tasks),
        tasks_with_pairs=with_pairs,
        note=note,
        config=cfg,
    )


def to_dpo_jsonl(dataset: PreferenceDataset, path: str) -> int:
    """Write DPO-format records. Returns the number written.

    Each record carries ``chosen``/``rejected`` plus the full selection
    provenance, so a training run can be traced back to the estimate that chose
    its data -- which is the whole point of the experiment.
    """
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for p in dataset.pairs:
            f.write(json.dumps(p.to_dict(), default=str) + "\n")
            n += 1
    return n


def to_ranking_jsonl(dataset: PreferenceDataset, path: str) -> int:
    """Write trajectory-level ranking records: one group per task, scored runs.

    Preference pairs discard the magnitude of the difference; a ranking format
    keeps it, which some trainers use. Both are emitted from the same dataset so
    the comparison is not confounded by the export format.
    """
    by_task: dict[str, list[dict]] = {}
    for p in dataset.pairs:
        g = by_task.setdefault(p.task_id, [])
        sides = (
            (p.chosen, p.chosen_score, p.metadata.get("chosen_id")),
            (p.rejected, p.rejected_score, p.metadata.get("rejected_id")),
        )
        for side, score, fallback_id in sides:
            tid = side.get("trajectory_id", fallback_id) if isinstance(side, dict) else fallback_id
            if tid is None:
                continue
            if not any(c.get("trajectory_id") == tid for c in g):
                g.append({"trajectory_id": tid, "trajectory": side, "score": score})
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for task, cands in by_task.items():
            cands.sort(key=lambda c: -c["score"])
            f.write(
                json.dumps(
                    {
                        "task_id": task,
                        "candidates": cands,
                        "selection_strategy": dataset.strategy,
                    },
                    default=str,
                )
                + "\n"
            )
            n += 1
    return n
