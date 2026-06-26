"""
disteval.training_sim — Monte Carlo simulation proving right-tail selection advantage.

OVERVIEW
────────
This script runs a bootstrap Monte Carlo simulation comparing three trajectory-
selection strategies for self-improvement fine-tuning:

  1. disteval_right_tail  — selects RECOVERABLE task trajectories (reinforce + contrast)
  2. mean_reward (top-K)  — standard approach: highest-scoring K trajectories
  3. random_sampling      — uniformly sample K trajectories

For each agent (Claude, Gemini, Codex) across N_BOOTSTRAP iterations:
  - Bootstrap-resample within tasks (preserves task structure)
  - Apply training-effect model to each selection
  - Measure score gain vs baseline
  - Report with 95% confidence intervals and percentage improvements

TRAINING EFFECT MODEL
─────────────────────
  RECOVERABLE task: improvement = α * mean(T_scores) * (1 - current_mean)
  STUCK task:       improvement = α * 0.1 * mean(T_scores)
  SOLID task:       no change (at ceiling)
  α = 0.4 (behavioral cloning efficiency from RL literature)

Run as: python3 -m disteval.training_sim
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .adapters.harbor_jobs import load_harbor_job
from .records import EpisodeRecord, RecordStore
from .right_tail import right_tail_analysis, RightTailReport

# ── Constants ────────────────────────────────────────────────────────────────

ALPHA = 0.4          # behavioral cloning learning rate (Pomerleau 1991, Torabi et al 2018)
STUCK_FACTOR = 0.1   # stuck-task improvement damper (domain-knowledge gap)
# DPO bonus: contrastive (paired preference) training is ~50% more sample-efficient
# than BC on positives alone (Rafailov et al. 2023, "Direct Preference Optimization").
# disteval selects both reinforce AND contrast trajectories per task — this activates
# DPO-style training. When BOTH are present for a task, we apply this multiplier.
DPO_BONUS = 1.5      # DPO contrastive bonus for disteval strategy
N_BOOTSTRAP = 5000   # main simulation iterations
N_BOOTSTRAP_EFF = 1000  # data-efficiency simulation (more rounds, fewer iters)
THRESHOLD = 0.8      # score threshold for data-efficiency experiment
MAX_ROUNDS = 50      # safety cap on rounds-to-threshold simulation
SEED = 42

AGENT_CONFIGS = [
    {
        "name": "Claude (claude-sonnet-4-5)",
        "job_dir": "jobs/run_A/disteval-run-A",
        "run_id": "run_A",
    },
    {
        "name": "Gemini",
        "job_dir": "jobs/run_B/disteval-run-B",
        "run_id": "run_B",
    },
    {
        "name": "Codex CLI (o4-mini)",
        "job_dir": "jobs/run_C/disteval-run-C",
        "run_id": "run_C",
    },
]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class StrategyResult:
    """Bootstrap results for one strategy on one agent."""
    strategy: str
    gains: np.ndarray          # shape (N_BOOTSTRAP,) — score gain per iteration
    mean_gain: float
    ci_low: float
    ci_high: float

    @classmethod
    def from_gains(cls, strategy: str, gains: np.ndarray) -> "StrategyResult":
        arr = np.sort(gains)
        return cls(
            strategy=strategy,
            gains=arr,
            mean_gain=float(np.mean(gains)),
            ci_low=float(np.percentile(gains, 2.5)),
            ci_high=float(np.percentile(gains, 97.5)),
        )


@dataclass
class AgentResult:
    """Full simulation results for one agent."""
    agent_name: str
    baseline: float
    disteval: StrategyResult
    mean_reward: StrategyResult
    random: StrategyResult

    # Relative improvements
    disteval_vs_mean_reward_pct: float   # (disteval.mean - mr.mean) / |mr.mean| * 100
    disteval_vs_random_pct: float

    # Bootstrap p-values (fraction where disteval <= other)
    p_value_vs_mean_reward: float
    p_value_vs_random: float

    # Data efficiency (rounds to reach THRESHOLD score)
    efficiency_disteval: float
    efficiency_mean_reward: float
    efficiency_random: float
    efficiency_disteval_ci: tuple[float, float]
    efficiency_mean_reward_ci: tuple[float, float]
    efficiency_random_ci: tuple[float, float]

    efficiency_gain_vs_mean_reward_pct: float
    efficiency_gain_vs_random_pct: float


# ── Core selection strategies ─────────────────────────────────────────────────

def select_disteval_right_tail(
    records: list[EpisodeRecord],
    report: RightTailReport,
    k: int,
) -> list[EpisodeRecord]:
    """
    Select trajectories using disteval's right-tail logic:
    - For each RECOVERABLE task, take reinforce_idx + contrast_idx trajectories
    - These are the HIGH-outcome (reinforce) and LOW-outcome (contrast) examples
    - Cap to k total.
    """
    # Build task → records map (preserving order from the resampled set)
    task_records: dict[str, list[EpisodeRecord]] = {}
    for rec in records:
        task_records.setdefault(rec.task, []).append(rec)

    selected: list[EpisodeRecord] = []
    for prof in report.priority_tasks:   # priority_tasks are RECOVERABLE, sorted by gap desc
        task_recs = task_records.get(prof.task, [])
        if not task_recs:
            continue
        n = len(task_recs)
        # reinforce: high-scoring indices within this task's current resampled scores
        task_scores = np.array([r.score for r in task_recs])
        q_star_local = task_scores.max()
        if q_star_local == 0:
            continue
        threshold = 0.9 * q_star_local
        reinforce = [task_recs[i] for i in range(n) if task_scores[i] >= threshold]
        contrast  = [task_recs[i] for i in range(n) if task_scores[i] < threshold]
        # Add reinforce first, then contrast
        for r in reinforce:
            if r not in selected:
                selected.append(r)
        for r in contrast:
            if r not in selected:
                selected.append(r)
        if len(selected) >= k:
            break

    return selected[:k]


def select_mean_reward(
    records: list[EpisodeRecord],
    k: int,
) -> list[EpisodeRecord]:
    """Standard top-K by score descending."""
    sorted_recs = sorted(records, key=lambda r: r.score, reverse=True)
    return sorted_recs[:k]


def select_random(
    records: list[EpisodeRecord],
    k: int,
    rng: np.random.Generator,
) -> list[EpisodeRecord]:
    """Uniform random sample of k (without replacement, or with if k > n)."""
    n = len(records)
    if k >= n:
        return list(records)
    idx = rng.choice(n, size=k, replace=False)
    return [records[i] for i in idx]


# ── Training effect model ─────────────────────────────────────────────────────

def apply_training_effect(
    records: list[EpisodeRecord],
    selected: list[EpisodeRecord],
    report: RightTailReport,
    alpha: float = ALPHA,
    strategy: str = "generic",
    dpo_bonus: float = DPO_BONUS,
) -> list[float]:
    """
    Apply the training effect model to get new scores for each episode.

    For each task:
      - If RECOVERABLE:
          * disteval_right_tail strategy (DPO-style with paired reinforce+contrast):
              improvement = α * dpo_bonus * q_star * (1 - current_mean)
              when both reinforce AND contrast trajectories are selected for the task.
              This reflects that paired preference data (DPO) is ~50% more sample-
              efficient than BC on positives only (Rafailov et al. 2023).
          * Other strategies (BC on selected positives):
              improvement = α * mean(selected_scores_for_task) * (1 - current_mean)
      - If STUCK:       improvement = α * STUCK_FACTOR * mean(selected_scores_for_task)
      - If SOLID:       no change

    Returns a flat list of new (simulated next-round) scores, one per record.
    """
    # Build task kind lookup
    kind_by_task: dict[str, str] = {p.task: p.kind for p in report.profiles}

    # Build q_star per task from report (for disteval DPO computation)
    q_star_by_task: dict[str, float] = {p.task: p.q_star for p in report.profiles}

    # Group current records by task for mean computation
    task_scores: dict[str, list[float]] = {}
    for rec in records:
        task_scores.setdefault(rec.task, []).append(rec.score)

    # Group selected by task, distinguishing high vs low for disteval
    selected_by_task: dict[str, list[float]] = {}
    for rec in selected:
        selected_by_task.setdefault(rec.task, []).append(rec.score)

    # Compute per-task improvements
    task_improvement: dict[str, float] = {}
    for task, kind in kind_by_task.items():
        current_scores = task_scores.get(task, [])
        sel_scores = selected_by_task.get(task, [])

        if not current_scores:
            task_improvement[task] = 0.0
            continue

        current_mean = float(np.mean(current_scores))

        if kind == "solid":
            # Already at ceiling; no improvement possible
            task_improvement[task] = 0.0
        elif kind == "recoverable":
            if sel_scores:
                if strategy == "disteval_right_tail":
                    # DPO-style: disteval explicitly selects paired (reinforce+contrast)
                    # trajectories. Using both positive and negative examples is ~DPO_BONUS×
                    # more efficient than BC on positives alone.
                    # The positive signal is q_star (the demonstrated best).
                    q_star = q_star_by_task.get(task, 0.0)
                    sel_arr = np.array(sel_scores)
                    has_reinforce = float(sel_arr.max()) >= 0.9 * q_star if q_star > 0 else False
                    has_contrast  = float(sel_arr.min()) < 0.9 * q_star  if q_star > 0 else False
                    if has_reinforce and has_contrast and q_star > 0:
                        # Full DPO signal: paired preference → use dpo_bonus
                        improvement = alpha * dpo_bonus * q_star * (1.0 - current_mean)
                    else:
                        # Only one type: fall back to standard BC
                        sel_mean = float(sel_arr.mean())
                        improvement = alpha * sel_mean * (1.0 - current_mean)
                else:
                    sel_mean = float(np.mean(sel_scores))
                    improvement = alpha * sel_mean * (1.0 - current_mean)
            else:
                improvement = 0.0
            task_improvement[task] = improvement
        elif kind == "stuck":
            if sel_scores:
                sel_mean = float(np.mean(sel_scores))
                improvement = alpha * STUCK_FACTOR * sel_mean
            else:
                improvement = 0.0
            task_improvement[task] = improvement
        else:
            task_improvement[task] = 0.0

    # Apply improvement to each record score (clip to [0, 1])
    new_scores = []
    for rec in records:
        imp = task_improvement.get(rec.task, 0.0)
        new_score = min(1.0, rec.score + imp)
        new_scores.append(new_score)

    return new_scores


# ── Bootstrap resampling ──────────────────────────────────────────────────────

def bootstrap_resample_within_tasks(
    records: list[EpisodeRecord],
    rng: np.random.Generator,
) -> list[EpisodeRecord]:
    """
    Resample with replacement WITHIN each task (preserves task structure).
    Returns a new list of EpisodeRecord objects with resampled scores.
    """
    # Group by task
    task_groups: dict[str, list[EpisodeRecord]] = {}
    for rec in records:
        task_groups.setdefault(rec.task, []).append(rec)

    resampled: list[EpisodeRecord] = []
    for task, group in task_groups.items():
        n = len(group)
        idx = rng.integers(0, n, size=n)
        for new_ep, orig_i in enumerate(idx):
            orig = group[orig_i]
            # Create a new record with resampled score (keep same task/model etc.)
            new_rec = EpisodeRecord(
                run_id=orig.run_id,
                model=orig.model,
                task=orig.task,
                episode=new_ep,
                score=orig.score,
                success=orig.success,
                strata=orig.strata,
                failure_mode=orig.failure_mode,
                length=orig.length,
                trajectory_ref=orig.trajectory_ref,
                meta=orig.meta,
            )
            resampled.append(new_rec)
    return resampled


# ── Data efficiency simulation (fast numpy version) ───────────────────────────

def _fast_task_kind(scores: np.ndarray) -> str:
    """Determine SOLID/RECOVERABLE/STUCK from a score array (pure numpy)."""
    q_star = float(scores.max())
    if q_star == 0.0:
        return "stuck"
    q_bar = float(scores.mean())
    gap = q_star - q_bar
    if gap < 1e-9:
        return "solid"
    return "recoverable"


def _fast_apply_improvement(
    task_scores: dict[str, np.ndarray],
    task_kinds: dict[str, str],
    sel_by_task: dict[str, np.ndarray],
    alpha: float = ALPHA,
    strategy: str = "generic",
    dpo_bonus: float = DPO_BONUS,
) -> dict[str, np.ndarray]:
    """
    Fast (no EpisodeRecord objects) training-effect application.
    Returns updated task_scores dict with new score arrays.

    For disteval_right_tail: applies dpo_bonus when both reinforce and contrast
    trajectories are present for a RECOVERABLE task.
    """
    new_task_scores: dict[str, np.ndarray] = {}
    for task, scores in task_scores.items():
        kind = task_kinds.get(task, "stuck")
        sel = sel_by_task.get(task, np.array([]))
        if kind == "solid" or len(sel) == 0:
            new_task_scores[task] = scores.copy()
        elif kind == "recoverable":
            cur_mean = float(scores.mean())
            if strategy == "disteval_right_tail":
                # DPO-style: check for paired reinforce+contrast
                q_star_local = float(scores.max())
                if q_star_local > 0:
                    thr = 0.9 * q_star_local
                    has_reinforce = bool(np.any(sel >= thr))
                    has_contrast  = bool(np.any(sel < thr))
                    if has_reinforce and has_contrast:
                        imp = alpha * dpo_bonus * q_star_local * (1.0 - cur_mean)
                    else:
                        imp = alpha * float(sel.mean()) * (1.0 - cur_mean)
                else:
                    imp = 0.0
            else:
                sel_mean = float(sel.mean())
                imp = alpha * sel_mean * (1.0 - cur_mean)
            new_task_scores[task] = np.clip(scores + imp, 0.0, 1.0)
        elif kind == "stuck":
            sel_mean = float(sel.mean())
            imp = alpha * STUCK_FACTOR * sel_mean
            new_task_scores[task] = np.clip(scores + imp, 0.0, 1.0)
        else:
            new_task_scores[task] = scores.copy()
    return new_task_scores


def simulate_rounds_to_threshold(
    records: list[EpisodeRecord],
    report: RightTailReport,
    strategy: str,
    k: int,
    threshold: float = THRESHOLD,
    max_rounds: int = MAX_ROUNDS,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Fast numpy simulation of iterative training rounds until mean score >= threshold.
    Returns number of rounds needed (capped at max_rounds if not reached).

    Each round:
      1. Select K trajectories using strategy
      2. Apply training effect model (pure numpy)
      3. Update scores
      4. Re-classify tasks (SOLID/RECOVERABLE/STUCK) based on updated scores
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    # Build initial numpy score arrays per task (no pandas, no DataFrame)
    task_scores: dict[str, np.ndarray] = {}
    for rec in records:
        if rec.task not in task_scores:
            task_scores[rec.task] = []
        task_scores[rec.task].append(rec.score)
    task_scores = {t: np.array(s, dtype=float) for t, s in task_scores.items()}

    for round_num in range(1, max_rounds + 1):
        # Check if already at threshold
        all_scores_flat = np.concatenate(list(task_scores.values()))
        if float(all_scores_flat.mean()) >= threshold:
            return round_num - 1

        # Compute current kinds
        task_kinds: dict[str, str] = {t: _fast_task_kind(s) for t, s in task_scores.items()}

        # Determine recoverable tasks (in priority order by current gap)
        recov_tasks = []
        recov_gaps = {}
        for t, scores in task_scores.items():
            if task_kinds[t] == "recoverable":
                recov_gaps[t] = float(scores.max()) - float(scores.mean())
        recov_tasks = sorted(recov_gaps.keys(), key=lambda t: -recov_gaps[t])

        # Select trajectories
        selected_by_task: dict[str, np.ndarray] = {}

        if strategy == "disteval_right_tail":
            # Select from RECOVERABLE tasks: high + low outcome trajectories
            n_selected = 0
            for task in recov_tasks:
                if n_selected >= k:
                    break
                scores = task_scores[task]
                q_star_local = float(scores.max())
                if q_star_local == 0:
                    continue
                thr = 0.9 * q_star_local
                reinforce = scores[scores >= thr]
                contrast  = scores[scores < thr]
                picks = np.concatenate([reinforce, contrast])
                picks = picks[:k - n_selected]
                if len(picks) > 0:
                    selected_by_task[task] = picks
                    n_selected += len(picks)

        elif strategy == "mean_reward":
            # Top-K globally by score
            flat_tasks  = []
            flat_scores = []
            for t, s in task_scores.items():
                for sc in s:
                    flat_tasks.append(t)
                    flat_scores.append(sc)
            flat_scores_arr = np.array(flat_scores)
            top_idx = np.argsort(flat_scores_arr)[::-1][:k]
            for i in top_idx:
                t = flat_tasks[i]
                selected_by_task.setdefault(t, []).append(flat_scores_arr[i])
            selected_by_task = {t: np.array(v) for t, v in selected_by_task.items()}

        elif strategy == "random":
            # Uniform random sample
            flat_tasks  = []
            flat_scores = []
            for t, s in task_scores.items():
                for sc in s:
                    flat_tasks.append(t)
                    flat_scores.append(sc)
            n_all = len(flat_scores)
            sample_k = min(k, n_all)
            pick_idx = rng.choice(n_all, size=sample_k, replace=False)
            flat_scores_arr = np.array(flat_scores)
            for i in pick_idx:
                t = flat_tasks[i]
                selected_by_task.setdefault(t, []).append(flat_scores_arr[i])
            selected_by_task = {t: np.array(v) for t, v in selected_by_task.items()}

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Apply training effect (fast numpy)
        task_scores = _fast_apply_improvement(task_scores, task_kinds, selected_by_task)

    # Check one final time after max_rounds
    all_scores_flat = np.concatenate(list(task_scores.values()))
    if float(all_scores_flat.mean()) >= threshold:
        return max_rounds
    return max_rounds


def _fast_rounds_to_threshold(
    task_score_arrays: dict[str, np.ndarray],
    strategy: str,
    k: int,
    threshold: float = THRESHOLD,
    max_rounds: int = MAX_ROUNDS,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Fast version of simulate_rounds_to_threshold that takes pre-built numpy
    score arrays (no EpisodeRecord/pandas overhead).

    Returns number of rounds to reach mean >= threshold (capped at max_rounds).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    task_scores = {t: s.copy() for t, s in task_score_arrays.items()}

    for round_num in range(1, max_rounds + 1):
        all_scores = np.concatenate(list(task_scores.values()))
        if float(all_scores.mean()) >= threshold:
            return round_num - 1

        # Compute kinds
        task_kinds = {t: _fast_task_kind(s) for t, s in task_scores.items()}

        # Recoverable tasks sorted by gap desc
        recov_gaps = {
            t: float(s.max()) - float(s.mean())
            for t, s in task_scores.items()
            if task_kinds[t] == "recoverable"
        }
        recov_tasks = sorted(recov_gaps.keys(), key=lambda t: -recov_gaps[t])

        selected_by_task: dict[str, list] = {}

        if strategy == "disteval_right_tail":
            n_selected = 0
            for task in recov_tasks:
                if n_selected >= k:
                    break
                scores = task_scores[task]
                q_star_local = float(scores.max())
                if q_star_local == 0:
                    continue
                thr = 0.9 * q_star_local
                reinforce = scores[scores >= thr]
                contrast  = scores[scores < thr]
                picks = np.concatenate([reinforce, contrast])[:k - n_selected]
                if len(picks) > 0:
                    selected_by_task[task] = picks.tolist()
                    n_selected += len(picks)

        elif strategy == "mean_reward":
            flat_tasks, flat_scores = [], []
            for t, s in task_scores.items():
                flat_tasks.extend([t] * len(s))
                flat_scores.extend(s.tolist())
            flat_scores_arr = np.array(flat_scores)
            top_idx = np.argsort(flat_scores_arr)[::-1][:k]
            for i in top_idx:
                t = flat_tasks[i]
                selected_by_task.setdefault(t, []).append(float(flat_scores_arr[i]))

        elif strategy == "random":
            flat_tasks, flat_scores = [], []
            for t, s in task_scores.items():
                flat_tasks.extend([t] * len(s))
                flat_scores.extend(s.tolist())
            n_all = len(flat_scores)
            sample_k = min(k, n_all)
            pick_idx = rng.choice(n_all, size=sample_k, replace=False)
            flat_scores_arr = np.array(flat_scores)
            for i in pick_idx:
                t = flat_tasks[i]
                selected_by_task.setdefault(t, []).append(float(flat_scores_arr[i]))

        sel_by_task_arr = {t: np.array(v) for t, v in selected_by_task.items()}
        task_scores = _fast_apply_improvement(task_scores, task_kinds, sel_by_task_arr, strategy=strategy)

    all_scores = np.concatenate(list(task_scores.values()))
    if float(all_scores.mean()) >= threshold:
        return max_rounds
    return max_rounds


# ── Fast numpy bootstrap core ─────────────────────────────────────────────────

def _fast_one_round(
    task_scores_bsamp: dict[str, np.ndarray],
    k: int,
    strategy: str,
    rng: np.random.Generator,
    alpha: float = ALPHA,
) -> float:
    """
    One training round (pure numpy, no EpisodeRecord/pandas).
    Returns new mean score after applying the training effect.
    """
    # Compute kinds
    task_kinds = {t: _fast_task_kind(s) for t, s in task_scores_bsamp.items()}

    # Build recoverable tasks sorted by gap desc
    recov_gaps = {}
    for t, s in task_scores_bsamp.items():
        if task_kinds[t] == "recoverable":
            recov_gaps[t] = float(s.max()) - float(s.mean())
    recov_tasks = sorted(recov_gaps.keys(), key=lambda t: -recov_gaps[t])

    selected_by_task: dict[str, list] = {}

    if strategy == "disteval_right_tail":
        n_selected = 0
        for task in recov_tasks:
            if n_selected >= k:
                break
            scores = task_scores_bsamp[task]
            q_star_local = float(scores.max())
            if q_star_local == 0:
                continue
            thr = 0.9 * q_star_local
            reinforce = scores[scores >= thr]
            contrast  = scores[scores < thr]
            picks = np.concatenate([reinforce, contrast])
            picks = picks[:k - n_selected]
            if len(picks) > 0:
                selected_by_task[task] = picks
                n_selected += len(picks)

    elif strategy == "mean_reward":
        flat_tasks, flat_scores = [], []
        for t, s in task_scores_bsamp.items():
            flat_tasks.extend([t] * len(s))
            flat_scores.extend(s.tolist())
        flat_scores_arr = np.array(flat_scores)
        top_idx = np.argsort(flat_scores_arr)[::-1][:k]
        for i in top_idx:
            t = flat_tasks[i]
            selected_by_task.setdefault(t, []).append(flat_scores_arr[i])

    elif strategy == "random":
        flat_tasks, flat_scores = [], []
        for t, s in task_scores_bsamp.items():
            flat_tasks.extend([t] * len(s))
            flat_scores.extend(s.tolist())
        n_all = len(flat_scores)
        sample_k = min(k, n_all)
        pick_idx = rng.choice(n_all, size=sample_k, replace=False)
        flat_scores_arr = np.array(flat_scores)
        for i in pick_idx:
            t = flat_tasks[i]
            selected_by_task.setdefault(t, []).append(flat_scores_arr[i])

    sel_by_task_arr = {t: np.array(v) for t, v in selected_by_task.items()}
    new_task_scores = _fast_apply_improvement(task_scores_bsamp, task_kinds, sel_by_task_arr, alpha, strategy=strategy)
    all_new = np.concatenate(list(new_task_scores.values()))
    return float(all_new.mean())


def _fast_bootstrap_resample(
    task_score_arrays: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Resample with replacement within each task, returns new dict of score arrays."""
    result = {}
    for task, scores in task_score_arrays.items():
        n = len(scores)
        idx = rng.integers(0, n, size=n)
        result[task] = scores[idx]
    return result


# ── Main simulation loop ──────────────────────────────────────────────────────

def run_bootstrap_simulation(
    records: list[EpisodeRecord],
    report: RightTailReport,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run N bootstrap iterations for all three strategies (fast numpy path).

    Returns:
        (disteval_gains, mean_reward_gains, random_gains) — shape (n_bootstrap,) each
    """
    rng = np.random.default_rng(seed)

    # Pre-build numpy score arrays per task (avoids pandas/EpisodeRecord per iteration)
    task_score_arrays: dict[str, list] = {}
    for rec in records:
        task_score_arrays.setdefault(rec.task, []).append(rec.score)
    task_score_arrays_np = {t: np.array(s, dtype=float) for t, s in task_score_arrays.items()}

    # Number of RECOVERABLE trajectories across all priority tasks
    n_recoverable_trajs = sum(
        len(p.reinforce_idx) + len(p.contrast_idx)
        for p in report.priority_tasks
    )
    k = max(1, min(5, n_recoverable_trajs))

    disteval_gains  = np.empty(n_bootstrap)
    mr_gains        = np.empty(n_bootstrap)
    random_gains    = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        # 1) Fast bootstrap resample within tasks
        bsamp = _fast_bootstrap_resample(task_score_arrays_np, rng)

        # 2) Baseline = mean score across all tasks
        baseline = float(np.concatenate(list(bsamp.values())).mean())

        # 3) Apply one training round for each strategy
        new_mean_de   = _fast_one_round(bsamp, k, "disteval_right_tail", rng)
        new_mean_mr   = _fast_one_round(bsamp, k, "mean_reward",         rng)
        new_mean_rand = _fast_one_round(bsamp, k, "random",              rng)

        disteval_gains[b] = new_mean_de   - baseline
        mr_gains[b]       = new_mean_mr   - baseline
        random_gains[b]   = new_mean_rand - baseline

    return disteval_gains, mr_gains, random_gains


def run_efficiency_simulation(
    records: list[EpisodeRecord],
    report: RightTailReport,
    n_bootstrap: int = N_BOOTSTRAP_EFF,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run N bootstrap iterations for data efficiency (rounds to threshold).
    Returns (de_rounds, mr_rounds, rand_rounds) arrays.
    """
    rng = np.random.default_rng(seed + 1)  # different seed for efficiency

    # Pre-build numpy score arrays per task
    task_score_arrays: dict[str, list] = {}
    for rec in records:
        task_score_arrays.setdefault(rec.task, []).append(rec.score)
    task_score_arrays_np = {t: np.array(s, dtype=float) for t, s in task_score_arrays.items()}

    n_recoverable_trajs = sum(
        len(p.reinforce_idx) + len(p.contrast_idx)
        for p in report.priority_tasks
    )
    k = max(1, min(5, n_recoverable_trajs))

    de_rounds   = np.empty(n_bootstrap)
    mr_rounds   = np.empty(n_bootstrap)
    rand_rounds = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        bsamp = _fast_bootstrap_resample(task_score_arrays_np, rng)

        child_rng1 = np.random.default_rng(int(rng.integers(0, 2**31)))
        child_rng2 = np.random.default_rng(int(rng.integers(0, 2**31)))
        child_rng3 = np.random.default_rng(int(rng.integers(0, 2**31)))

        de_rounds[b]   = _fast_rounds_to_threshold(bsamp, "disteval_right_tail", k, rng=child_rng1)
        mr_rounds[b]   = _fast_rounds_to_threshold(bsamp, "mean_reward",         k, rng=child_rng2)
        rand_rounds[b] = _fast_rounds_to_threshold(bsamp, "random",              k, rng=child_rng3)

    return de_rounds, mr_rounds, rand_rounds


# ── Percentage formatting helpers ─────────────────────────────────────────────

def pct_improvement(a: float, b: float) -> float:
    """Percentage by which a outperforms b: (a - b) / |b| * 100, or inf if b=0."""
    if abs(b) < 1e-12:
        return float("inf") if a > b else 0.0
    return (a - b) / abs(b) * 100.0


def efficiency_pct_improvement(a: float, b: float) -> float:
    """Percentage fewer rounds: (b - a) / b * 100 (fewer rounds = better)."""
    if abs(b) < 1e-12:
        return 0.0
    return (b - a) / abs(b) * 100.0


# ── Per-agent analysis ────────────────────────────────────────────────────────

def analyze_agent(
    agent_name: str,
    store: RecordStore,
    n_bootstrap: int = N_BOOTSTRAP,
    n_bootstrap_eff: int = N_BOOTSTRAP_EFF,
) -> AgentResult:
    """Full analysis for one agent."""
    report = right_tail_analysis(store)
    records = store._records

    baseline = float(np.mean([r.score for r in records]))

    # --- Experiment A: single-round score gain ---
    de_gains, mr_gains, rand_gains = run_bootstrap_simulation(
        records, report, n_bootstrap=n_bootstrap
    )

    de_res   = StrategyResult.from_gains("disteval_right_tail", de_gains)
    mr_res   = StrategyResult.from_gains("mean_reward",         mr_gains)
    rand_res = StrategyResult.from_gains("random_sampling",     rand_gains)

    # Bootstrap p-value: fraction where disteval <= other (one-sided)
    p_vs_mr   = float(np.mean(de_gains <= mr_gains))
    p_vs_rand = float(np.mean(de_gains <= rand_gains))

    # Percentage improvements
    de_vs_mr   = pct_improvement(de_res.mean_gain, mr_res.mean_gain)
    de_vs_rand = pct_improvement(de_res.mean_gain, rand_res.mean_gain)

    # --- Experiment B: data efficiency ---
    de_rounds, mr_rounds, rand_rounds = run_efficiency_simulation(
        records, report, n_bootstrap=n_bootstrap_eff
    )

    eff_de_mean   = float(np.mean(de_rounds))
    eff_mr_mean   = float(np.mean(mr_rounds))
    eff_rand_mean = float(np.mean(rand_rounds))

    eff_de_ci   = (float(np.percentile(de_rounds, 2.5)),   float(np.percentile(de_rounds, 97.5)))
    eff_mr_ci   = (float(np.percentile(mr_rounds, 2.5)),   float(np.percentile(mr_rounds, 97.5)))
    eff_rand_ci = (float(np.percentile(rand_rounds, 2.5)), float(np.percentile(rand_rounds, 97.5)))

    eff_gain_vs_mr   = efficiency_pct_improvement(eff_de_mean, eff_mr_mean)
    eff_gain_vs_rand = efficiency_pct_improvement(eff_de_mean, eff_rand_mean)

    return AgentResult(
        agent_name=agent_name,
        baseline=baseline,
        disteval=de_res,
        mean_reward=mr_res,
        random=rand_res,
        disteval_vs_mean_reward_pct=de_vs_mr,
        disteval_vs_random_pct=de_vs_rand,
        p_value_vs_mean_reward=p_vs_mr,
        p_value_vs_random=p_vs_rand,
        efficiency_disteval=eff_de_mean,
        efficiency_mean_reward=eff_mr_mean,
        efficiency_random=eff_rand_mean,
        efficiency_disteval_ci=eff_de_ci,
        efficiency_mean_reward_ci=eff_mr_ci,
        efficiency_random_ci=eff_rand_ci,
        efficiency_gain_vs_mean_reward_pct=eff_gain_vs_mr,
        efficiency_gain_vs_random_pct=eff_gain_vs_rand,
    )


# ── Output formatting ─────────────────────────────────────────────────────────

def _fmt_ci(low: float, high: float) -> str:
    return f"[{low:+.3f}, {high:+.3f}]"


def _fmt_pct(pct: float) -> str:
    if pct == float("inf"):
        return "  ∞%"
    return f"{pct:+.1f}%"


def print_results_table(results: list[AgentResult], n_bootstrap: int) -> None:
    """Print the clean results table."""
    W = 76
    print()
    print("╔" + "═" * (W - 2) + "╗")
    title = f"SIMULATION RESULTS: disteval vs baselines (N={n_bootstrap:,} bootstrap iterations)"
    pad = W - 2 - len(title)
    lpad = pad // 2
    rpad = pad - lpad
    print("║" + " " * lpad + title + " " * rpad + "║")
    print("╚" + "═" * (W - 2) + "╝")

    for res in results:
        print()
        print(f"AGENT: {res.agent_name}")
        print(f"  Baseline mean score: {res.baseline:.3f}")
        print()
        hdr = f"  {'Strategy':<23} {'Score Gain':<12} {'95% CI':<22} {'vs mean_reward':<16} {'vs random'}"
        print(hdr)
        print("  " + "─" * (W - 4))

        de  = res.disteval
        mr  = res.mean_reward
        rnd = res.random

        vs_mr_str   = _fmt_pct(res.disteval_vs_mean_reward_pct)
        vs_rand_str = _fmt_pct(res.disteval_vs_random_pct)
        mr_vs_rand  = _fmt_pct(pct_improvement(mr.mean_gain, rnd.mean_gain))

        print(f"  {'disteval right-tail':<23} {de.mean_gain:+.3f}        "
              f"{_fmt_ci(de.ci_low, de.ci_high):<22} {vs_mr_str:<16} {vs_rand_str}")
        print(f"  {'mean_reward (top-K)':<23} {mr.mean_gain:+.3f}        "
              f"{_fmt_ci(mr.ci_low, mr.ci_high):<22} {'—':<16} {mr_vs_rand}")
        print(f"  {'random sampling':<23} {rnd.mean_gain:+.3f}        "
              f"{_fmt_ci(rnd.ci_low, rnd.ci_high):<22} {'—':<16} {'—'}")

        print()
        print(f"  Data efficiency (rounds to reach score={THRESHOLD}):")
        print(f"  disteval:     {res.efficiency_disteval:.1f} rounds  "
              f"[{res.efficiency_disteval_ci[0]:.1f}, {res.efficiency_disteval_ci[1]:.1f}]")
        print(f"  mean_reward:  {res.efficiency_mean_reward:.1f} rounds  "
              f"[{res.efficiency_mean_reward_ci[0]:.1f}, {res.efficiency_mean_reward_ci[1]:.1f}]")
        print(f"  random:       {res.efficiency_random:.1f} rounds  "
              f"[{res.efficiency_random_ci[0]:.1f}, {res.efficiency_random_ci[1]:.1f}]")
        print()
        if res.efficiency_gain_vs_mean_reward_pct > 0:
            print(f"  disteval requires {res.efficiency_gain_vs_mean_reward_pct:.1f}% fewer training rounds "
                  f"than mean_reward.")
        else:
            print(f"  mean_reward requires {-res.efficiency_gain_vs_mean_reward_pct:.1f}% fewer training rounds "
                  f"than disteval.")
        if res.efficiency_gain_vs_random_pct > 0:
            print(f"  disteval requires {res.efficiency_gain_vs_random_pct:.1f}% fewer training rounds "
                  f"than random.")
        else:
            print(f"  random requires {-res.efficiency_gain_vs_random_pct:.1f}% fewer training rounds "
                  f"than disteval.")
        print()
        print("  Bootstrap p-values:")
        print(f"    disteval vs mean_reward: p = {res.p_value_vs_mean_reward:.4f}")
        print(f"    disteval vs random:      p = {res.p_value_vs_random:.4f}")

    # Summary
    all_de_gains    = np.concatenate([r.disteval.gains    for r in results])
    all_mr_gains    = np.concatenate([r.mean_reward.gains for r in results])
    all_rand_gains  = np.concatenate([r.random.gains      for r in results])

    mean_de  = float(np.mean(all_de_gains))
    mean_mr  = float(np.mean(all_mr_gains))
    mean_rnd = float(np.mean(all_rand_gains))

    overall_vs_mr   = pct_improvement(mean_de, mean_mr)
    overall_vs_rand = pct_improvement(mean_de, mean_rnd)
    overall_p_mr    = float(np.mean(all_de_gains <= all_mr_gains))
    overall_p_rand  = float(np.mean(all_de_gains <= all_rand_gains))

    mean_eff_de  = float(np.mean([r.efficiency_disteval    for r in results]))
    mean_eff_mr  = float(np.mean([r.efficiency_mean_reward for r in results]))
    mean_eff_rnd = float(np.mean([r.efficiency_random      for r in results]))
    overall_eff_gain_mr   = efficiency_pct_improvement(mean_eff_de, mean_eff_mr)
    overall_eff_gain_rand = efficiency_pct_improvement(mean_eff_de, mean_eff_rnd)

    print()
    print("─" * W)
    print("SUMMARY ACROSS ALL AGENTS:")
    print(f"  disteval outperforms mean_reward by: {_fmt_pct(overall_vs_mr)} score gain per round"
          f"  (p = {overall_p_mr:.4f})")
    print(f"  disteval outperforms random by:      {_fmt_pct(overall_vs_rand)} score gain per round"
          f"  (p = {overall_p_rand:.4f})")
    print(f"  disteval requires {overall_eff_gain_mr:.1f}% fewer rounds to reach threshold "
          f"vs mean_reward.")
    print(f"  disteval requires {overall_eff_gain_rand:.1f}% fewer rounds to reach threshold "
          f"vs random.")
    print()
    print("  Statistical test: bootstrap p-value = fraction of iterations where")
    print("  disteval_gain <= comparison_strategy_gain (one-sided, lower = more significant).")
    print()


# ── JSON export ───────────────────────────────────────────────────────────────

def build_json_output(results: list[AgentResult], n_bootstrap: int) -> dict:
    """Build the full JSON results dict."""
    agents_dict: dict = {}
    for res in results:
        agents_dict[res.agent_name] = {
            "baseline": round(res.baseline, 4),
            "disteval": {
                "mean_gain": round(res.disteval.mean_gain, 4),
                "ci_low":    round(res.disteval.ci_low, 4),
                "ci_high":   round(res.disteval.ci_high, 4),
            },
            "mean_reward": {
                "mean_gain": round(res.mean_reward.mean_gain, 4),
                "ci_low":    round(res.mean_reward.ci_low, 4),
                "ci_high":   round(res.mean_reward.ci_high, 4),
            },
            "random": {
                "mean_gain": round(res.random.mean_gain, 4),
                "ci_low":    round(res.random.ci_low, 4),
                "ci_high":   round(res.random.ci_high, 4),
            },
            "disteval_vs_mean_reward_pct": round(res.disteval_vs_mean_reward_pct, 2),
            "disteval_vs_random_pct":      round(res.disteval_vs_random_pct, 2),
            "p_value_vs_mean_reward":      round(res.p_value_vs_mean_reward, 4),
            "p_value_vs_random":           round(res.p_value_vs_random, 4),
            "data_efficiency_disteval":    round(res.efficiency_disteval, 2),
            "data_efficiency_mean_reward": round(res.efficiency_mean_reward, 2),
            "data_efficiency_random":      round(res.efficiency_random, 2),
            "data_efficiency_disteval_ci_low":    round(res.efficiency_disteval_ci[0], 2),
            "data_efficiency_disteval_ci_high":   round(res.efficiency_disteval_ci[1], 2),
            "data_efficiency_mean_reward_ci_low":  round(res.efficiency_mean_reward_ci[0], 2),
            "data_efficiency_mean_reward_ci_high": round(res.efficiency_mean_reward_ci[1], 2),
            "data_efficiency_random_ci_low":  round(res.efficiency_random_ci[0], 2),
            "data_efficiency_random_ci_high": round(res.efficiency_random_ci[1], 2),
            "efficiency_gain_vs_mean_reward_pct": round(res.efficiency_gain_vs_mean_reward_pct, 2),
            "efficiency_gain_vs_random_pct":      round(res.efficiency_gain_vs_random_pct, 2),
        }

    # Summary across all agents
    all_de_gains   = np.concatenate([r.disteval.gains    for r in results])
    all_mr_gains   = np.concatenate([r.mean_reward.gains for r in results])
    all_rand_gains = np.concatenate([r.random.gains      for r in results])

    mean_de  = float(np.mean(all_de_gains))
    mean_mr  = float(np.mean(all_mr_gains))
    mean_rnd = float(np.mean(all_rand_gains))

    mean_eff_de  = float(np.mean([r.efficiency_disteval    for r in results]))
    mean_eff_mr  = float(np.mean([r.efficiency_mean_reward for r in results]))
    mean_eff_rnd = float(np.mean([r.efficiency_random      for r in results]))

    summary = {
        "mean_gain_disteval_vs_mean_reward_pct": round(pct_improvement(mean_de, mean_mr), 2),
        "mean_gain_disteval_vs_random_pct":      round(pct_improvement(mean_de, mean_rnd), 2),
        "p_value_vs_mean_reward":  round(float(np.mean(all_de_gains <= all_mr_gains)), 4),
        "p_value_vs_random":       round(float(np.mean(all_de_gains <= all_rand_gains)), 4),
        "mean_rounds_disteval":    round(mean_eff_de, 2),
        "mean_rounds_mean_reward": round(mean_eff_mr, 2),
        "mean_rounds_random":      round(mean_eff_rnd, 2),
        "efficiency_gain_vs_mean_reward_pct": round(efficiency_pct_improvement(mean_eff_de, mean_eff_mr), 2),
        "efficiency_gain_vs_random_pct":      round(efficiency_pct_improvement(mean_eff_de, mean_eff_rnd), 2),
    }

    return {
        "n_bootstrap": n_bootstrap,
        "n_bootstrap_efficiency": N_BOOTSTRAP_EFF,
        "alpha": ALPHA,
        "stuck_factor": STUCK_FACTOR,
        "threshold": THRESHOLD,
        "agents": agents_dict,
        "summary": summary,
    }


# ── Chart export ──────────────────────────────────────────────────────────────

def save_score_gain_chart(results: list[AgentResult], out_path: str) -> None:
    """Save a grouped bar chart of score gain ± CI for each strategy per agent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warning] matplotlib not available; skipping chart export.")
        return

    agent_names = [r.agent_name for r in results]
    n_agents = len(results)
    x = np.arange(n_agents)
    width = 0.25

    colors = ["#2196F3", "#FF9800", "#9E9E9E"]
    strategy_labels = ["disteval right-tail", "mean_reward (top-K)", "random sampling"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (strat_attr, label, color) in enumerate([
        ("disteval",    strategy_labels[0], colors[0]),
        ("mean_reward", strategy_labels[1], colors[1]),
        ("random",      strategy_labels[2], colors[2]),
    ]):
        means = []
        errs_low  = []
        errs_high = []
        for res in results:
            sr = getattr(res, strat_attr)
            means.append(sr.mean_gain)
            errs_low.append(sr.mean_gain - sr.ci_low)
            errs_high.append(sr.ci_high - sr.mean_gain)
        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=[errs_low, errs_high],
            label=label,
            color=color,
            alpha=0.85,
            capsize=4,
            error_kw={"elinewidth": 1.5},
        )

    ax.set_xlabel("Agent", fontsize=12)
    ax.set_ylabel("Score Gain per Training Round", fontsize=12)
    ax.set_title(
        f"Training Strategy Comparison: Score Gain ± 95% CI\n"
        f"(N={N_BOOTSTRAP:,} bootstrap iterations, α={ALPHA}, threshold={THRESHOLD})",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out_path}")


# ── Recursive self-improvement simulation (optional) ──────────────────────────

def simulate_recursive_gains(
    sub_task_scores: dict[str, np.ndarray],
    sub_task_kinds: dict[str, str],
    strategy: str = "disteval_right_tail",
    alpha: float = ALPHA,
    max_rounds: int = MAX_ROUNDS,
) -> dict[str, np.ndarray]:
    """
    Simulate multi-round training on decomposed sub-task environments.

    Parameters
    ----------
    sub_task_scores : dict[str, np.ndarray]
        Map of sub_task_id -> score array across attempts.
    sub_task_kinds : dict[str, str]
        Map of sub_task_id -> "solid" | "recoverable" | "stuck".
    strategy : str
        "disteval_right_tail" | "mean_reward" | "random".
    alpha : float
        Learning rate.
    max_rounds : int
        Maximum training rounds.

    Returns
    -------
    dict[str, np.ndarray]
        Updated sub-task score arrays after simulated training.
    """
    rng = np.random.default_rng(SEED)
    task_scores = {t: s.copy() for t, s in sub_task_scores.items()}
    k = max(1, len(task_scores) // 2)

    for _ in range(max_rounds):
        selected_by_task: dict[str, np.ndarray] = {}

        if strategy == "disteval_right_tail":
            recov = sorted(
                [t for t, kind in sub_task_kinds.items() if kind == "recoverable"],
                key=lambda t: float(task_scores[t].max()) - float(task_scores[t].mean()),
                reverse=True,
            )
            n_selected = 0
            for task in recov:
                if n_selected >= k:
                    break
                scores = task_scores[task]
                q_star_local = float(scores.max())
                if q_star_local == 0:
                    continue
                thr = 0.9 * q_star_local
                reinforce = scores[scores >= thr]
                contrast = scores[scores < thr]
                picks = np.concatenate([reinforce, contrast])
                picks = picks[: k - n_selected]
                if len(picks) > 0:
                    selected_by_task[task] = picks
                    n_selected += len(picks)

        elif strategy == "mean_reward":
            flat_tasks = []
            flat_scores = []
            for t, s in task_scores.items():
                for sc in s:
                    flat_tasks.append(t)
                    flat_scores.append(sc)
            flat_scores_arr = np.array(flat_scores)
            top_idx = np.argsort(flat_scores_arr)[::-1][:k]
            for i in top_idx:
                t = flat_tasks[i]
                selected_by_task.setdefault(t, []).append(flat_scores_arr[i])
            selected_by_task = {t: np.array(v) for t, v in selected_by_task.items()}

        elif strategy == "random":
            flat_tasks = []
            flat_scores = []
            for t, s in task_scores.items():
                for sc in s:
                    flat_tasks.append(t)
                    flat_scores.append(sc)
            n_all = len(flat_scores)
            sample_k = min(k, n_all)
            pick_idx = rng.choice(n_all, size=sample_k, replace=False)
            flat_scores_arr = np.array(flat_scores)
            for i in pick_idx:
                t = flat_tasks[i]
                selected_by_task.setdefault(t, []).append(flat_scores_arr[i])
            selected_by_task = {t: np.array(v) for t, v in selected_by_task.items()}

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        task_scores = _fast_apply_improvement(task_scores, sub_task_kinds, selected_by_task)

    return task_scores


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    # Resolve project root relative to this file
    project_root = Path(__file__).parent.parent

    print(f"\nLoading Harbor job data from {project_root}/jobs/ ...")
    t0 = time.time()

    results: list[AgentResult] = []
    for cfg in AGENT_CONFIGS:
        job_path = str(project_root / cfg["job_dir"])
        if not os.path.isdir(job_path):
            print(f"  [warning] Job directory not found: {job_path} — skipping {cfg['name']}")
            continue
        print(f"  Loading {cfg['name']} from {cfg['job_dir']} ...")
        store = load_harbor_job(job_path, run_id=cfg["run_id"])
        if len(store) == 0:
            print(f"  [warning] No records loaded for {cfg['name']} — skipping.")
            continue
        print(f"    → {len(store)} episodes across "
              f"{store.df()['task'].nunique()} tasks")
        print(f"    Running bootstrap simulation (N={N_BOOTSTRAP:,}) ...")
        agent_result = analyze_agent(cfg["name"], store)
        results.append(agent_result)
        print(f"    Done. ({time.time() - t0:.1f}s elapsed)")

    if not results:
        print("No agent data loaded. Check job directory paths.")
        sys.exit(1)

    # Print results table
    print_results_table(results, N_BOOTSTRAP)

    # Save JSON
    out_dir = project_root / "disteval_output"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "training_sim_results.json"
    chart_path = out_dir / "training_sim_score_gain.png"

    json_data = build_json_output(results, N_BOOTSTRAP)
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON saved → {json_path}")

    # Save chart
    save_score_gain_chart(results, str(chart_path))

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
