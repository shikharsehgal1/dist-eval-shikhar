"""Seam: Harbor `jobs/` output  ->  RecordStore   (primary real integration).

Per the harbor docs and source:
  jobs/<job>/<trial_dir>/
      config.json
      result.json              # TrialResult: id, task_name, trial_name, task_id,
                               #              agent_result, verifier_result, ...
      verifier/reward.txt      # scalar reward  OR
      verifier/reward.json     # {"correctness": 0.75, "quality": 0.9}  (named rewards)
      agent/episode-*/         # per-rollout-step dirs (trace export grist)

Harbor's built-in metric is a reducer over rewards that DEFAULTS TO MEAN -- i.e.
it mean-collapses. This adapter lifts the per-trial rewards back into a
distribution so disteval can compute IQM / CVaR / pass^k / repeat-noise instead.

New in this version: reads difficulty metadata from task.toml files (via
--tasks-dir) so .slice(difficulty="hard") works on real Harbor output without
any manual annotation.
"""
from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore

from ..records import EpisodeRecord, RecordStore

REWARD_JSON_PATHS = (
    ("verifier_result", "reward"),
    ("verifier_result", "reward", "value"),
    ("agent_result", "reward"),
    ("reward",),
)


def _dig(d: dict, path: tuple[str, ...]) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _read_reward(trial_dir: str, result: dict) -> Optional[float | dict]:
    # 1) sidecar files (most reliable per docs)
    rj = os.path.join(trial_dir, "verifier", "reward.json")
    rt = os.path.join(trial_dir, "verifier", "reward.txt")
    if os.path.exists(rj):
        with open(rj) as f:
            return json.load(f)
    if os.path.exists(rt):
        with open(rt) as f:
            txt = f.read().strip()
        try:
            return float(txt)
        except ValueError:
            return None
    # 2) fall back to result.json fields
    for p in REWARD_JSON_PATHS:
        v = _dig(result, p)
        if v is not None:
            return v
    return None


def _scalarize(reward: float | dict | None, reward_key: Optional[str]) -> Optional[float]:
    if reward is None:
        return None
    if isinstance(reward, (int, float)):
        return float(reward)
    if isinstance(reward, dict):
        if reward_key and reward_key in reward:
            return float(reward[reward_key])
        # default: mean of named rewards (mirror Harbor's reducer choice)
        nums = [float(v) for v in reward.values() if isinstance(v, (int, float))]
        return float(sum(nums) / len(nums)) if nums else None
    return None


def _build_task_metadata_index(tasks_dir: str) -> dict[str, dict]:
    """Scan a tasks/ directory for task.toml files and index metadata by task name.

    Returns {task_name: {difficulty, category, ...}} so we can attach real
    difficulty strata to every trial record without manual annotation.
    """
    index: dict[str, dict] = {}
    if not tasks_dir or not os.path.isdir(tasks_dir):
        return index
    for toml_path in Path(tasks_dir).rglob("task.toml"):
        meta: dict = {}
        try:
            if tomllib is not None:
                with open(toml_path, "rb") as f:
                    doc = tomllib.load(f)
            else:
                # fallback: crude key=value parse for the fields we need
                doc = _parse_toml_simple(str(toml_path))
            task_name = _dig(doc, ("task", "name")) or ""
            raw_meta = doc.get("metadata", {}) or {}
            meta["difficulty"] = raw_meta.get("difficulty", "unknown")
            meta["category"] = raw_meta.get("category", "unknown")
            meta["difficulty_explanation"] = raw_meta.get("difficulty_explanation", "")
            if task_name:
                # index by full name AND short name (last component)
                index[task_name] = meta
                index[task_name.split("/")[-1]] = meta
        except Exception:
            pass
    return index


def _parse_toml_simple(path: str) -> dict:
    """Minimal TOML parser for task.toml when tomllib/tomli not available.
    Only handles the fields we care about: [task].name and [metadata].*
    """
    doc: dict = {}
    section = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and "]" in line:
                section = line[1:line.index("]")].strip()
                parts = section.split(".")
                cur = doc
                for p in parts:
                    cur = cur.setdefault(p, {})
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                parts = (section or "").split(".") if section else []
                cur = doc
                for p in parts:
                    cur = cur.setdefault(p, {})
                cur[k] = v
    return doc


def load_harbor_job(
    job_dir: str,
    run_id: Optional[str] = None,
    reward_key: Optional[str] = None,
    success_threshold: float = 0.99,
    strata_from_config: tuple[str, ...] = (),
    tasks_dir: Optional[str] = None,
) -> RecordStore:
    """Read one Harbor job directory into a RecordStore (one record per trial).

    run_id: label this job (use a distinct run_id per repeated job to feed
            repeat.py's meta-distribution).
    reward_key: if rewards are dicts, which key to treat as the primary score.
    tasks_dir: path to the tasks/ directory — used to read difficulty metadata
               from task.toml files so strata are populated automatically.
    """
    store = RecordStore()
    run_id = run_id or os.path.basename(os.path.normpath(job_dir))

    # Build task metadata index from task.toml files if tasks_dir provided
    meta_index = _build_task_metadata_index(tasks_dir) if tasks_dir else {}

    trial_dirs = sorted(
        d for d in glob(os.path.join(job_dir, "*")) if os.path.isdir(d)
    )
    per_task_ep: dict[str, int] = {}
    for td in trial_dirs:
        res_path = os.path.join(td, "result.json")
        if not os.path.exists(res_path):
            continue
        with open(res_path) as f:
            result = json.load(f)
        cfg = {}
        cfg_path = os.path.join(td, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)

        task = str(result.get("task_name") or result.get("task_id") or os.path.basename(td))
        model = str(
            (result.get("agent_info", {}) or {}).get("model")
            or cfg.get("model")
            or "unknown"
        )
        reward = _read_reward(td, result)
        score = _scalarize(reward, reward_key)
        if score is None:
            score = 0.0
            success = False
            fmode = "missing_reward"
        else:
            success = score >= success_threshold
            fmode = None if success else "below_threshold"

        # Build strata: start from task.toml metadata, supplement with config
        strata: dict[str, Any] = {}
        task_meta = meta_index.get(task, {})
        if task_meta:
            strata["difficulty"] = task_meta.get("difficulty", "unknown")
            strata["category"] = task_meta.get("category", "unknown")
        for k in strata_from_config:
            if k in cfg:
                strata[k] = cfg[k]

        ep = per_task_ep.get(task, 0)
        per_task_ep[task] = ep + 1
        store.add(EpisodeRecord(
            run_id=run_id, model=model, task=task, episode=ep,
            score=float(score), success=bool(success), strata=strata,
            failure_mode=fmode, trajectory_ref=td,
            meta={"trial_name": result.get("trial_name"), "task_metadata": task_meta},
        ))
    return store


def load_harbor_jobs(job_dirs: list[str], **kw) -> list[RecordStore]:
    """Load several Harbor jobs as repeated evals (one RecordStore each) ->
    feed directly to repeat.meta_distribution / repeat.bootstrap_vs_repeat."""
    return [load_harbor_job(d, run_id=f"job{i}", **kw) for i, d in enumerate(job_dirs)]
