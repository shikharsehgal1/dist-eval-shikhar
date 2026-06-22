"""Seam: Inspect (UK AISI) `.eval` / `.json` logs  ->  RecordStore.

Inspect already persists the richest per-sample records in the ecosystem
(EvalSample: score, metadata, transcript, multi-epoch reductions). We do NOT
reinvent that store -- we consume it. Inspect's headline `results` field is
scalar; the per-sample distribution lives in `samples`, which is what we lift.

Inspect log JSON shape (abridged, from inspect_ai eval logs):
  {
    "eval":   {"model": "...", "task": "...", "run_id": "..."},
    "samples": [
        {"id": ..., "epoch": 1,
         "score": {"value": 1.0 | "C"/"I", ...},
         "metadata": {"difficulty": "hard", ...},
         "error": null | {...}},
        ...
    ]
  }

If inspect_ai is installed you'd instead use `inspect_ai.log.read_eval_log(path)`
and iterate `log.samples`; the field mapping is identical.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from ..records import EpisodeRecord, RecordStore


def _default_score(sample: dict) -> float:
    sc = sample.get("score", {})
    v = sc.get("value", sc) if isinstance(sc, dict) else sc
    if isinstance(v, (int, float)):
        return float(v)
    # Inspect uses "C"/"I" (correct/incorrect) for many scorers
    return 1.0 if str(v).upper() in ("C", "CORRECT", "TRUE", "1") else 0.0


def load_inspect_json(
    path: str,
    strata_keys: tuple[str, ...] = (),
    score_fn: Callable[[dict], float] = _default_score,
    success_threshold: float = 1.0,
    failure_classifier: Optional[Callable[[dict], Optional[str]]] = None,
) -> RecordStore:
    """Parse an Inspect eval-log JSON into a RecordStore.

    strata_keys: which sample.metadata keys to promote to first-class strata.
    failure_classifier: optional fn(sample)->failure_mode label for non-success.
    """
    with open(path) as f:
        log = json.load(f)

    ev = log.get("eval", {})
    model = ev.get("model", "unknown")
    run_id = ev.get("run_id", ev.get("eval_id", "run0"))

    store = RecordStore()
    per_task_counter: dict[str, int] = {}
    for s in log.get("samples", []):
        task = str(s.get("id", s.get("sample_id", "task")))
        score = score_fn(s)
        success = score >= success_threshold and s.get("error") is None
        meta = s.get("metadata", {}) or {}
        strata = {k: meta.get(k) for k in strata_keys if k in meta}
        epoch = s.get("epoch", per_task_counter.get(task, 0))
        per_task_counter[task] = per_task_counter.get(task, 0) + 1
        fmode = None
        if not success:
            fmode = (failure_classifier(s) if failure_classifier
                     else (s.get("error", {}) or {}).get("type", "unlabeled")
                     if isinstance(s.get("error"), dict) else "incorrect")
        store.add(EpisodeRecord(
            run_id=run_id, model=model, task=task, episode=int(epoch),
            score=score, success=bool(success), strata=strata,
            failure_mode=fmode, meta={"inspect_id": s.get("id")},
        ))
    return store
