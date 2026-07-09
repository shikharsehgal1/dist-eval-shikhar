#!/usr/bin/env python3
"""Experiment 11 — real-model distributed eval.

Runs two REAL agents (Claude Code CLI headless: opus and haiku) on four
deterministic, exactly-verifiable tasks, several episodes each, then feeds the
outcomes through disteval's full distributed pipeline:

  1. per-agent distribution summaries (metrics.summarize)
  2. right-tail SOLID/RECOVERABLE/STUCK classification
  3. DistributedEvalPool aggregation: naive mean vs inverse-variance weighted
     vs robust Huber M-estimation
  4. cross-agent training pairs
  5. the "does distributed aggregation work better?" check: corrupt a fraction
     of scores (simulating flaky scoring / infra failures, like Harbor's
     missing_reward) and measure which aggregator stays closest to the clean
     per-task truth.

Requires the `claude` CLI on PATH with valid auth. Results land in results/.
"""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parent.parent.parent))

N_EPISODES = 4
AGENTS = {
    "claude-opus": "opus",
    "claude-haiku": "haiku",
}


# ── Tasks: deterministic prompts with exact expected outputs ─────────────────
#
# Two scoring channels per task:
#   strict  — the raw output must match exactly (measures instruction
#             compliance: the prompt says "ONLY ...").
#   lenient — normalizes formatting artifacts first (markdown code fences,
#             shown work before a final answer) and scores the content.
# QA on the first real run showed strict-only scoring conflates formatting
# compliance with capability (an agent answering correctly inside ``` fences
# scored 0.0) — report both, never just one.

def _normalize(out: str) -> str:
    """Strip markdown code fences and surrounding whitespace."""
    lines = [ln for ln in out.strip().splitlines()]
    lines = [ln for ln in lines if not ln.strip().startswith("```")]
    return "\n".join(lines).strip()


def _score_exact(expected: str):
    def strict(out: str) -> float:
        return 1.0 if out.strip() == expected else 0.0

    def lenient(out: str) -> float:
        norm = _normalize(out)
        if norm == expected:
            return 1.0
        # Accept a correct final line after shown work.
        last = norm.splitlines()[-1].strip() if norm else ""
        return 1.0 if last == expected else 0.0

    return strict, lenient


def _score_lines(expected_lines: list[str]):
    """Fraction of lines matching, position-wise — graded, not binary."""
    def frac(text: str) -> float:
        got = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if not got:
            return 0.0
        hits = sum(1 for i, e in enumerate(expected_lines) if i < len(got) and got[i] == e)
        return hits / len(expected_lines)

    def strict(out: str) -> float:
        return frac(out)

    def lenient(out: str) -> float:
        return frac(_normalize(out))

    return strict, lenient


_FIZZBUZZ_15 = [
    "1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz",
    "11", "Fizz", "13", "14", "FizzBuzz",
]

TASKS = {
    "fizzbuzz-15": {
        "prompt": (
            "Output FizzBuzz from 1 to 15, one entry per line. Multiples of 3 -> Fizz, "
            "multiples of 5 -> Buzz, both -> FizzBuzz. Output ONLY the 15 lines, nothing else."
        ),
        "score": _score_lines(_FIZZBUZZ_15),
    },
    "sort-desc": {
        "prompt": (
            "Sort these numbers in DESCENDING order and reply with ONLY the sorted list, "
            "comma-separated with no spaces: 17,3,42,8,25,1,33"
        ),
        "score": _score_exact("42,33,25,17,8,3,1"),
    },
    "count-r": {
        "prompt": (
            "How many times does the letter 'r' appear in the text: strawberry blueberry raspberry\n"
            "Reply with ONLY the number."
        ),
        "score": _score_exact("8"),
    },
    "base-7": {
        "prompt": "Convert the decimal number 100 to base 7. Reply with ONLY the base-7 digits.",
        "score": _score_exact("202"),
    },
}


def score_output(task: str, output: str, ok: bool) -> tuple[float, float]:
    strict_fn, lenient_fn = TASKS[task]["score"]
    if not ok:
        return 0.0, 0.0
    return float(strict_fn(output)), float(lenient_fn(output))


def run_episode(agent: str, model_flag: str, task: str, episode: int) -> dict:
    spec = TASKS[task]
    try:
        proc = subprocess.run(
            ["claude", "-p", spec["prompt"], "--model", model_flag],
            capture_output=True, text=True, timeout=180,
        )
        output = proc.stdout
        strict, lenient = score_output(task, output, proc.returncode == 0)
        if proc.returncode != 0:
            failure = "cli_error"
        elif lenient < 0.99:
            failure = "wrong_output"
        elif strict < 0.99:
            failure = "format_noncompliance"  # right content, wrong format
        else:
            failure = None
    except subprocess.TimeoutExpired:
        output, strict, lenient, failure = "", 0.0, 0.0, "timeout"
    return {
        "run_id": "real_run_0",
        "model": agent,
        "task": task,
        "episode": episode,
        "score": round(lenient, 4),          # capability is the headline score
        "success": lenient >= 0.99,
        "failure_mode": failure,
        "strict_score": round(strict, 4),    # promoted to a stratification key
        "metadata": {"raw_output": output.strip()[:500]},
    }


def rescore(records: list[dict]) -> list[dict]:
    """Re-score saved records offline from their stored raw_output."""
    out = []
    for r in records:
        raw = r.get("metadata", {}).get("raw_output", "")
        ok = r.get("failure_mode") not in ("cli_error", "timeout")
        strict, lenient = score_output(r["task"], raw, ok)
        r = dict(r)
        r["score"] = round(lenient, 4)
        r["success"] = lenient >= 0.99
        r["strict_score"] = round(strict, 4)
        if r["failure_mode"] not in ("cli_error", "timeout"):
            r["failure_mode"] = (
                None if strict >= 0.99
                else "format_noncompliance" if lenient >= 0.99
                else "wrong_output"
            )
        out.append(r)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rec_path = RESULTS / "records.jsonl"

    if "--rescore" in sys.argv and rec_path.exists():
        # ── 1b. Offline re-score of a previous real run (no API calls) ────
        records = [json.loads(ln) for ln in open(rec_path) if ln.strip()]
        records = rescore(records)
        print(f"Re-scored {len(records)} saved records (strict + lenient channels).")
    else:
        # ── 1. Run the real agents ────────────────────────────────────────
        jobs = [
            (agent, flag, task, ep)
            for agent, flag in AGENTS.items()
            for task in TASKS
            for ep in range(N_EPISODES)
        ]
        print(f"Running {len(jobs)} real episodes ({len(AGENTS)} agents x {len(TASKS)} tasks x {N_EPISODES} eps)...")
        with ThreadPoolExecutor(max_workers=4) as ex:
            records = list(ex.map(lambda j: run_episode(*j), jobs))

    with open(rec_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} records -> {rec_path}")

    n_format = sum(1 for r in records if r.get("failure_mode") == "format_noncompliance")
    print(f"Format-noncompliance episodes (right content, wrong format): {n_format}")

    # ── 2. Per-agent distribution summaries + right tail ──────────────────
    from disteval import metrics
    from disteval.adapters.generic import load_records
    from disteval.right_tail import right_tail_analysis

    store = load_records(str(rec_path))
    df = store.df()
    summary: dict = {"agents": {}}
    for agent in AGENTS:
        adf = df[df["model"] == agent]
        s = metrics.summarize(adf, ks=(1, 2, 4))
        rt = right_tail_analysis(
            __import__("disteval").records.RecordStore(
                [r for r in store._records if r.model == agent]
            ),
            model_name=agent,
        )
        strict_mean = float(np.mean([r["strict_score"] for r in records if r["model"] == agent]))
        summary["agents"][agent] = {
            "metrics": s,
            "strict_mean": strict_mean,
            "right_tail": {
                "n_solid": rt.n_solid, "n_recoverable": rt.n_recoverable,
                "n_stuck": rt.n_stuck, "consistency_index": rt.consistency_index,
            },
        }
        print(f"\n{agent}: capability mean={s['mean']:.3f} strict mean={strict_mean:.3f} "
              f"IQM={s['iqm']:.3f} pass^2={s['pass^2']:.3f} kappa={rt.consistency_index:.3f} "
              f"({rt.n_solid}S/{rt.n_recoverable}R/{rt.n_stuck}X)")

    # ── 3. Distributed pool: three aggregators + cross-agent pairs ────────
    from disteval.distributed_eval import DistributedEvalPool, DistributedEvalRecord

    def build_pool(recs) -> DistributedEvalPool:
        pool = DistributedEvalPool()
        for r in recs:
            pool.add(DistributedEvalRecord(
                agent_name=r["model"], model_name=r["model"], task=r["task"],
                score=r["score"], success=r["success"], failure_mode=r["failure_mode"],
            ))
        return pool

    pool = build_pool(records)
    naive = {a.task: a.mean_score for a in pool.aggregate_by_task()}
    ivw = {a.task: a.mean_score for a in pool.aggregate_by_task_ivw()}
    huber = {a.task: a.mean_score for a in pool.aggregate_by_task_robust()}
    pairs = pool.generate_cross_agent_pairs(min_gap=0.1)
    summary["aggregation"] = {"naive": naive, "ivw": ivw, "huber": huber}
    summary["cross_agent_pairs"] = [
        {"task": p.task, "positive": p.positive_agent, "negative": p.negative_agent,
         "gap": p.gap} for p in pairs
    ]
    print(f"\nCross-agent pairs found: {len(pairs)}")
    for p in pairs:
        print(f"  {p.task}: {p.positive_agent} ({p.positive_score:.2f}) > "
              f"{p.negative_agent} ({p.negative_score:.2f}), gap={p.gap:.2f}")

    # ── 4. Does distributed aggregation work better? Contamination study ──
    # Corrupt a fraction of scores (score -> 0, as Harbor's missing_reward
    # does) and measure each aggregator's mean absolute error vs the CLEAN
    # per-task aggregate. Robust methods should degrade less than naive mean.
    rng = np.random.default_rng(42)
    clean_truth = naive  # per-task means of the uncorrupted real data
    n_trials, frac = 200, 0.15
    errs = {"naive": [], "ivw": [], "huber": []}
    for _ in range(n_trials):
        corrupted = [dict(r) for r in records]
        idx = rng.choice(len(corrupted), size=max(1, int(frac * len(corrupted))), replace=False)
        for i in idx:
            corrupted[i]["score"] = 0.0
        cpool = build_pool(corrupted)
        for name, agg in [
            ("naive", cpool.aggregate_by_task()),
            ("ivw", cpool.aggregate_by_task_ivw()),
            ("huber", cpool.aggregate_by_task_robust()),
        ]:
            vals = {a.task: a.mean_score for a in agg}
            errs[name].append(np.mean([abs(vals[t] - clean_truth[t]) for t in clean_truth]))

    contamination = {k: {"mae": float(np.mean(v)), "std": float(np.std(v))} for k, v in errs.items()}
    summary["contamination_study"] = {
        "n_trials": n_trials, "corrupt_fraction": frac, "results": contamination,
        "huber_beats_naive": contamination["huber"]["mae"] < contamination["naive"]["mae"],
    }
    print(f"\nContamination study ({frac:.0%} of scores zeroed, {n_trials} trials):")
    for k, v in contamination.items():
        print(f"  {k:6s} MAE vs clean truth: {v['mae']:.4f} (+/- {v['std']:.4f})")

    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary -> {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()
