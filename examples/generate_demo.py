#!/usr/bin/env python3
"""Generate the demo dataset shipped in ``examples/``.

Deterministic (fixed seed), so re-running it reproduces the committed files
byte-for-byte. The dataset is small enough to read by hand and rich enough that
every analysis in the package has something to work on.

It is explicitly **synthetic**: no benchmark numbers are implied, no real agent
was run, and every file says so in its own metadata. It exists so that
``pip install -e . && python -m disteval evaluate examples/runs.json`` works
immediately after cloning, without an API key.

Shape:
  * 24 tasks across 4 professional domains (finance, legal, research, ops)
  * 8 runs per task, plus 3 tasks deliberately given only 2 runs so the
    UNCERTAIN path is exercised
  * one clearly SOLID task, one clearly STUCK task, several RECOVERABLE ones
  * per-criterion rubric scores with a declared dependency (r_retrieve -> r_report)
  * full event-level trajectories for every run
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SEED = 20240917

DOMAINS = {
    "finance": ["reconcile_ledger", "quarterly_variance", "fx_exposure", "audit_sample",
                "invoice_match", "cashflow_forecast"],
    "legal": ["contract_diff", "clause_extraction", "precedent_search", "redline_review",
              "obligation_matrix", "nda_compare"],
    "research": ["literature_sweep", "data_extraction", "figure_reproduce", "citation_audit",
                 "method_summary", "benchmark_table"],
    "ops": ["incident_triage", "runbook_execute", "capacity_plan", "log_rootcause",
            "access_review", "sla_report"],
}

#: (latent success prob, execution reliability, failure mode, n_runs).
#: Chosen so the demo contains one of each category, plus low-sample tasks.
PROFILES = {
    "reconcile_ledger": (0.98, 0.99, "tool_execution", 8),   # SOLID
    "precedent_search": (0.02, 0.30, "retrieval", 8),        # STUCK
    "quarterly_variance": (0.40, 0.45, "retrieval", 8),      # RECOVERABLE, concentrated
    "contract_diff": (0.45, 0.50, "state_tracking", 8),      # RECOVERABLE
    "incident_triage": (0.38, 0.42, "recovery", 8),          # RECOVERABLE, diffuse
    "literature_sweep": (0.55, 0.60, "retrieval", 8),        # RECOVERABLE
    "fx_exposure": (0.50, 0.55, "reasoning", 2),             # UNCERTAIN (low n)
    "redline_review": (0.60, 0.65, "verification", 2),       # UNCERTAIN (low n)
    "capacity_plan": (0.30, 0.35, "planning", 2),            # UNCERTAIN (low n)
}
DEFAULT_PROFILE = (0.55, 0.62, "tool_execution", 8)

CRITERIA = ["r_retrieve", "r_compute", "r_verify", "r_report"]
CRITERION_DEPENDENCIES = [["r_retrieve", "r_report"], ["r_compute", "r_report"]]

CANONICAL = [
    ("plan", "plan", None),
    ("search", "retrieval", "corpus/index"),
    ("read", "retrieval", "corpus/q3_actuals.csv"),
    ("compute", "tool_call", "workbook/sheet1"),
    ("verify", "verification", "workbook/sheet1"),
    ("write", "final_answer", "report.md"),
]

#: Where each failure mode makes the run go wrong, as a fraction of its length.
MODE_POSITION = {
    "retrieval": 0.25, "planning": 0.10, "reasoning": 0.45,
    "tool_selection": 0.35, "tool_execution": 0.55, "state_tracking": 0.6,
    "memory": 0.5, "verification": 0.8, "recovery": 0.7, "synthesis": 0.9,
}


def build():
    rng = np.random.default_rng(SEED)
    tasks, runs, trajectories = [], [], []

    for domain, names in DOMAINS.items():
        for name in names:
            p, r_exec, mode, n_runs = PROFILES.get(name, DEFAULT_PROFILE)
            q = min(p / max(r_exec, 1e-6), 1.0)
            complexity = int(rng.integers(6, 26))
            tasks.append(
                {
                    "task_id": name,
                    "domain": domain,
                    "environment": f"{domain}_env_v1",
                    "instruction": f"[synthetic] complete the {name.replace('_', ' ')} task",
                    "rubric_criteria": CRITERIA,
                    "criterion_dependencies": CRITERION_DEPENDENCIES,
                    "complexity": complexity,
                    "tools": ["search", "read", "compute", "verify", "write"],
                    "synthetic": True,
                }
            )

            for ep in range(n_runs):
                reached = bool(rng.random() < q)
                success = bool(reached and rng.random() < r_exec)
                # A run that fails always fails the same way for a
                # concentration-heavy task, and variably for a diffuse one.
                if success:
                    fmode = None
                elif name == "incident_triage":
                    fmode = str(rng.choice(
                        ["planning", "reasoning", "memory", "recovery", "synthesis"]
                    ))
                else:
                    fmode = mode if rng.random() < 0.75 else str(
                        rng.choice(["tool_execution", "verification"])
                    )

                n_steps = max(4, int(rng.poisson(complexity)))
                div = n_steps if success else max(
                    1, int(n_steps * MODE_POSITION.get(fmode, 0.5))
                )

                rubric = {}
                for i, c in enumerate(CRITERIA):
                    if success:
                        rubric[c] = 1.0
                    elif c == "r_retrieve":
                        rubric[c] = 0.0 if fmode == "retrieval" else 1.0
                    elif c == "r_compute":
                        rubric[c] = 0.0 if fmode in ("reasoning", "tool_execution") else 1.0
                    elif c == "r_verify":
                        rubric[c] = 0.0 if fmode in ("verification", "recovery") else 1.0
                    else:  # r_report depends on the two upstream criteria
                        rubric[c] = float(
                            rubric["r_retrieve"] > 0 and rubric["r_compute"] > 0
                        ) * (0.0 if fmode == "synthesis" else 1.0)

                score = 1.0 if success else 0.0
                tid = f"{name}#{ep}"
                cost_usd = round(0.018 * n_steps * float(rng.gamma(4, 0.25)), 4)

                runs.append(
                    {
                        "run_id": "demo_run",
                        "task_id": name,
                        "model": "demo-agent-v1",
                        "episode": ep,
                        "score": score,
                        "success": success,
                        "domain": domain,
                        "environment": f"{domain}_env_v1",
                        "rubric_scores": rubric,
                        "cost": {"usd": cost_usd, "tool_calls": n_steps},
                        "n_steps": n_steps,
                        "failure_mode": fmode,
                        "trajectory_ref": tid,
                        "milestone": reached,
                    }
                )

                events = []
                for i in range(n_steps):
                    tool, etype, target = CANONICAL[i % len(CANONICAL)]
                    off = i >= div
                    events.append(
                        {
                            "index": i,
                            "event_type": "error" if (off and i == div) else etype,
                            "tool_name": tool,
                            "tool_args": {"step": i, "variant": "alt" if off else "std"},
                            "target": (
                                (target or f"{name}/step{i}").replace("q3", "q2")
                                if off else (target or f"{name}/step{i}")
                            ),
                            "ok": not (off and i == div),
                            "observation": (
                                f"{fmode} failure while running {tool}" if (off and i == div)
                                else f"{tool} completed"
                            ),
                            "state_features": {
                                "progress": round((i + 1) / n_steps * (0.4 if off else 1.0), 4)
                            },
                            "rubric_state": {c: rubric[c] for c in CRITERIA} if i == n_steps - 1 else {},
                            "cost": {"usd": round(cost_usd / n_steps, 6)},
                        }
                    )
                trajectories.append(
                    {
                        "trajectory_id": tid,
                        "task_id": name,
                        "model": "demo-agent-v1",
                        "run_id": "demo_run",
                        "episode": ep,
                        "score": score,
                        "success": success,
                        "domain": domain,
                        "rubric_scores": rubric,
                        "cost": {"usd": cost_usd},
                        "metadata": {
                            "failure_mode": fmode,
                            "environment": f"{domain}_env_v1",
                            "synthetic": True,
                        },
                        "events": events,
                    }
                )

    return tasks, runs, trajectories


def main() -> None:
    tasks, runs, trajectories = build()
    banner = {
        "_README": (
            "SYNTHETIC demo data generated by examples/generate_demo.py. No real "
            "agent was run and no benchmark numbers are implied. Regenerate with: "
            "python examples/generate_demo.py"
        ),
        "_seed": SEED,
    }
    (HERE / "tasks.json").write_text(
        json.dumps({**banner, "tasks": tasks}, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "runs.json").write_text(
        json.dumps({**banner, "runs": runs}, indent=2) + "\n", encoding="utf-8"
    )
    with open(HERE / "trajectories.jsonl", "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t) + "\n")
    print(
        f"wrote {len(tasks)} tasks, {len(runs)} runs, {len(trajectories)} trajectories "
        f"to {HERE}"
    )


if __name__ == "__main__":
    main()
