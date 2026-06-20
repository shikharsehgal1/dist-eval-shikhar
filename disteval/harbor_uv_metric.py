"""Drop-in Harbor UV_SCRIPT metric: distribution summary instead of a bare mean.

Harbor's UV_SCRIPT metric writes all trial rewards to a JSONL file (one JSON
value per line, `null` for missing) and calls:

    uv run harbor_uv_metric.py -i <in.jsonl> -o <out.json>

This script reads those rewards and emits a *distribution* of metrics
(mean, IQM, std, VaR/CVaR tail risk) as a flat {name: float} dict, which Harbor
records as the job metric. This is the in-framework hook (option (a)) from the
viability review -- swap Harbor's MEAN reducer for a distribution-aware one
without forking core.

Run standalone for the demo; in Harbor, register it as a UV_SCRIPT metric.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

# allow running as a loose script (no package context)
try:
    from .metrics import cvar, iqm, var_at
except ImportError:  # pragma: no cover - standalone invocation
    sys.path.insert(0, __file__.rsplit("/", 2)[0])
    from disteval.metrics import cvar, iqm, var_at


def compute(rewards: list[float | None], alpha: float = 0.1) -> dict:
    vals = np.array([r for r in rewards if r is not None], dtype=float)
    n_missing = sum(1 for r in rewards if r is None)
    if vals.size == 0:
        return {"mean_reward": 0.0, "n": 0, "n_missing": float(n_missing)}
    return {
        "mean_reward": float(vals.mean()),          # what Harbor would have reported
        "iqm_reward": iqm(vals),                     # robust center
        "std_reward": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        f"VaR@{alpha}": var_at(vals, alpha),         # tail threshold
        f"CVaR@{alpha}": cvar(vals, alpha),          # mean of worst-alpha tail
        "min_reward": float(vals.min()),
        "frac_zero": float((vals == 0).mean()),      # collapse rate
        "n": float(vals.size),
        "n_missing": float(n_missing),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="rewards JSONL (one value/line)")
    ap.add_argument("-o", "--output", required=True, help="metric JSON out")
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    rewards: list[float | None] = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = json.loads(line)
            if isinstance(v, dict):  # named rewards -> take mean of values
                nums = [x for x in v.values() if isinstance(x, (int, float))]
                rewards.append(float(np.mean(nums)) if nums else None)
            else:
                rewards.append(None if v is None else float(v))

    out = compute(rewards, alpha=args.alpha)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
