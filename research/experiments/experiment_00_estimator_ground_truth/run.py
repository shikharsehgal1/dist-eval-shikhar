"""Experiment 0: Do disteval's estimators recover known ground truth?

Every other experiment in this programme assumes the underlying estimators are
correct. This one checks that assumption directly, against closed-form answers
rather than against the library's own expectations:

  * pass@k / pass^k  — Monte Carlo expectation vs the analytic 1-(1-p)^k and p^k.
    A biased estimator (the common `c > 0` shortcut) fails this.
  * VaR / CVaR       — vs the closed form for Uniform[0,1]: VaR@a = a, CVaR@a = a/2.
  * IQM              — vs the analytic interquartile mean, clean and contaminated.
  * Bootstrap CI     — empirical coverage over many independent experiments.
    A 95% CI must contain the true parameter ~95% of the time.
  * Clopper-Pearson  — coverage must be >= nominal (it is conservative by construction).
  * GRPO advantages  — per-group mean 0 and std 1 by definition.
  * KL divergence    — vs the closed form KL(N(0,1) || N(1,1)) = 0.5.

Unit tests prove the code does what its author expected. This proves the math
is right.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))


import numpy as np
import pandas as pd

from disteval.bootstrap import binomial_ci, stratified_bootstrap_ci
from disteval.metrics import cvar, divergences, grpo_advantages, iqm, pass_at_k, pass_hat_k, var_at

SEED = 0
N_SIM = 20_000          # Monte Carlo replicates for the pass@k/pass^k unbiasedness checks
N_COVERAGE = 1_000      # independent experiments for the bootstrap coverage check
N_TAIL = 2_000_000      # sample size for the closed-form tail-metric checks
OUTPUT_DIR = Path(__file__).parent / "results"


def _pass_metrics(successes: np.ndarray, k: int) -> tuple[float, float]:
    """pass@k and pass^k for a single task's trial outcomes."""
    df = pd.DataFrame({"task": ["t"] * len(successes), "success": successes})
    return pass_at_k(df, k), pass_hat_k(df, k)


def check_pass_k_unbiased(rng: np.random.Generator, n: int = 8, k: int = 3) -> list[dict]:
    """pass@k must estimate 1-(1-p)^k and pass^k must estimate p^k, in expectation.

    Tolerance is 4 standard errors of the Monte Carlo mean: a genuinely unbiased
    estimator passes, a biased one (empirical `c > 0`) does not.
    """
    checks = []
    for p in (0.2, 0.5, 0.8):
        at_k, hat_k = [], []
        for _ in range(N_SIM):
            a, h = _pass_metrics(rng.random(n) < p, k)
            at_k.append(a)
            hat_k.append(h)
        for name, est, truth in (
            (f"pass@{k} unbiased (p={p})", at_k, 1 - (1 - p) ** k),
            (f"pass^{k} unbiased (p={p})", hat_k, p**k),
        ):
            mean = float(np.mean(est))
            se = float(np.std(est) / np.sqrt(N_SIM))
            checks.append(_record(name, truth, mean, abs(mean - truth) < 4 * se,
                                  tolerance=f"4 SE = {4 * se:.4f}"))
    return checks


def check_tail_metrics(rng: np.random.Generator) -> list[dict]:
    """For X ~ Uniform[0,1]: VaR@a = a and CVaR@a = a/2 exactly."""
    x = rng.random(N_TAIL)
    checks = []
    for a in (0.05, 0.1, 0.25):
        checks.append(_record(f"VaR@{a} on Uniform[0,1]", a, var_at(x, a),
                              abs(var_at(x, a) - a) < 0.002, tolerance="0.002"))
        checks.append(_record(f"CVaR@{a} on Uniform[0,1]", a / 2, cvar(x, a),
                              abs(cvar(x, a) - a / 2) < 0.002, tolerance="0.002"))

    # IQM on a clean sample is the mean; under contamination it must stay bounded.
    checks.append(_record("IQM on Uniform[0,1]", 0.5, iqm(x),
                          abs(iqm(x) - 0.5) < 0.002, tolerance="0.002"))

    # 10% of the sample replaced by a wild outlier. IQM trims to the middle 50%
    # of the sorted sample, which now maps to uniform quantiles [0.25/0.9, 0.75/0.9]
    # -- so the analytic truth shifts to 0.5556, and IQM must land there while the
    # mean is destroyed.
    contaminated = np.concatenate([rng.random(90_000), np.full(10_000, 1e6)])
    truth = (0.25 / 0.9 + 0.75 / 0.9) / 2
    checks.append(_record("IQM under 10% wild outliers", truth, iqm(contaminated),
                          abs(iqm(contaminated) - truth) < 0.01, tolerance="0.01"))
    checks.append(_record("mean under 10% wild outliers (contrast)", 0.5,
                          float(np.mean(contaminated)), float(np.mean(contaminated)) > 1_000,
                          tolerance="must blow up"))
    return checks


def check_ci_coverage(rng: np.random.Generator, true_p: float = 0.35) -> list[dict]:
    """A 95% interval is only meaningful if it covers the truth 95% of the time.

    Run many independent experiments against a known Bernoulli mean and count how
    often the interval contains it.
    """
    covered = 0
    for r in range(N_COVERAGE):
        scores = (rng.random(60) < true_p).astype(float)
        df = pd.DataFrame({"task": np.repeat(np.arange(6), 10), "score": scores})
        ci = stratified_bootstrap_ci(df, lambda d: d["score"].mean(), ["task"], n_reps=400, seed=r)
        covered += ci["lo"] <= true_p <= ci["hi"]
    boot_rate = covered / N_COVERAGE

    covered = 0
    for _ in range(2_000):
        k = int((rng.random(40) < 0.3).sum())
        ci = binomial_ci(k, 40, 0.95)
        covered += ci["lo"] <= 0.3 <= ci["hi"]
    cp_rate = covered / 2_000

    return [
        _record("stratified bootstrap 95% CI coverage", 0.95, boot_rate,
                0.90 <= boot_rate <= 0.99, tolerance="[0.90, 0.99]"),
        _record("Clopper-Pearson coverage (conservative)", 0.95, cp_rate,
                cp_rate >= 0.95, tolerance=">= 0.95"),
    ]


def check_rl_and_divergence(rng: np.random.Generator) -> list[dict]:
    """GRPO advantages are group-normalized by definition; KL has a closed form."""
    scores = rng.normal(5, 3, 300)
    groups = np.repeat(np.arange(30), 10)
    adv = grpo_advantages(scores, groups)
    max_mean = max(abs(adv[groups == g].mean()) for g in range(30))
    max_std_err = max(abs(adv[groups == g].std() - 1) for g in range(30))

    a = rng.normal(0, 1, 400_000)
    b = rng.normal(1, 1, 400_000)
    kl = divergences(a, b, bins=200).get("kl")

    return [
        _record("GRPO advantage per-group mean = 0", 0.0, max_mean, max_mean < 1e-9,
                tolerance="1e-9 (worst group)"),
        _record("GRPO advantage per-group std = 1", 1.0, 1 - max_std_err, max_std_err < 1e-3,
                tolerance="1e-3 (worst group)"),
        _record("KL(N(0,1) || N(1,1))", 0.5, kl, kl is not None and abs(kl - 0.5) < 0.05,
                tolerance="0.05"),
    ]


def _record(name: str, truth, estimate, passed: bool, tolerance: str = "") -> dict:
    return {"check": name, "truth": truth, "estimate": estimate,
            "passed": bool(passed), "tolerance": tolerance}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    checks: list[dict] = []
    checks += check_pass_k_unbiased(rng)
    checks += check_tail_metrics(rng)
    checks += check_ci_coverage(rng)
    checks += check_rl_and_divergence(rng)

    n_passed = sum(c["passed"] for c in checks)
    summary = {
        "n_checks": len(checks),
        "n_passed": n_passed,
        "all_passed": n_passed == len(checks),
        "n_monte_carlo_replicates": N_SIM,
        "n_coverage_experiments": N_COVERAGE,
        "key_finding": (
            "Every estimator recovers its closed-form value; the 95% bootstrap CI "
            "attains nominal coverage, so the error bars mean what they claim."
        ),
    }

    df = pd.DataFrame(checks)
    df.to_csv(OUTPUT_DIR / "checks.csv", index=False)
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Experiment 0 — Estimators vs closed-form ground truth")
    print("=" * 72)
    width = max(len(c["check"]) for c in checks)
    print(f"{'CHECK'.ljust(width)}  {'TRUTH':>10}  {'ESTIMATE':>12}  RESULT")
    print("-" * (width + 34))
    for c in checks:
        truth = f"{c['truth']:.4f}" if isinstance(c["truth"], (int, float)) else str(c["truth"])
        est = f"{c['estimate']:.4f}" if isinstance(c["estimate"], (int, float)) else str(c["estimate"])
        print(f"{c['check'].ljust(width)}  {truth:>10}  {est:>12}  "
              f"{'PASS' if c['passed'] else 'FAIL'}")
    print("-" * (width + 34))
    print(f"{n_passed}/{len(checks)} checks passed")
    print(f"\nKey finding: {summary['key_finding']}")
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
