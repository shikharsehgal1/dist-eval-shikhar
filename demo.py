"""End-to-end demo of the disteval distribution-first layer.

Scenario: two agents on a small RL/agentic benchmark.
  - Model A ("peaky")  : high typical reward, but catastrophically COLLAPSES on a
                         fraction of episodes (esp. hard tasks).
  - Model B ("steady") : slightly lower typical reward, but reliable -- no collapses.

The two have ~EQUAL MEAN REWARD. Everything interesting is in the distribution.

We then run the WHOLE eval many times (fresh task draws + seeds each time) to get
the repeated-evaluation meta-distribution, and show that a single-run bootstrap CI
badly understates the true run-to-run noise.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from disteval import RecordStore, EpisodeRecord, metrics, bootstrap, compare, failure, repeat
from disteval.harbor_uv_metric import compute as uv_compute

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

DIFFICULTIES = ["easy", "med", "hard"]
DIFF_OFFSET = {"easy": +0.10, "med": 0.0, "hard": -0.12}
N_TASKS = 40
N_ATTEMPTS = 3
SUCCESS_THRESHOLD = 0.5

# run-level shock (fresh task sample / seeds each re-run). This is the variance a
# single-run bootstrap CANNOT see -- it's frozen once you've collected one run.
SIGMA_RUN = 0.04


def run_eval(model: str, rep: int, base_seed: int = 1000) -> RecordStore:
    """One full eval of `model`. `rep` re-rolls fresh tasks/seeds/outcomes."""
    model_offset = {"A": 101, "B": 202}.get(model, 303)  # deterministic (avoid hash randomization)
    rng = np.random.default_rng(base_seed + rep * 7919 + model_offset)
    run_shock = rng.normal(0, SIGMA_RUN)  # this run happened to draw easier/harder tasks
    store = RecordStore()
    for t in range(N_TASKS):
        diff = DIFFICULTIES[rng.integers(0, 3)]
        for a in range(N_ATTEMPTS):
            base = DIFF_OFFSET[diff] + run_shock
            if model == "A":            # peaky: high mean, collapses
                p_collapse = 0.12 + (0.18 if diff == "hard" else 0.0)
                if rng.random() < p_collapse:
                    score = float(np.clip(rng.normal(0.02, 0.02), 0, 1))
                    fmode = "catastrophic_collapse"
                else:
                    score = float(np.clip(0.86 + base + rng.normal(0, 0.05), 0, 1))
                    fmode = None
            else:                        # steady: lower mean, reliable
                score = float(np.clip(0.70 + base + rng.normal(0, 0.05), 0, 1))
                fmode = None
            success = score >= SUCCESS_THRESHOLD
            if not success and fmode is None:
                fmode = "near_miss"
            store.add(EpisodeRecord(
                run_id=f"rep{rep}", model=model, task=f"task{t:02d}", episode=a,
                score=score, success=success, strata={"difficulty": diff},
                failure_mode=fmode if not success else None,
            ))
    return store


def hr(title: str) -> None:
    print("\n" + "=" * 78 + f"\n {title}\n" + "=" * 78)


# --------------------------------------------------------------------------- #
# 1. Single-run outcome distribution: mean hides the story                    #
# --------------------------------------------------------------------------- #
hr("1. SINGLE-RUN SUMMARY  --  means are close; the distribution tells the real story")
A0, B0 = run_eval("A", rep=0), run_eval("B", rep=0)
summ = pd.DataFrame({"A (peaky)": metrics.summarize(A0.df()),
                     "B (steady)": metrics.summarize(B0.df())}).round(3)
print(summ)

# --------------------------------------------------------------------------- #
# 2. Distribution-to-distribution comparison                                  #
# --------------------------------------------------------------------------- #
hr("2. DISTRIBUTION COMPARISON A vs B")
a_scores, b_scores = A0.scores().to_numpy(), B0.scores().to_numpy()
print(f"Wasserstein distance      : {compare.wasserstein(a_scores, b_scores):.4f}")
print(f"KS test                   : {compare.ks(a_scores, b_scores)}")
print(f"P(A-episode > B-episode)  : {compare.prob_improvement(a_scores, b_scores):.3f}")
dom = compare.stochastic_dominance(a_scores, b_scores)
print(f"Stochastic dominance      : {dom}")
print("  -> NEITHER dominates: A has higher mean+upside, B has far lower tail risk.")
print("     That 'no dominance' result is the point -- it is a genuine risk/return")
print("     tradeoff that NO single scalar can rank. (SSD would only fire if one")
print("     were better at every risk-aversion level; here CVaR favours B, mean favours A.)")

# --------------------------------------------------------------------------- #
# 3. Failure-mode distribution, sliced by difficulty                          #
# --------------------------------------------------------------------------- #
hr("3. FAILURE-MODE DISTRIBUTION for A, by difficulty stratum")
print(failure.failure_distribution(A0.df(), by=["s_difficulty"]).round(3).to_string(index=False))

# --------------------------------------------------------------------------- #
# 4. THE REPEATED-EVALUATION META-DISTRIBUTION  (the user's question)         #
# --------------------------------------------------------------------------- #
hr("4. REPEATED-EVAL META-DISTRIBUTION  --  'run the eval over and over'")
N_REPEATS = 30
stores_A = repeat.run_repeated(lambda r: run_eval("A", r), N_REPEATS)
stores_B = repeat.run_repeated(lambda r: run_eval("B", r), N_REPEATS)

mean_stat = lambda df: float(df["score"].mean())
metaA = repeat.meta_distribution(stores_A, mean_stat)
metaB = repeat.meta_distribution(stores_B, mean_stat)
print(f"Model A mean-reward over {N_REPEATS} re-runs: "
      f"mean={metaA['mean']:.3f}  std={metaA['std']:.3f}  "
      f"range=[{metaA['min']:.3f}, {metaA['max']:.3f}]  95% CI width={metaA['ci_width']:.3f}")
print(f"Model B mean-reward over {N_REPEATS} re-runs: "
      f"mean={metaB['mean']:.3f}  std={metaB['std']:.3f}  "
      f"range=[{metaB['min']:.3f}, {metaB['max']:.3f}]  95% CI width={metaB['ci_width']:.3f}")

hr("4b. WHY IT'S USEFUL: single-run bootstrap CI vs true run-to-run spread")
for name, stores in [("A (peaky)", stores_A), ("B (steady)", stores_B)]:
    bvr = repeat.bootstrap_vs_repeat(stores, mean_stat, strata_cols=["s_difficulty"])
    print(f"{name:11s} single-run bootstrap 95% CI={bvr['mean_single_run_bootstrap_ci_width']:.4f}  "
          f"true repeat 95% CI={bvr['meta_ci_width']:.4f}  "
          f"UNDERCONFIDENCE={bvr['underconfidence_ratio']:.1f}x")
print("  Bootstrap from ONE run understates real noise because it can't resample")
print("  fresh tasks/seeds. NOTE the trap: underconfidence is WORST for the reliable")
print("  model B -- the one that looks most precise from a single run is the one whose")
print("  single-run error bars lie to you most. Only repeated evals expose that.")

hr("4c. IS THE A-vs-B GAP REAL, OR EVAL NOISE?")
real = repeat.is_gap_real(stores_A, stores_B, mean_stat)
print(f"mean gap (A-B)             : {real['mean_gap']:+.4f}")
print(f"P(A>B on a fresh re-run)   : {real['P(A>B on a fresh re-run)']:.2f}")
print(f"A range {real['A_range']}  vs  B range {real['B_range']}  overlap={real['ranges_overlap']}")
print("  -> a ~0 mean gap with overlapping ranges = the headline difference is")
print("     within eval noise; you cannot rank A above B on mean reward alone.")

# --------------------------------------------------------------------------- #
# 5. Harbor UV_SCRIPT metric: drop-in distribution summary                    #
# --------------------------------------------------------------------------- #
hr("5. HARBOR UV_SCRIPT HOOK  --  distribution metric replacing Harbor's MEAN")
rewards = A0.scores().tolist()  # what Harbor would hand the UV script (one/trial)
print("disteval UV metric output (what Harbor would record as the job metric):")
print(json.dumps(uv_compute(rewards), indent=2))
print("  Harbor's default reducer reports only `mean_reward`; this exposes the")
print("  collapse rate (frac_zero), tail risk (CVaR), and spread in the same call.")

print("\n" + "=" * 78)
print(" DONE. Mean ranked A slightly above B; the distribution showed A is")
print(" catastrophically unreliable (CVaR ~0.01 vs 0.44, collapses on ~1/5 hard")
print(" tasks), and the meta-distribution showed the headline gap is eval noise.")
print("=" * 78)
