# Experiment 11 — Real-model distributed eval

**Question:** Does disteval's distributed aggregation (IVW / robust Huber) work
better than a naive mean on *real* agent outcomes — and does the pipeline hold
up end-to-end on live data?

**Setup:** Two real agents (Claude Code CLI headless: `opus`, `haiku`) ran 4
episodes on each of 4 deterministic, exactly-verifiable tasks (fizzbuzz-15,
sort-desc, count-r, base-7) — 32 live episodes. Outputs are scored on two
channels: **strict** (exact output match — instruction compliance) and
**lenient** (formatting-normalized — capability). Records flow through
`metrics.summarize`, `right_tail_analysis`, and `DistributedEvalPool` (naive /
IVW / Huber aggregation + cross-agent pairs).

Re-run analysis offline without API calls: `python3 run.py --rescore`.

## Findings

1. **The strict/lenient split matters on real models.** Opus answered correctly
   but wrapped output in markdown fences or showed work before the final answer
   in 3/16 episodes: capability mean 0.842 vs strict mean 0.654. Scoring only
   strict conflates formatting compliance with capability — the harness now
   reports both, with `format_noncompliance` as a distinct failure mode.
2. **Distribution metrics beat the mean on real data.** Opus's IQM is 1.000
   while its mean is 0.842 — the robust center says the model is solid and the
   mean is dragged by tail episodes (exactly the mean-collapse thesis).
3. **Robust distributed aggregation works better under contamination.** Zeroing
   15% of scores (simulating Harbor's `missing_reward` infra failures), Huber
   M-estimation tracks the clean per-task truth ~24% closer than the naive mean
   (MAE 0.089 vs 0.116 over 200 trials); IVW is between (0.110).
4. **Real data surfaced two library bugs** (both fixed with regression tests):
   the Huber IRLS weight at zero residual was ~0 instead of 1, and cross-agent
   pair generation missed pairs when two agents tie at the max score.
