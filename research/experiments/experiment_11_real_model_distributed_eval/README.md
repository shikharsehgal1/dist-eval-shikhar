# Experiment 11 — Real-model distributed eval

**Question:** Does disteval's distributed aggregation (IVW / robust Huber) work
better than a naive mean on *real* agent outcomes — and does the pipeline hold
up end-to-end on live data?

**Setup:** Four real agents ran 4 episodes on each of 4 deterministic,
exactly-verifiable tasks (fizzbuzz-15, sort-desc, count-r, base-7) — 64 live
episodes total:

- `cli-opus`, `cli-haiku` — Claude Code CLI headless (agent-wrapped models
  with tools/thinking scaffold, CLI auth)
- `api-opus`, `api-haiku` — raw models via the Anthropic SDK
  (`claude-opus-4-8`, `claude-haiku-4-5`; key read from env, never stored)

Outputs are scored on two channels: **strict** (exact output match —
instruction compliance) and **lenient** (formatting-normalized — capability).
Records flow through `metrics.summarize`, `right_tail_analysis`, and
`DistributedEvalPool` (naive / IVW / Huber aggregation + cross-agent pairs).

Re-run analysis offline without API calls: `python3 run.py --rescore`.
Redraw the charts embedded in the top-level README: `python3 plots.py`.

## Findings

1. **The scaffold is worth a full STUCK task.** Both raw-API models fail
   letter-counting (count-r) in 8/8 episodes (answering 5–6 vs the true 8) —
   a genuine capability gap — while both CLI agents solve it in 8/8 (they can
   count programmatically). The cross-agent pair generator attributes this
   exactly: `count-r: cli-opus (1.00) > api-opus (0.00)`.
2. **Strict/lenient scoring split matters on real models.** 7/64 episodes had
   the right content in the wrong format (markdown fences, bold `**202**`,
   shown work before the answer). Raw api-opus's base-7 "failures" were 100%
   formatting: capability 0.750 vs strict 0.562. Scoring only strict conflates
   compliance with capability — the harness reports both, with
   `format_noncompliance` as a distinct failure mode.
3. **Distribution metrics beat the mean on real data.** cli-opus IQM is 1.000
   vs mean 0.887 — the robust center says the model is solid; the mean is
   dragged by flaky fizzbuzz tail episodes (the mean-collapse thesis, live).
4. **Robust distributed aggregation works better under contamination.** With
   15% of scores zeroed (simulating Harbor `missing_reward` infra failures),
   Huber M-estimation tracks the clean per-task truth **59% closer** than the
   naive mean (MAE 0.049 vs 0.120 over 200 trials); IVW ≈ naive (0.120) —
   contamination corrupts the variance estimates IVW relies on.
5. **Real data surfaced two library bugs** (both fixed with regression tests):
   the Huber IRLS weight at zero residual was ~0 instead of 1, and cross-agent
   pair generation missed pairs when two agents tie at the max score.
