# Research: Recursive Self-Improvement for disteval

This directory contains the research output for the recursive self-improvement task. The work investigates how disteval's agent improvement engine can be enhanced using recursive reinforcement learning (arXiv:2206.11430) and a recursion engine that produces RL environments from distributed agent evals.

## Quick navigation

| Document | Purpose |
|---|---|
| `phase1_master_report.md` | Summary of RMDP literature mapping to disteval. |
| `phase1a_rmdp_formalism.md` | Formal summary of arXiv:2206.11430. |
| `phase1b_disteval_mapping.md` | 14-row mapping from RMDP concepts to disteval code. |
| `phase1c_integration_questions.md` | Integration questions and highest-leverage changes. |
| `phase2_master_report.md` | Consolidated `RecursionEngine` design. |
| `phase2a_recursion_engine_api.md` | `RecursionEngine` API and data classes. |
| `phase2b_decomposition_algorithm.md` | Sub-task decomposition algorithm. |
| `phase2c_integration_design.md` | Integration with existing disteval files. |
| `phase3_master_report.md` | Consolidated RL environment and self-improvement loop design. |
| `phase3a_environment_schema.md` | Generated RL environment schema (`GenEnv`). |
| `phase3b_self_improvement_loop.md` | Multi-cycle recursive self-improvement loop. |
| `phase3c_distributed_evals.md` | Distributed agent evals and cross-agent sharing. |
| `phase4_master_report.md` | Final prototype integration plan and go/no-go criteria. |
| `phase4a_implementation_plan.md` | Milestone-based implementation plan with code skeletons. |
| `phase4b_end_to_end_validation.md` | End-to-end flow and validation plan. |

## Multi-step task structure

1. **Phase 1** — Map arXiv:2206.11430 (Recursive RL / RMDPs) to disteval's existing `SelfEngine`.
2. **Phase 2** — Design a `RecursionEngine` that decomposes tasks into sub-task RMDPs.
3. **Phase 3** — Design RL environment generation and the self-improvement loop where cycle `n` affects cycle `n+1`.
4. **Phase 4** — Produce a concrete prototype implementation plan and validation strategy.

Each phase's deliverables are inputs to the next phase. The final recommendation is in `phase4_master_report.md`.

## Task specification

The research skill that defined this task is at `.devin/skills/research-recursive-self-improvement/SKILL.md`.

## Status

Research complete. No existing disteval code was modified. Implementation can begin after approval of the Phase 4 plan.
