#!/usr/bin/env python3
"""
Unified CLI dispatcher for the disteval package.

Run as ``python -m disteval <subcommand>`` or via the ``disteval`` script.

Reliability-analysis subcommands (the main entry points):
- evaluate:   estimate reliability from run records and write a research report
- diagnose:   per-task posteriors, classifications and recoverability ranking
- select:     build preference pairs from a selection strategy
- experiment: run the full evaluate -> select -> train -> re-evaluate loop
- sweep:      run a matrix of experiments from a sweep config
- simulate:   run the ground-truth validation suite for the estimators
- metrics:    print the metric registry (definitions, assumptions, edge cases)

Pre-existing subcommands, unchanged:
- report:  generate evaluation reports from a Harbor jobs directory
- compare: generate comparison reports
- sim:     run training simulations
- engine:  run SelfEngine on Harbor job directories
- train:   train a policy from a SelfImprovementPlan using a DPO trainer
"""

import argparse
import json
import sys


def main() -> None:
    """Main CLI dispatcher that routes to appropriate subcommand handlers."""
    if len(sys.argv) == 1:
        print_help_and_exit()
    
    # Handle help for subcommands
    if len(sys.argv) >= 2 and sys.argv[1] in ['-h', '--help']:
        print_help_and_exit()
    
    # Handle help for specific subcommand
    if len(sys.argv) >= 3 and sys.argv[2] in ['-h', '--help']:
        subcommand = sys.argv[1]
        if subcommand in _NEW_HANDLERS:
            _NEW_HANDLERS[subcommand](sys.argv[2:])
            sys.exit(0)
        if subcommand == "engine":
            print_engine_help_and_exit()
        elif subcommand == "report":
            print_report_help_and_exit()
        elif subcommand == "compare":
            print_compare_help_and_exit()
        elif subcommand == "sim":
            print_sim_help_and_exit()
        elif subcommand == "train":
            print_train_help_and_exit()
        else:
            print_help_and_exit()
    
    # Parse subcommand and route
    subcommand = sys.argv[1] if len(sys.argv) > 1 else None
    remaining_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    if not subcommand:
        print_help_and_exit()
    
    # Route to appropriate subcommand handler
    if subcommand in _NEW_HANDLERS:
        _NEW_HANDLERS[subcommand](remaining_args)
    elif subcommand == "report":
        handle_report(remaining_args)
    elif subcommand == "compare":
        handle_compare(remaining_args)
    elif subcommand == "sim":
        handle_sim(remaining_args)
    elif subcommand == "engine":
        handle_engine(remaining_args)
    elif subcommand == "train":
        handle_train(remaining_args)
    else:
        print(f"Unknown subcommand: {subcommand}", file=sys.stderr)
        print_help_and_exit(error=True)


def print_help_and_exit(error: bool = False) -> None:
    """Print main help message and exit."""
    help_text = """usage: disteval [-h] SUBCOMMAND ...

Distributional evaluation and reliability-aware post-training for long-horizon
AI agents.

Reliability analysis:
  evaluate    Estimate reliability from run records; write a research report
  diagnose    Per-task posteriors, classification and recoverability ranking
  select      Select training tasks and build preference pairs
  experiment  Run the full evaluate -> select -> train -> re-evaluate loop
  sweep       Run a matrix of experiments from a sweep config
  simulate    Validate the estimators against known ground truth
  metrics     Print the metric registry (definitions, assumptions, edge cases)

Existing tooling:
  report      Generate evaluation reports
  compare     Generate comparison reports
  sim         Run training simulations
  engine      Run SelfEngine on Harbor job directories
  train       Train a policy from a SelfImprovementPlan using a DPO trainer

Quickstart:
  python -m disteval evaluate examples/runs.json --trajectories examples/trajectories.jsonl

Use 'disteval <subcommand> --help' for more information on a specific command.
"""
    if error:
        print(help_text, file=sys.stderr)
        sys.exit(1)
    else:
        print(help_text)
        sys.exit(0)


def print_engine_help_and_exit() -> None:
    """Print engine subcommand help and exit."""
    help_text = """usage: disteval engine [-h] [--agent AGENT] [--model MODEL] [--tasks-dir TASKS_DIR]
                        [--output OUTPUT] [--cycle CYCLE] [--enable-recursion]
                        [--max-depth MAX_DEPTH]
                        job_dirs [job_dirs ...]

Run SelfEngine on Harbor job directories to generate improvement plans

positional arguments:
  job_dirs              One or more Harbor job directories containing evaluation results

optional arguments:
  -h, --help            show this help message and exit
  --agent AGENT         Agent name (default: agent)
  --model MODEL         Model name (default: unknown)
  --tasks-dir TASKS_DIR Directory containing task definitions (default: tasks)
  --output OUTPUT, -o OUTPUT
                        Output path for the improvement plan JSON (default: improvement_plan.json)
  --cycle CYCLE         SelfEngine cycle number (default: 1)
  --enable-recursion    Enable recursive sub-task decomposition (default: disabled)
  --max-depth MAX_DEPTH Maximum recursion depth for sub-task decomposition (default: 3)
"""
    print(help_text)
    sys.exit(0)


_SIMPLE_SUBCOMMAND_DESCRIPTIONS: dict[str, str] = {
    "report": "Generate evaluation reports",
    "compare": "Generate comparison reports",
    "sim": "Run training simulations",
}


def _print_simple_subcommand_help_and_exit(subcommand: str) -> None:
    """Print a minimal help message for simple subcommands and exit."""
    desc = _SIMPLE_SUBCOMMAND_DESCRIPTIONS.get(subcommand, subcommand)
    print(
        f"usage: disteval {subcommand} [-h]\n\n{desc}\n\n"
        "optional arguments:\n  -h, --help  show this help message and exit\n"
    )
    sys.exit(0)


def print_report_help_and_exit() -> None:
    """Print report subcommand help and exit."""
    _print_simple_subcommand_help_and_exit("report")


def print_compare_help_and_exit() -> None:
    """Print compare subcommand help and exit."""
    _print_simple_subcommand_help_and_exit("compare")


def print_sim_help_and_exit() -> None:
    """Print sim subcommand help and exit."""
    _print_simple_subcommand_help_and_exit("sim")


def print_train_help_and_exit() -> None:
    """Print train subcommand help and exit."""
    help_text = """usage: disteval train [-h] --curriculum CURRICULUM --trainer {noop,simulated,trl,axolotl}
                      [--output OUTPUT]

Train a policy from a SelfImprovementPlan using a DPO trainer

optional arguments:
  -h, --help            show this help message and exit
  --curriculum CURRICULUM
                        Path to a SelfImprovementPlan JSON file
  --trainer {noop,simulated,trl,axolotl}
                        Trainer backend to use
  --output OUTPUT, -o OUTPUT
                        Output directory for the trained policy artifacts
"""
    print(help_text)
    sys.exit(0)


def handle_report(remaining_args: list[str]) -> None:
    """Delegate to disteval.report.main(), passing through all args."""
    sys.argv = ["disteval-report"] + remaining_args
    try:
        from disteval.report import main as report_main
        report_main()
    except ImportError as e:
        print(f"Error importing report module: {e}", file=sys.stderr)
        sys.exit(1)


def handle_compare(remaining_args: list[str]) -> None:
    """Delegate to disteval.compare_report.main(), passing through all args."""
    sys.argv = ["disteval-compare"] + remaining_args
    try:
        from disteval.compare_report import main as compare_main
        compare_main()
    except ImportError as e:
        print(f"Error importing compare_report module: {e}", file=sys.stderr)
        sys.exit(1)


def handle_sim(remaining_args: list[str]) -> None:
    """Delegate to disteval.training_sim.main(), passing through all args."""
    sys.argv = ["disteval-sim"] + remaining_args
    try:
        from disteval.training_sim import main as sim_main
        sim_main()
    except ImportError as e:
        print(f"Error importing training_sim module: {e}", file=sys.stderr)
        sys.exit(1)


def handle_train(remaining_args: list[str]) -> None:
    """Handle the 'train' subcommand for running a DPO trainer on a curriculum."""
    parser = argparse.ArgumentParser(
        prog="disteval train",
        description="Train a policy from a SelfImprovementPlan using a DPO trainer",
    )
    parser.add_argument(
        "--curriculum",
        required=True,
        help="Path to a SelfImprovementPlan JSON file",
    )
    parser.add_argument(
        "--trainer",
        choices=["noop", "simulated", "trl", "axolotl"],
        default="noop",
        help="Trainer backend to use",
    )
    parser.add_argument(
        "--output", "-o",
        default="disteval_train_output",
        help="Output directory for the trained policy artifacts",
    )
    parser.add_argument(
        "--model",
        default="model",
        help="Base model name for trl/axolotl trainers",
    )
    args = parser.parse_args(remaining_args)

    try:
        with open(args.curriculum, "r", encoding="utf-8") as f:
            curriculum = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading curriculum: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        from disteval.training_harness import (
            NoOpTrainer,
            SimulatedTrainer,
            TRLReferenceTrainer,
            AxolotlReferenceTrainer,
            run_training,
        )
        if args.trainer == "noop":
            trainer = NoOpTrainer()
        elif args.trainer == "simulated":
            trainer = SimulatedTrainer()
        elif args.trainer == "trl":
            trainer = TRLReferenceTrainer(args.model)
        else:
            trainer = AxolotlReferenceTrainer(args.model)
        scores = run_training(curriculum, trainer, args.output)
        print("Training complete. Improved scores:")
        for task, score in scores.items():
            print(f"  {task}: {score:.3f}")
    except Exception as e:
        print(f"Error running trainer: {e}", file=sys.stderr)
        sys.exit(1)


def handle_engine(remaining_args: list[str]) -> None:
    """Handle the 'engine' subcommand for running SelfEngine on Harbor job directories."""
    parser = argparse.ArgumentParser(
        prog="disteval engine",
        description="Run SelfEngine on Harbor job directories to generate improvement plans"
    )
    
    parser.add_argument(
        "job_dirs",
        nargs="+",
        help="One or more Harbor job directories containing evaluation results"
    )
    
    parser.add_argument(
        "--agent",
        default="agent",
        help="Agent name (default: agent)"
    )
    
    parser.add_argument(
        "--model",
        default="unknown", 
        help="Model name (default: unknown)"
    )
    
    parser.add_argument(
        "--tasks-dir",
        default="tasks",
        help="Directory containing task definitions (default: tasks)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="improvement_plan.json",
        help="Output path for the improvement plan JSON (default: improvement_plan.json)"
    )
    
    parser.add_argument(
        "--cycle",
        type=int,
        default=1,
        help="SelfEngine cycle number (default: 1)"
    )

    parser.add_argument(
        "--enable-recursion",
        action="store_true",
        default=False,
        help="Enable recursive sub-task decomposition (default: disabled)"
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum recursion depth for sub-task decomposition (default: 3)"
    )

    args = parser.parse_args(remaining_args)

    try:
        # Import SelfEngine (lazy import to avoid circular dependencies)
        from disteval.self_engine import SelfEngine

        print(f"Running SelfEngine cycle {args.cycle} for agent {args.agent}...")

        # Create SelfEngine from job directories
        engine = SelfEngine.from_job_dirs(
            args.job_dirs,
            agent_name=args.agent,
            model_name=args.model,
            tasks_dir=args.tasks_dir,
            enable_recursion=args.enable_recursion,
            recursion_config={"max_depth": args.max_depth},
        )
        
        # Run the specified cycle
        plan = engine.run_cycle(args.cycle)
        
        # Print plan summary
        print("\nImprovement Plan Summary:")
        print(f"  Cycle: {args.cycle}")
        print(f"  Agent: {args.agent}")
        print(f"  Model: {args.model}")
        print(f"  Job directories: {len(args.job_dirs)}")
        if hasattr(plan, 'summary'):
            print(f"  Summary:\n{plan.summary()}")
        
        # Save plan to JSON
        plan_dict = plan.to_dict() if hasattr(plan, 'to_dict') else vars(plan)
        with open(args.output, 'w') as f:
            json.dump(plan_dict, f, indent=2, default=str)
        
        print(f"\nSaved improvement plan → {args.output}")
        
    except ImportError as e:
        print(f"Error importing SelfEngine: {e}", file=sys.stderr)
        print("Make sure disteval is installed correctly (pip install disteval).", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running SelfEngine: {e}", file=sys.stderr)
        sys.exit(1)


# --------------------------------------------------------------------------- #
# Reliability-analysis subcommands                                            #
# --------------------------------------------------------------------------- #
def _load_inputs(args):
    """Shared loading for the reliability subcommands."""
    from disteval.loaders import load_runs, load_tasks, load_trajectories

    runs = load_runs(args.runs, success_threshold=args.success_threshold)
    tasks = load_tasks(args.tasks) if getattr(args, "tasks", None) else None
    trajs = (
        load_trajectories(args.trajectories, success_threshold=args.success_threshold)
        if getattr(args, "trajectories", None) else None
    )
    return runs, tasks, trajs


def _thresholds(args):
    from disteval.reliability.classify import ReliabilityThresholds

    return ReliabilityThresholds(
        tau_cap=args.tau_cap, tau_rel=args.tau_rel, tau_stuck=args.tau_stuck,
        confidence=args.confidence, min_runs=args.min_runs,
    )


def _add_common(parser):
    parser.add_argument("runs", help="Path to run records (JSON/JSONL/Parquet)")
    parser.add_argument("--tasks", help="Path to task definitions")
    parser.add_argument("--trajectories", help="Path to trajectory event logs")
    parser.add_argument("--success-threshold", type=float, default=1.0,
                        help="Score at or above which a run counts as a success")
    parser.add_argument("--tau-cap", type=float, default=0.15,
                        help="Capability threshold: C_t = P(p_t > tau_cap)")
    parser.add_argument("--tau-rel", type=float, default=0.90,
                        help="Reliability threshold: R_t = P(p_t > tau_rel)")
    parser.add_argument("--tau-stuck", type=float, default=0.10,
                        help="Below this the agent effectively cannot do the task")
    parser.add_argument("--confidence", type=float, default=0.80,
                        help="Posterior mass required to assert a classification")
    parser.add_argument("--min-runs", type=int, default=3,
                        help="Fewer runs than this is always UNCERTAIN")
    return parser


def handle_evaluate(argv):
    """`disteval evaluate` -- the main entry point: runs in, full report out."""
    p = _add_common(argparse.ArgumentParser(
        prog="disteval evaluate",
        description="Estimate latent reliability from repeated runs and write a report",
    ))
    p.add_argument("--output-dir", "-o", default="disteval_report")
    p.add_argument("--title", default="Agent reliability report")
    p.add_argument("--figure-format", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--pooling", default="auto", choices=["auto", "on", "off"],
                   help="Hierarchical partial pooling; 'auto' decides by "
                        "leave-one-run-out predictive log-likelihood")
    p.add_argument("--no-html", action="store_true")
    args = p.parse_args(argv)

    from disteval.research_report import generate_report

    runs, tasks, trajs = _load_inputs(args)
    paths = generate_report(
        runs, args.output_dir, tasks=tasks, trajectories=trajs,
        thresholds=_thresholds(args), title=args.title,
        figure_format=args.figure_format, html_output=not args.no_html,
        hierarchical={"auto": "auto", "on": True, "off": False}[args.pooling],
    )
    print(f"Analysed {len(runs)} runs across "
          f"{len({r['task'] for r in runs})} tasks.")
    for k, v in paths.items():
        if not k.startswith("figure:"):
            print(f"  {k:22s} {v}")
    n_fig = sum(1 for k in paths if k.startswith("figure:"))
    print(f"  {'figures':22s} {n_fig} written to {args.output_dir}/figures")


def handle_diagnose(argv):
    """`disteval diagnose` -- per-task table to stdout, no files written."""
    p = _add_common(argparse.ArgumentParser(
        prog="disteval diagnose",
        description="Per-task posteriors, classification and recoverability ranking",
    ))
    p.add_argument("--top", type=int, default=20, help="Rows to show")
    p.add_argument("--label", help="Show only this classification")
    p.add_argument("--csv", help="Also write the full table to this CSV path")
    args = p.parse_args(argv)

    from disteval.reliability.classify import (
        diagnose, rank_by_recoverability, tie_diagnostics,
    )
    from disteval.reliability.posterior import posterior_from_scores

    runs, _tasks, _trajs = _load_inputs(args)
    by, dom = {}, {}
    for r in runs:
        by.setdefault(r["task"], []).append(float(r["score"]))
        if r.get("domain"):
            dom[r["task"]] = r["domain"]
    th = _thresholds(args)
    diags = [
        diagnose(posterior_from_scores(v, success_threshold=args.success_threshold),
                 task=t, model=runs[0]["model"], thresholds=th, scores=v,
                 domain=dom.get(t))
        for t, v in sorted(by.items())
    ]
    counts = {}
    for d in diags:
        counts[d.label] = counts.get(d.label, 0) + 1
    print(f"{len(diags)} tasks: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"thresholds: tau_cap={th.tau_cap} tau_rel={th.tau_rel} "
          f"tau_stuck={th.tau_stuck} confidence={th.confidence} min_runs={th.min_runs}")
    print()
    hdr = f"{'task':<26}{'n':>4}{'succ':>6}{'post':>8}{'95% CI':>16}{'C_t':>7}{'R_t':>7}  {'label':<12}{'recov':>7}"
    print(hdr)
    print("-" * len(hdr))
    shown = [d for d in rank_by_recoverability(diags, labels=None)
             if args.label is None or d.label == args.label][: args.top]
    for d in shown:
        print(f"{d.task[:26]:<26}{d.n_runs:>4}{d.n_success:>6.0f}"
              f"{d.posterior_mean:>8.3f}{f'[{d.ci_lo:.2f},{d.ci_hi:.2f}]':>16}"
              f"{d.capability:>7.3f}{d.reliability:>7.3f}  {d.label:<12}{d.recoverability:>7.3f}")
    t = tie_diagnostics(diags)
    if t.get("cutoff_is_arbitrary"):
        print(f"\nNote: only {t['n_distinct_scores']} distinct recoverability scores "
              f"across {t['n_tasks']} tasks (largest tie {t['largest_tier_size']}). "
              "Ranking within a tier is arbitrary.")
    if args.csv:
        import pandas as pd

        pd.DataFrame([d.to_dict() for d in diags]).to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


def handle_select(argv):
    """`disteval select` -- choose tasks and export preference pairs."""
    from disteval.selection.selectors import SELECTORS

    p = _add_common(argparse.ArgumentParser(
        prog="disteval select",
        description="Select training tasks and build success/failure preference pairs",
    ))
    p.add_argument("--strategy", default="recoverability",
                   choices=[k for k in SELECTORS if k != "oracle"])
    p.add_argument("--n-tasks", type=int, default=40)
    p.add_argument("--n-pairs", type=int, default=100)
    p.add_argument("--max-pairs-per-task", type=int, default=4)
    p.add_argument("--min-margin", type=float, default=0.5)
    p.add_argument("--output", "-o", default="preference_pairs.jsonl")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from disteval.reliability.classify import diagnose
    from disteval.reliability.posterior import posterior_from_scores
    from disteval.selection.pairs import PairConfig, build_dataset, to_dpo_jsonl
    from disteval.selection.selectors import make_selector

    runs, _tasks, trajs = _load_inputs(args)
    if trajs is None:
        print("error: --trajectories is required to build preference pairs",
              file=sys.stderr)
        sys.exit(2)
    by, dom = {}, {}
    for r in runs:
        by.setdefault(r["task"], []).append(float(r["score"]))
        if r.get("domain"):
            dom[r["task"]] = r["domain"]
    th = _thresholds(args)
    diags = [
        diagnose(posterior_from_scores(v, success_threshold=args.success_threshold),
                 task=t, model=runs[0]["model"], thresholds=th, scores=v,
                 domain=dom.get(t))
        for t, v in sorted(by.items())
    ]
    sel = make_selector(args.strategy, seed=args.seed).select(diags, args.n_tasks)
    grouped = {task: v for (_m, task), v in trajs.group().items()}
    ds = build_dataset(
        sel.tasks, grouped, strategy=args.strategy, target_pairs=args.n_pairs,
        recoverability={d.task: d.recoverability for d in diags},
        config=PairConfig(min_margin=args.min_margin,
                          max_pairs_per_task=args.max_pairs_per_task),
        seed=args.seed,
    )
    n = to_dpo_jsonl(ds, args.output)
    print(f"strategy={args.strategy}  tasks selected={sel.delivered}/{sel.requested} "
          f"(eligible {sel.n_eligible})")
    for k, v in ds.summary().items():
        if k != "note":
            print(f"  {k:22s} {v}")
    if sel.note:
        print(f"  note: {sel.note}")
    if ds.note:
        print(f"  note: {ds.note}")
    print(f"Wrote {n} pairs to {args.output}")


def handle_experiment(argv):
    """`disteval experiment` -- the full A-D loop from a config file."""
    p = argparse.ArgumentParser(
        prog="disteval experiment",
        description="Run the evaluate -> select -> train -> re-evaluate loop",
    )
    p.add_argument("config", help="Path to an experiment YAML config")
    p.add_argument("--runs", help="Run records; omit to use the built-in simulator")
    p.add_argument("--trajectories", help="Trajectory event logs")
    p.add_argument("--output-dir", "-o", default="runs")
    p.add_argument("--n-tasks", type=int, default=200,
                   help="Simulator task count when --runs is not given")
    p.add_argument("--coupling", default="weak",
                   choices=["strong", "weak", "none", "adversarial"],
                   help="Simulator coupling between recoverability and true benefit")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    import numpy as np

    from disteval.experiments.config import load_config
    from disteval.experiments.pipeline import run_experiment
    from disteval.experiments.tracking import ExperimentRun

    config = load_config(args.config)
    for w in config.validate():
        print(f"warning: {w}", file=sys.stderr)

    if args.runs:
        from disteval.loaders import load_runs, load_trajectories

        runs = load_runs(args.runs)
        by, dom = {}, {}
        for r in runs:
            by.setdefault(r["task"], []).append(float(r["score"]))
            if r.get("domain"):
                dom[r["task"]] = r["domain"]
        pool = {t: list(v) for t, v in by.items()}
        counters = {t: 0 for t in pool}

        def run_fn(task, i):
            v = pool[task]
            j = counters[task] % len(v)
            counters[task] += 1
            return v[j]

        tasks = list(pool)
        trajs = load_trajectories(args.trajectories) if args.trajectories else None
        traj_map = ({t: v for (_m, t), v in trajs.group().items()} if trajs else None)
        benefit = None
        if config.training.backend == "simulated":
            print("error: replaying recorded runs cannot supply ground-truth "
                  "training benefit, which the simulated backend requires. Set "
                  "training.backend to 'none' to export datasets, or omit --runs "
                  "to use the simulator.", file=sys.stderr)
            sys.exit(2)
    else:
        from disteval.sim.world import SimulatedWorld, WorldConfig

        world = SimulatedWorld(WorldConfig(
            n_tasks=args.n_tasks, benefit_coupling=args.coupling,
            seed=config.experiment.seed,
        ))
        rng = np.random.default_rng(config.experiment.seed)
        tasks = world.tasks()
        dom = {k: v.domain for k, v in world.truth.items()}
        benefit = world.true_benefit()
        traj_map = None

        def run_fn(task, i):
            return float(world.run(task, rng)["score"])

        print(f"Using the built-in simulator: {args.n_tasks} tasks, "
              f"benefit coupling '{args.coupling}'.")

    with ExperimentRun.create(config, args.output_dir, overwrite=args.overwrite) as run:
        result = run_experiment(
            tasks, run_fn, config, trajectories=traj_map, domains=dom,
            true_benefit=benefit, output_dir=str(run.path("data")),
        )
        summary = result.summary()
        run.write_table("task_metrics", result.phase_a.to_frame())
        run.write_table("strategy_outcomes", result.to_frame())
        run.write_table("summary", summary)
        run.log(**{k: v for k, v in result.phase_a.aggregate.items()
                   if isinstance(v, (int, float))})
        for w in result.warnings:
            print(f"warning: {w}", file=sys.stderr)
        print(f"\nPhase A: {result.phase_a.n_executions} executions, "
              f"{len(result.phase_a.diagnoses)} tasks")
        print(f"Split: {result.split.strategy}, "
              f"{len(result.split.train)} train / {len(result.split.test)} held out")
        print()
        print(summary.to_string(index=False))
        print(f"\nWrote results to {run.root}")


def handle_sweep(argv):
    """`disteval sweep` -- run a matrix of experiments."""
    p = argparse.ArgumentParser(
        prog="disteval sweep", description="Run a matrix of experiments",
    )
    p.add_argument("config", help="Path to a sweep YAML config")
    p.add_argument("--output-dir", "-o", default="runs")
    p.add_argument("--n-tasks", type=int, default=150)
    p.add_argument("--coupling", default="weak",
                   choices=["strong", "weak", "none", "adversarial"])
    p.add_argument("--csv", default="sweep_results.csv")
    args = p.parse_args(argv)

    import numpy as np

    from disteval.experiments.pipeline import run_experiment
    from disteval.experiments.sweep import aggregate_sweep, load_sweep, run_sweep
    from disteval.sim.world import SimulatedWorld, WorldConfig

    spec = load_sweep(args.config)
    print(f"Sweep {spec.name!r}: {spec.n_cells} cells over axes {spec.axes()}")

    def run_one(config):
        world = SimulatedWorld(WorldConfig(
            n_tasks=args.n_tasks, benefit_coupling=args.coupling,
            seed=config.experiment.seed,
        ))
        rng = np.random.default_rng(config.experiment.seed)
        return run_experiment(
            world.tasks(), lambda t, i: float(world.run(t, rng)["score"]), config,
            domains={k: v.domain for k, v in world.truth.items()},
            true_benefit=world.true_benefit(),
        )

    index = run_sweep(spec, run_one, base_dir=args.output_dir)
    n_ok = int((index["status"] == "ok").sum())
    print(f"{n_ok}/{len(index)} cells succeeded")
    if n_ok < len(index):
        for _, r in index[index["status"] != "ok"].iterrows():
            print(f"  FAILED {r['experiment_id']}: {r['error']}", file=sys.stderr)
    table = aggregate_sweep(index)
    table.to_csv(args.csv, index=False)
    print(f"\nWrote {args.csv} ({len(table)} rows)")
    cols = [c for c in ("strategy", "heldout_delta", "benefit_captured") if c in table]
    if cols and n_ok:
        print(table[cols + [c for c in index.columns if c in table and c not in cols]]
              .head(30).to_string(index=False))


def handle_simulate(argv):
    """`disteval simulate` -- validate the estimators against known ground truth."""
    p = argparse.ArgumentParser(
        prog="disteval simulate",
        description="Run the ground-truth validation suite for the estimators",
    )
    p.add_argument("--study", default="all",
                   choices=["all", "max_bias", "posterior_recovery", "shrinkage",
                            "classification", "decomposition", "selection", "adaptive"])
    p.add_argument("--quick", action="store_true", help="Smaller, faster version")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", "-o", help="Write each study to CSV here")
    args = p.parse_args(argv)

    from disteval.sim import studies

    if args.study == "all":
        results = studies.run_all_studies(seed=args.seed, quick=args.quick)
    else:
        fn = getattr(studies, f"{args.study}_study")
        results = {args.study: fn(seed=args.seed)}

    for name, df in results.items():
        print(f"\n=== {name.replace('_', ' ').upper()} ===")
        print(df.round(4).to_string(index=False))
        if args.output_dir:
            import os

            os.makedirs(args.output_dir, exist_ok=True)
            path = os.path.join(args.output_dir, f"{name}.csv")
            df.to_csv(path, index=False)
    if args.output_dir:
        print(f"\nWrote {len(results)} tables to {args.output_dir}")


def handle_metrics(argv):
    """`disteval metrics` -- print the metric registry."""
    p = argparse.ArgumentParser(
        prog="disteval metrics",
        description="Print metric definitions, assumptions and edge cases",
    )
    p.add_argument("name", nargs="?", help="A single metric; omit for all")
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args(argv)

    from disteval import metrics_spec

    names = [args.name] if args.name else list(metrics_spec.REGISTRY)
    unknown = [n for n in names if n not in metrics_spec.REGISTRY]
    if unknown:
        print(f"unknown metric(s) {unknown}; have "
              f"{sorted(metrics_spec.REGISTRY)}", file=sys.stderr)
        sys.exit(2)
    if args.markdown:
        print(metrics_spec.to_markdown(names))
        return
    for n in names:
        s = metrics_spec.REGISTRY[n]
        print(f"\n{s.name}\n{'=' * len(s.name)}")
        print(f"  {s.summary}")
        print(f"  definition : {s.definition}")
        print(f"  domain     : {s.domain} ({'higher' if s.higher_is_better else 'lower'} is better)")
        print(f"  estimator  : {s.estimator}")
        print(f"  uncertainty: {s.uncertainty}")
        for a in s.assumptions:
            print(f"  assumes    : {a}")
        for e in s.edge_cases:
            print(f"  edge case  : {e}")
        for w in s.prior_work:
            print(f"  prior work : {w}")
        if s.implementation:
            print(f"  code       : {s.implementation}")


_NEW_HANDLERS = {
    "evaluate": handle_evaluate,
    "diagnose": handle_diagnose,
    "select": handle_select,
    "experiment": handle_experiment,
    "sweep": handle_sweep,
    "simulate": handle_simulate,
    "metrics": handle_metrics,
}


if __name__ == "__main__":
    main()