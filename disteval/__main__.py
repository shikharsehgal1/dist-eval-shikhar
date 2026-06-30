#!/usr/bin/env python3
"""
Unified CLI dispatcher for disteval package.

This module provides a command-line interface that allows running different
disteval subcommands via `python -m disteval <subcommand>` or the `disteval`
script entry point.

Available subcommands:
- report: Generate evaluation reports
- compare: Generate comparison reports  
- sim: Run training simulations
- engine: Run SelfEngine on Harbor job directories
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
    if subcommand == "report":
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
    help_text = """usage: disteval [-h] {report,compare,sim,engine,train} ...

Distribution-first evaluation and self-improvement for long-horizon AI agents

positional arguments:
  {report,compare,sim,engine,train}
                        Subcommand to run

optional arguments:
  -h, --help            show this help message and exit

Available subcommands:
  report    Generate evaluation reports
  compare   Generate comparison reports
  sim       Run training simulations
  engine    Run SelfEngine on Harbor job directories
  train     Train a policy from a SelfImprovementPlan using a DPO trainer

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


if __name__ == "__main__":
    main()