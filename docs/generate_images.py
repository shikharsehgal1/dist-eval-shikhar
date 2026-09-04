"""Regenerate every chart embedded in README.md.

Two sources, because they answer different questions:

  * The five `real_*.png` charts come from experiment 11 — 64 live episodes from
    four real agents. That is the worked example in the README.
  * `repeat_eval_reliability.png` comes from jobs/run_A|B|C, the same eval run
    three separate times. Experiment 11 cannot produce this chart: measuring
    run-to-run spread requires repeating the whole eval, which the live run did
    not do.

    python3 docs/generate_images.py

Deterministic, and nothing here calls a model.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "experiments" / "experiment_11_real_model_distributed_eval"))

import plots as real_model_plots  # noqa: E402  (path set above)

from disteval import viz  # noqa: E402
from disteval.adapters.harbor_jobs import load_harbor_job  # noqa: E402
from disteval.repeat import bootstrap_vs_repeat  # noqa: E402

OUTPUT_DIR = ROOT / "docs" / "images"
REPEATED_RUNS = ("A", "B", "C")


def real_model_charts() -> dict[str, Path]:
    """The worked example: five views of 64 real episodes."""
    return real_model_plots.generate_all(OUTPUT_DIR)


def repeated_eval_chart() -> Path:
    """Published single-run error bar vs the true spread across three whole evals."""
    stores = [
        load_harbor_job(
            str(ROOT / f"jobs/run_{r}/disteval-run-{r}"),
            run_id=f"run_{r}",
            tasks_dir=str(ROOT / "tasks"),
        )
        for r in REPEATED_RUNS
    ]
    spread = bootstrap_vs_repeat(stores, lambda d: d["score"].mean(), strata_cols=["task"])
    out = OUTPUT_DIR / "repeat_eval_reliability.png"
    viz.plot_eval_reliability(
        spread["mean_single_run_bootstrap_ci_width"],
        spread["meta_ci_width"],
        str(out),
        agent_name="disteval-run-A",
    )
    print(f"Single-run bootstrap CI width: {spread['mean_single_run_bootstrap_ci_width']:.4f}")
    print(f"True run-to-run spread:        {spread['meta_ci_width']:.4f}")
    print(f"Underconfidence ratio:         {spread['underconfidence_ratio']:.2f}x")
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = dict(real_model_charts())
    paths["eval_reliability"] = repeated_eval_chart()
    print()
    for name, path in paths.items():
        print(f"  {name:20} -> {Path(path).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
