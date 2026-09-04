"""Running a matrix of experiments from one config, and aggregating the results.

A single experiment answers "does this work with these settings". An ablation
sweep answers the question that actually matters: "does the conclusion survive
changing the settings". This module runs the cross-product of a sweep
specification and produces one aggregate table.

Sweep file format (YAML), where ``base`` is an ordinary experiment config and
``sweep`` maps dotted config paths to lists of values::

    base:
      experiment: {name: recoverability_ablation, n_seeds: 5}
      evaluation: {runs_per_task: 8}

    sweep:
      reliability.prior: [jeffreys, uniform]
      reliability.model: [independent, hierarchical]
      selection.method: [random, hardest, highest_variance, recoverability]
      evaluation.runs_per_task: [2, 4, 8]

Each cell gets its own config -- and therefore its own hash, directory and
metadata -- so a sweep is just many ordinary runs plus an index. Failed cells are
recorded with their error and do not abort the sweep; a sweep that silently drops
failures produces a biased aggregate table.
"""
from __future__ import annotations

import itertools
import traceback
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import ExperimentConfig

__all__ = ["SweepSpec", "load_sweep", "expand_sweep", "run_sweep", "aggregate_sweep"]


@dataclass
class SweepSpec:
    """A base config plus the axes to vary."""

    base: dict
    sweep: dict[str, list]
    name: str = "sweep"

    @property
    def n_cells(self) -> int:
        n = 1
        for v in self.sweep.values():
            n *= max(len(v), 1)
        return n

    def axes(self) -> list[str]:
        return list(self.sweep)


def load_sweep(path: str | Path) -> SweepSpec:
    """Load a sweep YAML. The ``base`` section is validated as a full config."""
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    unknown = set(data) - {"base", "sweep", "name"}
    if unknown:
        raise ValueError(f"unknown sweep key(s) {sorted(unknown)}; expected base/sweep/name")
    base = data.get("base", {})
    ExperimentConfig.from_dict(base)  # validate eagerly, before running anything
    sweep = data.get("sweep", {}) or {}
    for k, v in sweep.items():
        if not isinstance(v, list) or not v:
            raise ValueError(f"sweep axis {k!r} must be a non-empty list, got {v!r}")
        if "." not in k:
            raise ValueError(
                f"sweep axis {k!r} must be a dotted config path such as "
                "'evaluation.runs_per_task'"
            )
    return SweepSpec(base=base, sweep=sweep, name=data.get("name", Path(path).stem))


def _set_path(d: dict, dotted: str, value: Any) -> None:
    section, _, key = dotted.partition(".")
    d.setdefault(section, {})[key] = value


def expand_sweep(spec: SweepSpec) -> list[tuple[dict, ExperimentConfig]]:
    """Cross-product of the sweep axes. Returns ``(cell, config)`` pairs.

    ``cell`` is the axis assignment, which becomes the identifying columns of the
    aggregate table; ``config`` is the fully-materialised experiment config,
    whose hash is the run directory name.
    """
    axes = list(spec.sweep)
    out = []
    for combo in itertools.product(*(spec.sweep[a] for a in axes)):
        data = deepcopy(spec.base)
        cell = dict(zip(axes, combo))
        for k, v in cell.items():
            _set_path(data, k, v)
        # Make the name reflect the cell so run directories are legible.
        base_name = data.get("experiment", {}).get("name", spec.name)
        suffix = "_".join(f"{k.split('.')[-1]}={v}" for k, v in cell.items())
        data.setdefault("experiment", {})["name"] = f"{base_name}__{suffix}"
        out.append((cell, ExperimentConfig.from_dict(data)))
    return out


def run_sweep(
    spec: SweepSpec,
    run_one: Callable[[ExperimentConfig], Any],
    *,
    base_dir: str | Path = "runs",
    on_error: str = "record",
) -> "object":
    """Run every cell, returning an index DataFrame.

    ``run_one(config)`` must execute one experiment and return an object with a
    ``summary()`` DataFrame (i.e. an :class:`ExperimentResult`). Failures are
    recorded with their traceback and the sweep continues -- dropping failed
    cells silently would bias the aggregate.
    """
    import pandas as pd

    if on_error not in ("record", "raise"):
        raise ValueError("on_error must be 'record' or 'raise'")
    rows = []
    for cell, config in expand_sweep(spec):
        row = {**cell, "experiment_id": config.experiment_id}
        try:
            result = run_one(config)
            summary = result.summary()
            row["status"] = "ok"
            row["_summary"] = summary
            row["n_warnings"] = len(getattr(result, "warnings", []))
        except Exception as exc:  # noqa: BLE001 -- a failed cell must not abort the sweep
            if on_error == "raise":
                raise
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc(limit=5)
            row["_summary"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_sweep(index) -> "object":
    """Flatten a sweep index into one long table: cell axes x strategy x metrics."""
    import pandas as pd

    frames = []
    axis_cols = [
        c for c in index.columns
        if c not in ("status", "error", "traceback", "_summary", "experiment_id", "n_warnings")
    ]
    for _, row in index.iterrows():
        if row["status"] != "ok" or row["_summary"] is None:
            frames.append(
                pd.DataFrame([{**{c: row[c] for c in axis_cols},
                               "experiment_id": row["experiment_id"],
                               "status": row["status"],
                               "error": row.get("error")}])
            )
            continue
        s = row["_summary"].copy()
        for c in axis_cols:
            s[c] = row[c]
        s["experiment_id"] = row["experiment_id"]
        s["status"] = "ok"
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
