"""Structured, self-describing experiment output directories.

Every run writes a directory that contains everything needed to understand and
reproduce it, with no reliance on an external service::

    runs/<experiment_id>/
      config.yaml            the exact config, including defaults that were not
                             written by hand
      metadata.json          environment, versions, git commit, timings, seeds
      task_metrics.parquet   one row per (model, task): posteriors, labels, scores
      run_metrics.parquet    one row per execution
      trajectory_metrics.parquet
      preference_pairs.jsonl
      figures/
      report.md / report.html
      logs/

Parquet rather than CSV for the metric tables: they are columnar (so a sweep can
read one column across hundreds of runs without parsing everything), typed (so a
boolean stays a boolean), and compact at benchmark scale. JSONL for preference
pairs because they are nested and are consumed by external trainers.

The experiment id is a hash of the config, so re-running the same config lands in
the same directory. That is deliberate -- it makes accidental duplicate runs
visible -- and :class:`ExperimentRun` refuses to overwrite an existing completed
run unless told to.

W&B / MLflow are supported through :meth:`ExperimentRun.attach_logger` but are
never required and are not imported unless used.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

__all__ = ["ExperimentRun", "git_commit", "environment_metadata"]


def git_commit(cwd: Optional[str] = None) -> Optional[str]:
    """Current git commit, or None outside a repository. Never raises."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def git_dirty(cwd: Optional[str] = None) -> Optional[bool]:
    """Whether the working tree has uncommitted changes. None if unknown.

    Recorded because a result produced from a dirty tree is not reproducible from
    its commit hash, and that should be visible in the metadata rather than
    discovered later.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=5, check=False,
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def environment_metadata() -> dict:
    """Everything about the machine and libraries that could change a result."""
    versions = {}
    for mod in ("numpy", "scipy", "pandas", "sklearn", "pyarrow"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": versions,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }


@dataclass
class ExperimentRun:
    """A single experiment's output directory, with helpers for writing into it."""

    root: Path
    experiment_id: str
    config: Any = None
    _started: float = field(default_factory=time.time)
    _artifacts: dict[str, str] = field(default_factory=dict)
    _metrics: dict[str, Any] = field(default_factory=dict)
    _logger: Optional[Callable[[dict], None]] = None

    @classmethod
    def create(
        cls,
        config,
        base_dir: str | Path = "runs",
        *,
        overwrite: bool = False,
    ) -> "ExperimentRun":
        """Create (or reopen) the directory for ``config``.

        Refuses to clobber a run that already completed, because the id is a hash
        of the config and two completed runs with the same id should be identical.
        Pass ``overwrite=True`` when you genuinely mean to redo it.
        """
        from .config import save_config

        root = Path(base_dir) / config.experiment_id
        marker = root / "COMPLETED"
        if marker.exists() and not overwrite:
            raise FileExistsError(
                f"{root} already contains a completed run with this exact config. "
                "Pass overwrite=True to redo it, or change the config (its hash is "
                "the directory name, so any change gives a new directory)."
            )
        for sub in ("", "figures", "logs", "data"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        save_config(config, root / "config.yaml")
        run = cls(root=root, experiment_id=config.experiment_id, config=config)
        run._metrics["config_warnings"] = list(config.validate())
        return run

    # -- writing ------------------------------------------------------------
    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def write_table(self, name: str, df, *, parquet: bool = True) -> Path:
        """Write a DataFrame as Parquet (with a CSV sibling for eyeballing)."""
        p = self.path(f"{name}.parquet" if parquet else f"{name}.csv")
        if parquet:
            try:
                df.to_parquet(p, index=False)
            except Exception:
                # No parquet engine: degrade to CSV rather than losing the result.
                p = self.path(f"{name}.csv")
                df.to_csv(p, index=False)
        else:
            df.to_csv(p, index=False)
        if p.suffix == ".parquet":
            df.to_csv(self.path(f"{name}.csv"), index=False)
        self._artifacts[name] = str(p)
        return p

    def write_jsonl(self, name: str, records: Sequence[Mapping]) -> Path:
        p = self.path(f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")
        self._artifacts[name] = str(p)
        return p

    def write_text(self, name: str, text: str) -> Path:
        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        self._artifacts[name] = str(p)
        return p

    def figure_path(self, name: str, fmt: str = "pdf") -> Path:
        return self.path("figures", f"{name}.{fmt}")

    def log(self, **metrics) -> None:
        """Record scalar metrics; forwarded to an attached logger if any."""
        self._metrics.update(metrics)
        if self._logger is not None:
            self._logger(dict(metrics))

    def attach_logger(self, logger: Callable[[dict], None]) -> None:
        """Attach a W&B/MLflow-style callback. Optional; nothing is imported here."""
        self._logger = logger

    # -- lifecycle ----------------------------------------------------------
    def finish(self, status: str = "ok", error: str = "") -> Path:
        """Write metadata.json and mark the run completed."""
        meta = {
            "experiment_id": self.experiment_id,
            "status": status,
            "error": error,
            "started_at": self._started,
            "finished_at": time.time(),
            "duration_s": time.time() - self._started,
            "environment": environment_metadata(),
            "artifacts": dict(self._artifacts),
            "metrics": self._metrics,
            "config": self.config.to_dict() if self.config is not None else None,
        }
        p = self.path("metadata.json")
        p.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        if status == "ok":
            self.path("COMPLETED").write_text(self.experiment_id, encoding="utf-8")
        return p

    def __enter__(self) -> "ExperimentRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finish(
            status="ok" if exc_type is None else "error",
            error="" if exc is None else f"{exc_type.__name__}: {exc}",
        )
        return False
