"""Seam: RecordStore  ->  rliable (google-research/rliable).

rliable consumes a dict {algo_name: ndarray[n_runs, n_tasks]} of *aggregate*
per-(run,task) scores and computes IQM / optimality-gap / probability-of-
improvement with stratified-bootstrap CIs, plus performance profiles.

We provide the matrix builder. If rliable is installed, the commented block
shows the exact call. Our bootstrap.py reimplements the same math dependency-free
for the prototype, but for production you'd defer to rliable here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..records import RecordStore


def score_matrix(store: RecordStore, agg: str = "mean") -> tuple[np.ndarray, list[str]]:
    """Build rliable's [n_runs, n_tasks] matrix for a single model.

    Each cell = aggregate score for one (run_id, task). run_id is the repetition
    axis -- which is *exactly* the repeated-eval axis rliable was designed around.
    """
    df = store.df()
    pivot = df.pivot_table(index="run_id", columns="task", values="score", aggfunc=agg)
    return pivot.to_numpy(dtype=float), list(pivot.columns)


def to_rliable_dict(stores_by_model: dict[str, RecordStore], agg: str = "mean") -> dict[str, np.ndarray]:
    return {name: score_matrix(s, agg)[0] for name, s in stores_by_model.items()}


# --- production path (requires `pip install rliable`) -------------------------
# from rliable import library as rly, metrics, plot_utils
# score_dict = to_rliable_dict(stores_by_model)
# aggregate_fn = lambda x: np.array([metrics.aggregate_iqm(x),
#                                    metrics.aggregate_optimality_gap(x)])
# point, interval = rly.get_interval_estimates(score_dict, aggregate_fn, reps=50000)
# -----------------------------------------------------------------------------
