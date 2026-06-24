"""Failure-mode distribution per stratum -- the second missing primitive.

The failure-mode *mix* is as informative as the score distribution: two agents
with the same success rate can fail in entirely different ways.

Criterion-level grading
-----------------------
Real evaluation rubrics score agents against multiple pass/fail criteria (e.g.
"Did the output minimise cost?", "Is the file correctly formatted?"). Each
episode may pass some criteria and fail others. ``criterion_failure_rates``
aggregates those per-criterion outcomes, exposing *which specific rubric items*
an agent consistently fails — the natural next step after distributional
failure analysis.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def failure_distribution(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Proportion of each failure_mode, optionally broken down by stratum columns.

    Only non-success episodes contribute. Returns a tidy frame:
    [<by...>, failure_mode, n, share_of_failures].
    """
    if "success" not in df.columns:
        raise ValueError("DataFrame must have a 'success' column")
    if by and any(c not in df.columns for c in by):
        missing = [c for c in by if c not in df.columns]
        raise ValueError(f"Stratum columns not found in DataFrame: {missing}")
    fails = df[~df["success"].astype(bool)].copy()
    if fails.empty:
        return pd.DataFrame(columns=(by or []) + ["failure_mode", "n", "share_of_failures"])
    fails["failure_mode"] = fails["failure_mode"].fillna("unlabeled")
    group_cols = (by or []) + ["failure_mode"]
    counts = fails.groupby(group_cols).size().rename("n").reset_index()
    denom_cols = by or []
    if denom_cols:
        totals = counts.groupby(denom_cols)["n"].transform("sum")
    else:
        totals = counts["n"].sum()
    counts["share_of_failures"] = counts["n"] / totals
    return counts.sort_values((by or []) + ["n"], ascending=False).reset_index(drop=True)


def criterion_failure_rates(
    episodes: list[dict[str, Any]],
    by: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate per-criterion pass/fail outcomes across episodes.

    This mirrors the criterion-level grading used in rubric-based evaluation
    frameworks: each episode carries a ``criteria`` dict mapping criterion name
    → bool (True = passed).  Criteria that are absent for a given episode are
    treated as not applicable and excluded from that criterion's denominator.

    Args:
        episodes: List of episode dicts.  Each must contain:
            - ``criteria`` (dict[str, bool]): per-criterion pass/fail verdicts.
            - Any additional keys listed in ``by`` for stratification.
        by: Optional list of episode-level keys to stratify by (e.g.
            ``["difficulty"]``).

    Returns:
        A tidy DataFrame with columns:
        [<by...>, criterion, n_episodes, n_failed, failure_rate].
        Sorted by ``failure_rate`` descending so the most-failed criterion
        appears first — a direct pointer to the weakest rubric dimension.

    Example::

        episodes = [
            {"criteria": {"cost_ok": True,  "format_ok": False}, "difficulty": "hard"},
            {"criteria": {"cost_ok": False, "format_ok": False}, "difficulty": "hard"},
            {"criteria": {"cost_ok": True,  "format_ok": True},  "difficulty": "easy"},
        ]
        df = criterion_failure_rates(episodes, by=["difficulty"])
    """
    if not episodes:
        cols = (by or []) + ["criterion", "n_episodes", "n_failed", "failure_rate"]
        return pd.DataFrame(columns=cols)

    records = []
    for ep in episodes:
        criteria: dict[str, Any] = ep.get("criteria") or {}
        strata = {k: ep.get(k) for k in (by or [])}
        for criterion, passed in criteria.items():
            records.append({**strata, "criterion": criterion, "passed": bool(passed)})

    if not records:
        cols = (by or []) + ["criterion", "n_episodes", "n_failed", "failure_rate"]
        return pd.DataFrame(columns=cols)

    raw = pd.DataFrame(records)
    group_cols = (by or []) + ["criterion"]
    agg = raw.groupby(group_cols)["passed"].agg(
        n_episodes="count",
        n_passed="sum",
    ).reset_index()
    agg["n_failed"] = agg["n_episodes"] - agg["n_passed"]
    agg["failure_rate"] = agg["n_failed"] / agg["n_episodes"]
    agg = agg.drop(columns=["n_passed"])
    return agg.sort_values("failure_rate", ascending=False).reset_index(drop=True)


def top_failing_criteria(
    episodes: list[dict[str, Any]],
    n: int = 5,
    by: list[str] | None = None,
) -> pd.DataFrame:
    """Return the ``n`` criteria with the highest failure rate.

    Convenience wrapper around ``criterion_failure_rates`` for quick
    identification of the most actionable rubric weaknesses.
    """
    df = criterion_failure_rates(episodes, by=by)
    return df.head(n).reset_index(drop=True)
