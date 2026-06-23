"""Failure-mode distribution per stratum -- the second missing primitive.

The failure-mode *mix* is as informative as the score distribution: two agents
with the same success rate can fail in entirely different ways.
"""
from __future__ import annotations

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
