"""Held-out splits, including the ones that actually test transfer.

Holding out individual *tasks* is the weakest useful split: the held-out tasks
share domains, environments and tool vocabularies with training, so an
improvement can come from domain-specific adaptation rather than a transferable
change in policy. That is worth measuring, but it should not be the only thing
measured, and it should not be described as out-of-distribution.

Strategies, roughly in increasing order of how much they demand:

``task``               random held-out tasks. In-distribution.
``environment``        held-out environment/world instances within known domains.
``tool_composition``   held-out combinations of tools. Tests whether the agent
                       generalises across how capabilities are composed.
``rubric_type``        held-out grading styles.
``domain_holdout``     entire professional domains held out. The strongest test
                       available without new data collection: train on
                       recoverable finance tasks, ask whether reliability
                       improves on unseen legal tasks.

Every split returns a :class:`Split` recording which groups went where, so a
result can state exactly what was held out rather than "a held-out set".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

__all__ = ["Split", "make_split", "SPLIT_STRATEGIES"]


@dataclass
class Split:
    """A train/test partition of tasks, with the grouping that produced it."""

    strategy: str
    train: list[str]
    test: list[str]
    group_of: dict[str, str]
    train_groups: list[str]
    test_groups: list[str]
    note: str = ""

    @property
    def is_group_disjoint(self) -> bool:
        """Whether no group appears on both sides. False means leakage."""
        return not (set(self.train_groups) & set(self.test_groups))

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "n_train": len(self.train),
            "n_test": len(self.test),
            "train_groups": list(self.train_groups),
            "test_groups": list(self.test_groups),
            "group_disjoint": self.is_group_disjoint,
            "note": self.note,
        }


def _group_split(
    tasks: Sequence[str],
    group_of: Mapping[str, str],
    strategy: str,
    test_fraction: float,
    holdout_groups: Sequence[str],
    seed: int,
) -> Split:
    groups = sorted({group_of.get(t, "_none") for t in tasks})
    rng = np.random.default_rng(seed)
    if holdout_groups:
        missing = set(holdout_groups) - set(groups)
        if missing:
            raise ValueError(f"holdout groups not present in the data: {sorted(missing)}")
        test_groups = list(holdout_groups)
    else:
        n_test = max(1, int(round(len(groups) * test_fraction)))
        if n_test >= len(groups):
            raise ValueError(
                f"test_fraction={test_fraction} would hold out every group "
                f"({len(groups)} available); nothing would be left to train on"
            )
        test_groups = sorted(rng.choice(groups, size=n_test, replace=False).tolist())
    train_groups = [g for g in groups if g not in set(test_groups)]
    train = [t for t in tasks if group_of.get(t, "_none") in set(train_groups)]
    test = [t for t in tasks if group_of.get(t, "_none") in set(test_groups)]
    note = ""
    if len(groups) < 4:
        note = (
            f"only {len(groups)} groups available for a {strategy!r} split; with so "
            "few groups the split is dominated by which one happened to be held out, "
            "and the result should be read as a single observation, not an estimate"
        )
    return Split(strategy, train, test, dict(group_of), train_groups, test_groups, note)


def make_split(
    tasks: Sequence[str],
    *,
    strategy: str = "domain_holdout",
    domains: Optional[Mapping[str, str]] = None,
    environments: Optional[Mapping[str, str]] = None,
    tool_signatures: Optional[Mapping[str, str]] = None,
    rubric_types: Optional[Mapping[str, str]] = None,
    test_fraction: float = 0.3,
    holdout_groups: Sequence[str] = (),
    seed: int = 0,
) -> Split:
    """Build a train/test split of ``tasks`` under the named strategy."""
    tasks = list(tasks)
    if not tasks:
        raise ValueError("no tasks to split")

    if strategy == "task":
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(tasks))
        n_test = max(1, int(round(len(tasks) * test_fraction)))
        test = sorted(tasks[i] for i in idx[:n_test])
        train = sorted(tasks[i] for i in idx[n_test:])
        return Split(
            "task", train, test, {t: t for t in tasks}, train, test,
            note="in-distribution split: held-out tasks share domains, environments "
                 "and tools with training, so improvement here does not demonstrate "
                 "transfer",
        )

    sources = {
        "domain_holdout": domains,
        "environment": environments,
        "tool_composition": tool_signatures,
        "rubric_type": rubric_types,
    }
    if strategy not in sources:
        raise ValueError(
            f"unknown split strategy {strategy!r}; have {sorted(SPLIT_STRATEGIES)}"
        )
    group_of = sources[strategy]
    if not group_of:
        raise ValueError(
            f"strategy {strategy!r} needs a grouping map, but none was supplied"
        )
    return _group_split(
        tasks, group_of, strategy, test_fraction, holdout_groups, seed
    )


SPLIT_STRATEGIES = ("task", "environment", "tool_composition", "rubric_type", "domain_holdout")
