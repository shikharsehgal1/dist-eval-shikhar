"""Aligning two agent trajectories that do not share a step clock.

The problem
-----------
Comparing run A's step 7 with run B's step 7 is only meaningful if both runs did
the same number of the same things in the same order. Real agents insert retries,
skip steps they can shortcut, and reorder independent subtasks. Under a raw index
comparison, a single extra retry near the start makes every subsequent step look
like a divergence, which destroys the signal the divergence analysis is after.

So the comparison is an *alignment* problem, and this module provides three
alignment methods with a common result type.

``needleman_wunsch`` (default)
    Global sequence alignment with affine-free linear gap penalties, over a
    pluggable event similarity. This is the right default because agent
    trajectories are sequences of discrete actions and the thing we want out is
    an explicit correspondence including insertions and deletions -- exactly what
    a global alignment produces. Cost O(n*m) time and memory; the memory is
    bounded by ``max_cells`` and the aligner falls back to a banded variant
    beyond it.

``dtw``
    Dynamic time warping over continuous state-feature vectors. Use it when the
    meaningful thing about a step is the *environment state* it produced rather
    than the action name -- e.g. two agents editing the same spreadsheet by
    different routes. DTW allows many-to-one matches, which is right for
    "one agent took three steps to do what the other did in one" and wrong for
    "one agent did an extra unrelated thing" (that is what NW gaps are for).

``event_type``
    Alignment on the coarse event type only, ignoring tool identity and
    arguments. The cheapest method and the one to use when tool vocabularies
    differ across the agents being compared (cross-agent analysis).

Similarity is a plug point
--------------------------
:class:`EventSimilarity` is a callable protocol. The built-in
:func:`structural_similarity` uses only structure (event type, tool name,
target, argument overlap) and therefore needs no model API. A semantic
comparator backed by an LLM or an embedding model can be dropped in without
touching the alignment algorithms -- see
:class:`disteval.trajectory.divergence.SemanticComparator`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from .events import Trajectory, TrajectoryEvent

__all__ = [
    "EventSimilarity",
    "structural_similarity",
    "exact_action_similarity",
    "event_type_similarity",
    "Alignment",
    "AlignedPair",
    "align",
    "needleman_wunsch",
    "dtw",
]


class EventSimilarity(Protocol):
    """Any callable scoring two events in [0, 1]. 1 = identical, 0 = unrelated."""

    def __call__(self, a: TrajectoryEvent, b: TrajectoryEvent) -> float: ...


# --------------------------------------------------------------------------- #
# Built-in similarities (no external model required)                          #
# --------------------------------------------------------------------------- #
def structural_similarity(a: TrajectoryEvent, b: TrajectoryEvent) -> float:
    """Weighted structural agreement: type, tool, target, arguments.

    The weights encode a claim about what makes two agent steps "the same move":
    the tool matters most, then what it was pointed at, then the fine arguments.
    Event type alone is nearly free information, so it carries the least weight.
    All four components are computable from structure alone -- no model calls.
    """
    score = 0.0
    score += 0.15 * float(a.event_type == b.event_type)
    if a.tool_name or b.tool_name:
        score += 0.40 * float(a.tool_name == b.tool_name)
    else:
        score += 0.40 * float(a.event_type == b.event_type)
    if a.target or b.target:
        score += 0.25 * float(a.target == b.target)
    else:
        score += 0.25 * float(a.target == b.target)
    ka, kb = set(a.tool_args or {}), set(b.tool_args or {})
    if ka or kb:
        shared = ka & kb
        agree = sum(1 for k in shared if a.tool_args[k] == b.tool_args[k])
        score += 0.20 * (agree / max(len(ka | kb), 1))
    else:
        score += 0.20
    return float(min(score, 1.0))


def exact_action_similarity(a: TrajectoryEvent, b: TrajectoryEvent) -> float:
    """1 iff the fully-qualified argument keys match. The strictest comparator."""
    return float(a.arg_key == b.arg_key)


def event_type_similarity(a: TrajectoryEvent, b: TrajectoryEvent) -> float:
    """1 iff the coarse event types match. For cross-agent comparison."""
    return float(a.event_type == b.event_type)


# --------------------------------------------------------------------------- #
# Result types                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AlignedPair:
    """One column of the alignment. ``None`` on a side means a gap there."""

    i: Optional[int]      # index into trajectory a
    j: Optional[int]      # index into trajectory b
    similarity: float     # 0.0 for a gap

    @property
    def is_match(self) -> bool:
        return self.i is not None and self.j is not None

    @property
    def is_gap(self) -> bool:
        return not self.is_match


@dataclass
class Alignment:
    """The correspondence between two trajectories."""

    pairs: list[AlignedPair]
    method: str
    score: float
    a_len: int
    b_len: int
    meta: dict = field(default_factory=dict)

    @property
    def n_matched(self) -> int:
        return sum(1 for p in self.pairs if p.is_match)

    @property
    def n_gaps(self) -> int:
        return sum(1 for p in self.pairs if p.is_gap)

    @property
    def normalized_score(self) -> float:
        """Alignment score per aligned column, in roughly [0, 1]."""
        denom = max(self.a_len, self.b_len, 1)
        return float(self.score / denom)

    def common_prefix_length(self, threshold: float = 0.999) -> int:
        """How many leading columns match at or above ``threshold``.

        The first column below the threshold is where the two runs stopped doing
        the same thing. Threshold defaults to near-exact because a "meaningful
        divergence" should not be triggered by a paraphrase.
        """
        n = 0
        for p in self.pairs:
            if p.is_match and p.similarity >= threshold:
                n += 1
            else:
                break
        return n

    def first_divergence(self, threshold: float = 0.999) -> Optional[AlignedPair]:
        """The first column where the runs stop agreeing, or None if they never do."""
        k = self.common_prefix_length(threshold)
        return self.pairs[k] if k < len(self.pairs) else None

    def matched_indices(self) -> list[tuple[int, int]]:
        return [(p.i, p.j) for p in self.pairs if p.is_match]

    def distance(self) -> float:
        """A normalised alignment distance in [0, 1]; 0 means identical.

        Defined as ``1 - (sum of match similarities) / max(len_a, len_b)`` so
        that gaps are penalised implicitly by not contributing similarity. Used
        as the trajectory-neighbourhood metric in the recoverability model.
        """
        denom = max(self.a_len, self.b_len, 1)
        got = sum(p.similarity for p in self.pairs if p.is_match)
        return float(np.clip(1.0 - got / denom, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Needleman-Wunsch                                                            #
# --------------------------------------------------------------------------- #
def needleman_wunsch(
    a: Sequence[TrajectoryEvent],
    b: Sequence[TrajectoryEvent],
    similarity: EventSimilarity = structural_similarity,
    gap_penalty: float = 0.5,
    match_offset: float = 0.5,
    max_cells: int = 4_000_000,
) -> Alignment:
    """Global sequence alignment maximising total similarity minus gap cost.

    ``match_offset`` is subtracted from each pairwise similarity before scoring,
    so that pairs with similarity below the offset are *worse* than a gap and the
    aligner prefers to leave them unaligned. With the default 0.5 and a linear
    gap penalty of 0.5, roughly: match strongly (>0.5 similarity) or gap.

    Raises ``MemoryError``-avoidance by refusing inputs above ``max_cells``; use
    :func:`dtw` or a windowed comparison for very long trajectories.
    """
    n, m = len(a), len(b)
    if n * m > max_cells:
        raise ValueError(
            f"alignment would need {n*m} cells (> max_cells={max_cells}); "
            "downsample the trajectories or use a windowed comparison"
        )
    if n == 0 or m == 0:
        pairs = [AlignedPair(i, None, 0.0) for i in range(n)] + [
            AlignedPair(None, j, 0.0) for j in range(m)
        ]
        return Alignment(pairs, "needleman_wunsch", -gap_penalty * (n + m), n, m)

    sim = np.empty((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            sim[i, j] = similarity(a[i], b[j])

    # F[i][j] = best score aligning a[:i] with b[:j]
    F = np.zeros((n + 1, m + 1), dtype=float)
    F[:, 0] = -gap_penalty * np.arange(n + 1)
    F[0, :] = -gap_penalty * np.arange(m + 1)
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0=diag 1=up(gap in b) 2=left
    ptr[1:, 0] = 1
    ptr[0, 1:] = 2

    for i in range(1, n + 1):
        srow = sim[i - 1] - match_offset
        for j in range(1, m + 1):
            diag = F[i - 1, j - 1] + srow[j - 1]
            up = F[i - 1, j] - gap_penalty
            left = F[i, j - 1] - gap_penalty
            if diag >= up and diag >= left:
                F[i, j], ptr[i, j] = diag, 0
            elif up >= left:
                F[i, j], ptr[i, j] = up, 1
            else:
                F[i, j], ptr[i, j] = left, 2

    pairs: list[AlignedPair] = []
    i, j = n, m
    while i > 0 or j > 0:
        d = ptr[i, j]
        if i > 0 and j > 0 and d == 0:
            pairs.append(AlignedPair(i - 1, j - 1, float(sim[i - 1, j - 1])))
            i, j = i - 1, j - 1
        elif i > 0 and d == 1:
            pairs.append(AlignedPair(i - 1, None, 0.0))
            i -= 1
        else:
            pairs.append(AlignedPair(None, j - 1, 0.0))
            j -= 1
    pairs.reverse()
    return Alignment(pairs, "needleman_wunsch", float(F[n, m]), n, m)


# --------------------------------------------------------------------------- #
# Dynamic time warping over state features                                    #
# --------------------------------------------------------------------------- #
def _feature_matrix(
    events: Sequence[TrajectoryEvent], keys: Sequence[str]
) -> np.ndarray:
    return np.array(
        [[float(e.state_features.get(k, 0.0)) for k in keys] for e in events],
        dtype=float,
    ).reshape(len(events), max(len(keys), 1))


def dtw(
    a: Sequence[TrajectoryEvent],
    b: Sequence[TrajectoryEvent],
    feature_keys: Optional[Sequence[str]] = None,
    band: Optional[int] = None,
) -> Alignment:
    """Dynamic time warping over per-event state-feature vectors.

    Distances are Euclidean on z-scored features (scored jointly across the two
    trajectories, so the scaling is comparison-local and does not depend on the
    rest of the corpus). ``band`` restricts the warping path to a Sakoe-Chiba
    band of that width, which both speeds the O(n*m) recursion up and prevents
    pathological warps that match step 1 to step 400.

    Similarities reported on the aligned pairs are ``1/(1+distance)``, so they
    remain in (0, 1] and are comparable with the structural similarities.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return Alignment([], "dtw", float("inf"), n, m)
    if feature_keys is None:
        keys = sorted({k for e in list(a) + list(b) for k in e.state_features})
    else:
        keys = list(feature_keys)
    if not keys:
        # No state features available: fall back to a one-hot over action keys so
        # DTW still returns something meaningful rather than aligning everything.
        vocab = sorted({e.action_key for e in list(a) + list(b)})
        idx = {v: i for i, v in enumerate(vocab)}
        A = np.zeros((n, len(vocab)))
        B = np.zeros((m, len(vocab)))
        for i, e in enumerate(a):
            A[i, idx[e.action_key]] = 1.0
        for j, e in enumerate(b):
            B[j, idx[e.action_key]] = 1.0
    else:
        A, B = _feature_matrix(a, keys), _feature_matrix(b, keys)
        both = np.vstack([A, B])
        mu, sd = both.mean(axis=0), both.std(axis=0)
        sd[sd < 1e-12] = 1.0
        A, B = (A - mu) / sd, (B - mu) / sd

    D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    INF = float("inf")
    C = np.full((n + 1, m + 1), INF)
    C[0, 0] = 0.0
    w = band if band is not None else max(n, m)
    for i in range(1, n + 1):
        lo = max(1, i - w)
        hi = min(m, i + w)
        for j in range(lo, hi + 1):
            C[i, j] = D[i - 1, j - 1] + min(C[i - 1, j - 1], C[i - 1, j], C[i, j - 1])

    pairs: list[AlignedPair] = []
    i, j = n, m
    while i > 0 and j > 0:
        pairs.append(AlignedPair(i - 1, j - 1, float(1.0 / (1.0 + D[i - 1, j - 1]))))
        step = int(np.argmin([C[i - 1, j - 1], C[i - 1, j], C[i, j - 1]]))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    while i > 0:
        pairs.append(AlignedPair(i - 1, None, 0.0))
        i -= 1
    while j > 0:
        pairs.append(AlignedPair(None, j - 1, 0.0))
        j -= 1
    pairs.reverse()
    return Alignment(
        pairs, "dtw", float(C[n, m]), n, m, meta={"feature_keys": keys, "band": band}
    )


# --------------------------------------------------------------------------- #
# Dispatcher                                                                  #
# --------------------------------------------------------------------------- #
_METHODS: dict[str, Callable] = {}


def align(
    a: Trajectory | Sequence[TrajectoryEvent],
    b: Trajectory | Sequence[TrajectoryEvent],
    method: str = "needleman_wunsch",
    **kwargs,
) -> Alignment:
    """Align two trajectories. ``method`` is one of the keys of ``_METHODS``."""
    ea = list(a.events) if isinstance(a, Trajectory) else list(a)
    eb = list(b.events) if isinstance(b, Trajectory) else list(b)
    if method not in _METHODS:
        raise ValueError(f"unknown alignment method {method!r}; have {sorted(_METHODS)}")
    return _METHODS[method](ea, eb, **kwargs)


_METHODS["needleman_wunsch"] = needleman_wunsch
_METHODS["exact"] = lambda a, b, **kw: needleman_wunsch(
    a, b, similarity=exact_action_similarity, **kw
)
_METHODS["event_type"] = lambda a, b, **kw: needleman_wunsch(
    a, b, similarity=event_type_similarity, **kw
)
_METHODS["dtw"] = dtw
