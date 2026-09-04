"""Trajectory embeddings: putting runs in a space where distance means something.

What this is for
----------------
Three concrete questions, each of which needs runs to be comparable as points:

1. **Neighbourhood recoverability.** Do this task's failed runs sit close to its
   successful runs? If a failure is one argument away from a success, preference
   training has a short gradient to travel. If it is in a different region
   entirely, it probably needs a different strategy, not a nudge.
2. **Failure-mode clustering.** Do failures form tight clusters (a few repeated
   defects) or scatter (chaos)? This is the geometric counterpart of failure
   entropy and is computed independently of the taxonomy, so the two are a
   genuine cross-check rather than the same measurement twice.
3. **Whether the embedding predicts training benefit at all** -- which is itself
   one of the hypotheses under test, not an assumption.

Three feature families, deliberately simple
-------------------------------------------
``structural``
    Hand-engineered run statistics: length, error/retry/verification counts,
    tool-call rate, unique-tool count, error position, state-feature summaries.
    Interpretable, cheap, and needs nothing external. This is the default.

``bag_of_actions``
    Normalised counts over the action-key vocabulary, optionally with action
    bigrams. Captures *what* the agent did and in what rough order; this is what
    separates "used the search tool five times" from "never searched".

``text``
    Any user-supplied embedding of the concatenated observation text. No provider
    is imported here; you pass a callable. Absent one, this family is skipped
    rather than faked.

Families are concatenated after per-family L2 normalisation so that no family
dominates purely by dimensionality, and the whole matrix is standardised.

Why not a learned encoder by default
------------------------------------
A learned trajectory encoder needs a training signal, and the only signal
available here is run outcome -- which is what we are trying to predict. Fitting
an encoder on outcome and then reporting that its embedding separates successes
from failures would be circular. :func:`fit_supervised_projection` exists for
when you want it, holds out a fold, and reports honest held-out separation; it
is not used by any default path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .events import EventType, Trajectory

__all__ = [
    "EmbeddingConfig",
    "TrajectoryEmbedding",
    "embed_trajectories",
    "structural_features",
    "neighbourhood_distance",
    "cluster_failures",
    "project_2d",
    "fit_supervised_projection",
]


@dataclass(frozen=True)
class EmbeddingConfig:
    """Which feature families to use and how."""

    structural: bool = True
    bag_of_actions: bool = True
    action_bigrams: bool = True
    text: bool = False
    #: Maps a list of strings to a (n, d) array. Required iff ``text`` is True.
    text_embedder: Optional[Callable[[Sequence[str]], np.ndarray]] = None
    #: Drop action types occurring in fewer than this many trajectories.
    min_action_df: int = 1
    standardize: bool = True


_STRUCTURAL_NAMES = (
    "n_events", "log_n_events", "n_errors", "error_rate", "n_retries",
    "retry_rate", "n_verifications", "n_tool_calls", "tool_call_rate",
    "n_unique_tools", "n_unique_targets", "first_error_frac", "last_error_frac",
    "n_state_changes", "n_retrievals", "mean_state_norm", "max_state_norm",
    "n_rubric_updates",
)


def structural_features(t: Trajectory) -> np.ndarray:
    """Interpretable per-run statistics. Names in ``_STRUCTURAL_NAMES``."""
    n = max(len(t.events), 1)
    errs = [e.index for e in t.events if e.ok is False or e.event_type == EventType.ERROR]
    tools = [e.tool_name for e in t.events if e.tool_name]
    targets = [e.target for e in t.events if e.target]
    norms = [
        float(np.linalg.norm(list(e.state_features.values())))
        for e in t.events if e.state_features
    ]
    return np.array(
        [
            len(t.events),
            np.log1p(len(t.events)),
            len(errs),
            len(errs) / n,
            t.n_retries(),
            t.n_retries() / n,
            t.n_verifications(),
            len(tools),
            len(tools) / n,
            len(set(tools)),
            len(set(targets)),
            (errs[0] / n) if errs else 1.0,
            (errs[-1] / n) if errs else 1.0,
            sum(1 for e in t.events if e.event_type == EventType.STATE_CHANGE),
            sum(1 for e in t.events if e.event_type == EventType.RETRIEVAL),
            float(np.mean(norms)) if norms else 0.0,
            float(np.max(norms)) if norms else 0.0,
            sum(1 for e in t.events if e.rubric_state),
        ],
        dtype=float,
    )


@dataclass
class TrajectoryEmbedding:
    """Embedded trajectories plus the metadata needed to interpret them."""

    matrix: np.ndarray                 # (n_trajectories, d)
    ids: list[str]
    labels: list[bool]                 # success flags
    tasks: list[str]
    feature_names: list[str]
    config: EmbeddingConfig
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.matrix.shape[0]

    def index_of(self, trajectory_id: str) -> int:
        return self.ids.index(trajectory_id)

    def subset(self, mask: Sequence[bool]) -> "TrajectoryEmbedding":
        m = np.asarray(mask, dtype=bool)
        return TrajectoryEmbedding(
            self.matrix[m],
            [i for i, k in zip(self.ids, m) if k],
            [l for l, k in zip(self.labels, m) if k],
            [t for t, k in zip(self.tasks, m) if k],
            list(self.feature_names),
            self.config,
            dict(self.meta),
        )


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return x / n


def embed_trajectories(
    trajectories: Sequence[Trajectory], config: Optional[EmbeddingConfig] = None
) -> TrajectoryEmbedding:
    """Embed a set of trajectories into a common space.

    The vocabulary and any standardisation are fit on exactly the set passed in,
    so distances are comparison-local. Embed all runs of a task together (or the
    whole corpus) -- embedding two subsets separately and comparing them is
    meaningless.
    """
    cfg = config or EmbeddingConfig()
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError("no trajectories to embed")
    if cfg.text and cfg.text_embedder is None:
        raise ValueError("config.text is True but no text_embedder was supplied")

    blocks: list[np.ndarray] = []
    names: list[str] = []

    if cfg.structural:
        M = np.vstack([structural_features(t) for t in trajectories])
        blocks.append(_l2(M))
        names.extend(f"struct::{n}" for n in _STRUCTURAL_NAMES)

    if cfg.bag_of_actions:
        docs = [t.action_keys for t in trajectories]
        if cfg.action_bigrams:
            docs = [
                d + [f"{a}>{b}" for a, b in zip(d, d[1:])] for d in docs
            ]
        df: dict[str, int] = {}
        for d in docs:
            for tok in set(d):
                df[tok] = df.get(tok, 0) + 1
        vocab = sorted(k for k, v in df.items() if v >= cfg.min_action_df)
        idx = {v: i for i, v in enumerate(vocab)}
        M = np.zeros((len(docs), max(len(vocab), 1)))
        for r, d in enumerate(docs):
            for tok in d:
                if tok in idx:
                    M[r, idx[tok]] += 1.0
            if len(d):
                M[r] /= len(d)
        blocks.append(_l2(M))
        names.extend(f"action::{v}" for v in vocab) if vocab else names.append("action::_none")

    if cfg.text:
        texts = [
            " ".join((e.observation or "") for e in t.events)[:8000]
            for t in trajectories
        ]
        M = np.asarray(cfg.text_embedder(texts), dtype=float)
        if M.shape[0] != len(trajectories):
            raise ValueError("text_embedder returned the wrong number of vectors")
        blocks.append(_l2(M))
        names.extend(f"text::{i}" for i in range(M.shape[1]))

    X = np.hstack(blocks)
    if cfg.standardize and X.shape[0] > 1:
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd < 1e-12] = 1.0
        X = (X - mu) / sd

    return TrajectoryEmbedding(
        matrix=X,
        ids=[t.trajectory_id for t in trajectories],
        labels=[bool(t.success) for t in trajectories],
        tasks=[t.task for t in trajectories],
        feature_names=names,
        config=cfg,
        meta={"n_blocks": len(blocks)},
    )


def neighbourhood_distance(
    emb: TrajectoryEmbedding, aggregate: str = "min"
) -> dict:
    """How close are the failures to the successes, in embedding space?

    Returns the per-failure distance to its nearest (or mean) successful run,
    plus a **normalised** summary: the failure-to-success distance divided by the
    typical success-to-success distance. The normalisation is what makes the
    number comparable across tasks -- a raw distance of 2.0 means nothing without
    knowing how spread out that task's successful runs are in the first place.

    A normalised distance near 1 means failures are no further from successes
    than successes are from each other: the failure is "in the neighbourhood",
    which is the geometric reading of recoverable. Large values mean the failed
    runs are doing something structurally different.
    """
    lab = np.asarray(emb.labels, dtype=bool)
    if lab.all() or (~lab).all():
        return {
            "n_success": int(lab.sum()), "n_failure": int((~lab).sum()),
            "mean_distance": float("nan"), "normalized_distance": float("nan"),
            "per_failure": {},
            "note": "runs are all successes or all failures; no comparison possible",
        }
    S, F = emb.matrix[lab], emb.matrix[~lab]
    D = np.linalg.norm(F[:, None, :] - S[None, :, :], axis=2)
    per = D.min(axis=1) if aggregate == "min" else D.mean(axis=1)

    note = ""
    if S.shape[0] > 1:
        SS = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=2)
        iu = np.triu_indices(S.shape[0], k=1)
        scale = float(np.mean(SS[iu]))
        if scale < 1e-9:
            note = ("every successful run embeds identically, so there is no "
                    "success spread to normalise against")
            scale = float("nan")
    else:
        scale = float("nan")
        note = "fewer than two successful runs; no spread to normalise against"

    fail_ids = [i for i, k in zip(emb.ids, ~lab) if k]
    return {
        "n_success": int(lab.sum()),
        "n_failure": int((~lab).sum()),
        "mean_distance": float(per.mean()),
        "min_distance": float(per.min()),
        "success_spread": scale,
        "normalized_distance": float(per.mean() / scale) if scale and scale > 1e-9 else float("nan"),
        "per_failure": dict(zip(fail_ids, map(float, per))),
        "note": note,
    }


def cluster_failures(
    emb: TrajectoryEmbedding, k: Optional[int] = None, max_k: int = 6, seed: int = 0
) -> dict:
    """Cluster the failed runs; report how concentrated the clusters are.

    ``k`` is chosen by silhouette score over ``2..max_k`` when not given. The
    headline is ``dominant_cluster_share``: the fraction of failures in the
    largest cluster, the geometric analogue of failure-mode dominance. Requires
    at least four failed runs; below that clustering is not a measurement.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    lab = np.asarray(emb.labels, dtype=bool)
    F = emb.matrix[~lab]
    ids = [i for i, m in zip(emb.ids, ~lab) if m]
    n = F.shape[0]
    if n < 4:
        return {"n_failures": n, "k": None, "assignments": {},
                "dominant_cluster_share": float("nan"),
                "note": "fewer than 4 failed runs; clustering not attempted"}

    n_distinct = len(np.unique(np.round(F, 9), axis=0))
    if n_distinct < 2:
        # Every failed run embeds to the same point: maximal concentration, and
        # k-means on it is meaningless (and noisy).
        return {"n_failures": n, "k": 1, "assignments": dict.fromkeys(ids, 0),
                "cluster_sizes": [n], "dominant_cluster_share": 1.0,
                "silhouette": float("nan"), "inertia": 0.0,
                "note": "all failed runs embed identically; one cluster by construction"}
    max_k = int(min(max_k, n_distinct))

    if k is None:
        best, best_s = 2, -np.inf
        for kk in range(2, max(min(max_k, n - 1), 2) + 1):
            km = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(F)
            if len(set(km.labels_)) < 2:
                continue
            s = silhouette_score(F, km.labels_)
            if s > best_s:
                best, best_s = kk, s
        k = best
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(F)
    counts = np.bincount(km.labels_, minlength=k)
    return {
        "n_failures": n,
        "k": int(k),
        "assignments": dict(zip(ids, map(int, km.labels_))),
        "cluster_sizes": counts.tolist(),
        "dominant_cluster_share": float(counts.max() / n),
        "silhouette": float(silhouette_score(F, km.labels_)) if len(set(km.labels_)) > 1 else float("nan"),
        "inertia": float(km.inertia_),
    }


def project_2d(emb: TrajectoryEmbedding, method: str = "pca", seed: int = 0) -> np.ndarray:
    """Project to 2D for plotting. ``pca`` always available; ``umap`` if installed.

    PCA is the default because it is deterministic, has no hyperparameters that
    change the apparent story, and its axes carry an explained-variance number.
    UMAP produces prettier separation but can manufacture apparent clusters, so
    it is opt-in.
    """
    if method == "pca":
        from sklearn.decomposition import PCA

        p = PCA(n_components=2, random_state=seed)
        out = p.fit_transform(emb.matrix)
        emb.meta["explained_variance_ratio"] = p.explained_variance_ratio_.tolist()
        return out
    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "umap-learn is not installed; use method='pca' or pip install umap-learn"
            ) from exc
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(emb.matrix)
    raise ValueError(f"unknown projection method {method!r}")


def fit_supervised_projection(
    emb: TrajectoryEmbedding, n_splits: int = 5, seed: int = 0
) -> dict:
    """Honest check of whether the embedding predicts run outcome at all.

    Cross-validated logistic regression on the embedding, reporting held-out AUC.
    This is deliberately *not* used to build any default embedding: fitting an
    encoder on outcome and then reporting that it separates outcomes would be
    circular. It answers a different question -- "is there outcome signal in the
    structure at all?" -- and a near-0.5 AUC is a genuine and useful negative
    result about the feature set.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    y = np.asarray(emb.labels, dtype=int)
    if len(set(y.tolist())) < 2:
        return {"auc": float("nan"), "note": "only one outcome class present"}
    n_splits = int(min(n_splits, np.bincount(y).min()))
    if n_splits < 2:
        return {"auc": float("nan"), "note": "too few runs of the minority class"}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(
        LogisticRegression(max_iter=2000), emb.matrix, y, cv=cv, scoring="roc_auc"
    )
    return {
        "auc": float(scores.mean()),
        "auc_std": float(scores.std()),
        "n_splits": n_splits,
        "n": int(y.size),
    }
