"""Publication-quality figures for the reliability analysis.

Design rules applied throughout, since a figure that misleads is worse than no
figure:

* **Uncertainty is drawn, not implied.** Every point estimate that has an
  interval gets one. Rankings get their intervals too, which is usually the
  moment a reader realises the ordering is not resolved.
* **Vector output by default** (PDF/SVG), because these go in documents.
* **Colour is never the only channel.** Categories differ by marker and position
  as well as hue, so the figures survive greyscale printing and colour-vision
  deficiency.
* **No dual axes, no truncated y-axes on bar charts.** Both are standard ways to
  manufacture an effect.
* **Sample sizes on the figure.** A cell computed from 3 runs is annotated as
  such rather than sharing a colour scale with one computed from 30.

Every function takes an explicit ``ax`` or creates one, returns the Figure, and
never calls ``plt.show`` -- so they compose into multi-panel figures and run
headless.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "set_style",
    "capability_reliability_map",
    "recoverability_ranking",
    "pass_at_k_vs_pass_hat_k",
    "score_distributions",
    "reliability_vs_horizon",
    "failure_entropy_plot",
    "survival_curves",
    "learning_curves",
    "budget_vs_uncertainty",
    "rubric_reliability_matrix",
    "trajectory_embedding_plot",
    "save_all",
]

#: Colour-blind-safe qualitative palette (Okabe-Ito), with markers as a second channel.
_LABEL_STYLE = {
    "SOLID": ("#009E73", "o"),
    "RECOVERABLE": ("#E69F00", "s"),
    "STUCK": ("#D55E00", "X"),
    "UNCERTAIN": ("#999999", "^"),
}


def set_style() -> None:
    """Apply the shared figure style. Idempotent."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.constrained_layout.use": True,
        }
    )


def _new(ax=None, figsize=(6, 4)):
    import matplotlib.pyplot as plt

    if ax is None:
        set_style()
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax


# --------------------------------------------------------------------------- #
def capability_reliability_map(diagnoses: Sequence, ax=None, annotate_top: int = 0):
    """C_t against R_t, the headline diagnostic scatter.

    Reads as four quadrants: top-right is SOLID, bottom-right is the
    high-capability/low-reliability region this project is about, and the left
    edge is STUCK. Point size encodes run count, so a task whose position rests
    on three runs is visibly smaller than one resting on thirty -- the cheapest
    honest way to keep sample size in view on a scatter.
    """
    fig, ax = _new(ax, (5.5, 5))
    seen = set()
    for d in diagnoses:
        colour, marker = _LABEL_STYLE.get(d.label, ("#333333", "."))
        ax.scatter(
            d.capability, d.reliability,
            s=18 + 6 * np.sqrt(max(d.n_runs, 1)),
            c=colour, marker=marker, alpha=0.75, linewidths=0.4, edgecolors="white",
            label=d.label if d.label not in seen else None,
        )
        seen.add(d.label)
    ax.axhline(0.5, color="#666666", lw=0.6, ls=":")
    ax.axvline(0.5, color="#666666", lw=0.6, ls=":")
    ax.set_xlabel(r"estimated capability  $C_t = P(p_t > \tau_{cap})$")
    ax.set_ylabel(r"estimated reliability  $R_t = P(p_t > \tau_{rel})$")
    ax.set_title("Capability vs reliability\n(point size = number of runs)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.text(0.97, 0.14, "capable but\nunreliable", ha="right", va="bottom",
            fontsize=8, color="#B8860B", alpha=0.8, transform=ax.transAxes)
    if seen:
        ax.legend(loc="upper left", title="classification")
    if annotate_top:
        top = sorted(diagnoses, key=lambda d: -d.recoverability)[:annotate_top]
        for d in top:
            ax.annotate(d.task, (d.capability, d.reliability), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    return fig


def recoverability_ranking(diagnoses: Sequence, top_n: int = 20, ax=None):
    """Ranked recoverability with credible intervals on the underlying posterior.

    The intervals are the point of the figure. A ranking whose bars all overlap
    is not a ranking, and the reader should be able to see that immediately
    rather than reading the order as settled.
    """
    fig, ax = _new(ax, (6.5, max(3.0, 0.28 * min(top_n, len(diagnoses)))))
    ranked = sorted(diagnoses, key=lambda d: -d.recoverability)[:top_n][::-1]
    if not ranked:
        ax.text(0.5, 0.5, "no tasks to rank", ha="center", va="center")
        return fig
    y = np.arange(len(ranked))
    scores = [d.recoverability for d in ranked]
    colours = [_LABEL_STYLE.get(d.label, ("#333333", "."))[0] for d in ranked]
    ax.barh(y, scores, color=colours, alpha=0.85, height=0.7)
    for i, d in enumerate(ranked):
        ax.plot([d.ci_lo, d.ci_hi], [i, i], color="#333333", lw=1.0, alpha=0.55)
        ax.plot([d.posterior_mean], [i], marker="|", color="#333333", ms=6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d.task}  (n={d.n_runs})" for d in ranked], fontsize=7)
    ax.set_xlabel("recoverability score  |  grey bar: 95% CI on the posterior mean")
    ax.set_title(f"Top {len(ranked)} post-training targets by estimated recoverability")
    ax.set_xlim(0, 1.02)
    return fig


def pass_at_k_vs_pass_hat_k(df, ks: Sequence[int] = (1, 2, 4, 8), ax=None):
    """pass@k and pass^k on one axis: the reliability gap, drawn.

    They answer different questions -- can it ever, versus can it always -- and
    the shaded gap between them is the fraction of apparent capability that does
    not reproduce.
    """
    from .metrics import pass_at_k, pass_hat_k

    fig, ax = _new(ax, (5.5, 4))
    ks = list(ks)
    at = [pass_at_k(df, k) for k in ks]
    hat = [pass_hat_k(df, k) for k in ks]
    ax.plot(ks, at, "o-", color="#0072B2", label=r"pass@$k$  (at least one succeeds)")
    ax.plot(ks, hat, "s--", color="#D55E00", label=r"pass$^k$  (all $k$ succeed)")
    ax.fill_between(ks, hat, at, color="#999999", alpha=0.18, label="reliability gap")
    ax.set_xlabel("$k$ (repeated attempts)")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(ks)
    ax.set_title("Peak capability vs reproducibility")
    ax.legend(loc="center left")
    return fig


def score_distributions(scores_by_model: Mapping[str, Sequence[float]], ax=None):
    """Per-model run-score distributions, not just their means.

    Two agents with the same mean can have completely different shapes, and the
    mean marker is drawn on top of the distribution precisely to show how little
    it conveys.
    """
    fig, ax = _new(ax, (6, 4))
    names = list(scores_by_model)
    data = [np.asarray(scores_by_model[n], dtype=float) for n in names]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor("#0072B2")
        pc.set_alpha(0.28)
    for i, arr in enumerate(data, start=1):
        ax.scatter([i], [arr.mean()], marker="D", c="#D55E00", zorder=3, s=28,
                   label="mean" if i == 1 else None)
        lo = np.quantile(arr, 0.1)
        ax.scatter([i], [arr[arr <= lo].mean() if (arr <= lo).any() else lo],
                   marker="v", c="#000000", zorder=3, s=22,
                   label="lower-tail CVaR" if i == 1 else None)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("run score")
    ax.set_title("Run-score distributions\n(the mean is one number from this shape)")
    ax.legend()
    return fig


def reliability_vs_horizon(fits: Sequence, horizons=None, ax=None):
    """Fitted P(success | horizon) curves per agent: who degrades more slowly."""
    fig, ax = _new(ax, (5.5, 4))
    horizons = np.asarray(horizons if horizons is not None else np.arange(1, 80))
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for i, f in enumerate(fits):
        ax.plot(horizons, f.predict(horizons), color=palette[i % len(palette)],
                label=f"{f.model or 'agent'} (slope {f.slope:+.3f})")
    ax.set_xlabel(f"task horizon ({fits[0].measure if fits else 'steps'})")
    ax.set_ylabel("P(success)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Reliability decay with task horizon")
    ax.legend()
    return fig


def failure_entropy_plot(distributions: Sequence, ax=None):
    """Failure concentration against failure count, to keep small-N in view."""
    fig, ax = _new(ax, (5.5, 4))
    x = [d.n_failures for d in distributions]
    y = [d.concentration for d in distributions]
    ax.scatter(x, y, c="#CC79A7", alpha=0.7, edgecolors="white", linewidths=0.4)
    ax.set_xlabel("number of observed failures")
    ax.set_ylabel("failure concentration  (1 - normalised entropy)")
    ax.set_title("One repeated defect, or many?\n(entropy from few failures is a weak measurement)")
    ax.set_ylim(-0.02, 1.02)
    ax.axvspan(0, 4, color="#D55E00", alpha=0.07)
    ax.text(0.5, 0.02, "too few failures\nto estimate", fontsize=7, color="#D55E00",
            ha="left", va="bottom")
    return fig


def survival_curves(curves: Mapping[str, object], ax=None):
    """Kaplan-Meier survival across trajectory depth, with Greenwood bands."""
    fig, ax = _new(ax, (5.5, 4))
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for i, (name, curve) in enumerate(curves.items()):
        if getattr(curve, "times", np.array([])).size == 0:
            continue
        c = palette[i % len(palette)]
        ax.step(curve.times, curve.survival, where="post", color=c, label=name)
        lo, hi = curve.ci()
        ax.fill_between(curve.times, lo, hi, step="post", color=c, alpha=0.15)
    ax.set_xlabel("trajectory step")
    ax.set_ylabel("P(no irrecoverable failure yet)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Survival across trajectory depth")
    ax.legend()
    return fig


def learning_curves(df, x="n_pairs", y="heldout_delta", hue="strategy",
                    err="heldout_delta_se", ax=None):
    """Sample-efficiency curves: improvement against training-set size, per strategy.

    The area under these curves, not the endpoint, is the quantity the central
    hypothesis is about -- "more improvement per training example", not "a higher
    final score".
    """
    fig, ax = _new(ax, (6, 4))
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#999999"]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, (name, g) in enumerate(df.groupby(hue)):
        g = g.sort_values(x)
        c, m = palette[i % len(palette)], markers[i % len(markers)]
        ax.plot(g[x], g[y], marker=m, color=c, label=str(name))
        if err and err in g:
            ax.fill_between(g[x], g[y] - g[err], g[y] + g[err], color=c, alpha=0.15)
    ax.axhline(0, color="#333333", lw=0.7, ls=":")
    ax.set_xlabel("training pairs")
    ax.set_ylabel("held-out improvement")
    ax.set_title("Sample efficiency by selection strategy\n(shaded: +/- 1 se across seeds)")
    ax.legend()
    return fig


def budget_vs_uncertainty(df, x="executions", y="agreement", hue="policy", ax=None):
    """Evaluation budget against label quality, per allocation policy."""
    fig, ax = _new(ax, (5.5, 4))
    palette = ["#0072B2", "#D55E00", "#009E73"]
    for i, (name, g) in enumerate(df.groupby(hue)):
        g = g.sort_values(x)
        ax.plot(g[x], g[y], marker="o", color=palette[i % len(palette)], label=str(name))
    ax.set_xlabel("agent executions used")
    ax.set_ylabel("classification agreement with reference")
    ax.set_title("Evaluation budget vs label quality")
    ax.legend()
    return fig


def rubric_reliability_matrix(matrix, ax=None, max_rows: int = 30):
    """Tasks x criteria posterior-mean heatmap. Missing criteria show as gaps."""
    import matplotlib.pyplot as plt

    m = matrix.iloc[:max_rows]
    fig, ax = _new(ax, (max(5, 0.35 * m.shape[1] + 3), max(3, 0.22 * m.shape[0] + 1.5)))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#DDDDDD")
    im = ax.imshow(np.ma.masked_invalid(m.values), aspect="auto", cmap=cmap,
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(m.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels([f"{a}/{b}" for a, b in m.index], fontsize=7)
    ax.set_title("Per-criterion reliability (posterior mean)\ngrey = criterion not present")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="P(criterion satisfied)")
    return fig


def trajectory_embedding_plot(embedding, coords=None, ax=None):
    """2D projection of trajectories, successes vs failures.

    The question it answers: do failed runs sit inside the successful cloud
    (recoverable) or in their own region (a different strategy is needed)?
    """
    from .trajectory.embed import project_2d

    fig, ax = _new(ax, (5.5, 5))
    xy = coords if coords is not None else project_2d(embedding)
    lab = np.asarray(embedding.labels, dtype=bool)
    ax.scatter(xy[lab, 0], xy[lab, 1], c="#009E73", marker="o", s=26, alpha=0.75,
               edgecolors="white", linewidths=0.4, label="success")
    ax.scatter(xy[~lab, 0], xy[~lab, 1], c="#D55E00", marker="X", s=32, alpha=0.8,
               edgecolors="white", linewidths=0.4, label="failure")
    var = embedding.meta.get("explained_variance_ratio")
    if var:
        ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% var)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.0f}% var)")
    ax.set_title("Trajectory embedding\n(are failures near the successful cloud?)")
    ax.legend()
    return fig


def save_all(figures: Mapping[str, object], directory: str, fmt: str = "pdf") -> dict:
    """Save every figure to ``directory`` in ``fmt``. Returns name -> path."""
    from pathlib import Path

    import matplotlib.pyplot as plt

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fig in figures.items():
        if fig is None:
            continue
        p = out / f"{name}.{fmt}"
        fig.savefig(p, format=fmt)
        plt.close(fig)
        paths[name] = str(p)
    return paths
