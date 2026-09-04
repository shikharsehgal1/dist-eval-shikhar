"""Failure causality graphs: from "these runs differ here" to a chain of causes.

What this builds
----------------
A per-run DAG whose nodes are attributed failure events and whose sinks are the
rubric criteria that ended up unsatisfied::

    retrieval error -> wrong context -> incorrect spreadsheet edit -> rubric r4 fails

and, at the task level, an aggregated graph over many runs showing which chains
recur.

How edges are justified
-----------------------
No causal discovery is performed and none is claimed. Edges are proposed only
where three conditions hold simultaneously, all of which are observable:

1. **Temporal order.** The parent event precedes the child in the run.
2. **Stage order.** The parent's failure mode is at an earlier pipeline stage
   than the child's (:mod:`disteval.diagnosis.taxonomy` assigns stages).
3. **Dependence.** Either the child event reads something the parent wrote (a
   shared ``target``), or the child's state features moved away from the
   successful runs' states only after the parent, or an explicit criterion
   dependency graph says so.

These are necessary conditions for causation, not sufficient ones, and every
edge carries the reason it was proposed plus a confidence. The output is best
read as "these are the chains consistent with the evidence", which is exactly
what a human triaging a failure wants and is a much stronger statement than an
unstructured diff.

Root cause vs downstream symptom
--------------------------------
The practical payoff is rubric attribution. If one wrong retrieval causes five
rubric criteria to fail, counting five independent capability failures
overstates the deficit fivefold. :func:`attribute_rubric_failures` walks the
graph back from each failed criterion to its root events and reports the
*distinct root count* alongside the raw failed-criterion count.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..trajectory.events import Trajectory
from .taxonomy import FailureLabel, classify_events, stage_of

__all__ = [
    "CausalEdge",
    "FailureGraph",
    "CriterionGraph",
    "build_failure_graph",
    "attribute_rubric_failures",
    "aggregate_graphs",
]


@dataclass(frozen=True)
class CausalEdge:
    """A proposed cause -> effect link, with the evidence that proposed it."""

    src: str          # node id
    dst: str          # node id
    reason: str       # which condition justified it
    confidence: float

    def to_dict(self) -> dict:
        return {"src": self.src, "dst": self.dst, "reason": self.reason,
                "confidence": self.confidence}


@dataclass
class FailureGraph:
    """A DAG over one run's failure events and the rubric criteria they reach."""

    trajectory_id: str
    task: str
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[CausalEdge] = field(default_factory=list)

    def add_node(self, node_id: str, **attrs) -> str:
        self.nodes.setdefault(node_id, {}).update(attrs)
        return node_id

    def add_edge(self, edge: CausalEdge) -> None:
        if edge.src in self.nodes and edge.dst in self.nodes and edge.src != edge.dst:
            self.edges.append(edge)

    def parents(self, node_id: str) -> list[str]:
        return [e.src for e in self.edges if e.dst == node_id]

    def children(self, node_id: str) -> list[str]:
        return [e.dst for e in self.edges if e.src == node_id]

    def transitive_reduction(self) -> "FailureGraph":
        """Drop edges implied by a longer path, so chains stay readable.

        Without this, a run with k attributed failures produces O(2^k) chains
        because every earlier-stage failure is linked to every later one and the
        shortcuts are all redundant. The reduction keeps A->C only when there is
        no A->...->C path of length >= 2.
        """
        reachable: dict[str, set[str]] = {n: set() for n in self.nodes}

        def reach(node: str, seen: set[str]) -> set[str]:
            if reachable[node]:
                return reachable[node]
            acc: set[str] = set()
            for c in self.children(node):
                if c in seen:
                    continue
                acc.add(c)
                acc |= reach(c, seen | {c})
            reachable[node] = acc
            return acc

        for n in self.nodes:
            reach(n, {n})

        keep = []
        for e in self.edges:
            via = {
                c for c in self.children(e.src)
                if c != e.dst and e.dst in reachable.get(c, set())
            }
            if not via:
                keep.append(e)
        g = FailureGraph(self.trajectory_id, self.task, dict(self.nodes), keep)
        return g

    def roots(self) -> list[str]:
        """Nodes with no parents: the candidate root causes."""
        has_parent = {e.dst for e in self.edges}
        return [n for n in self.nodes if n not in has_parent]

    def ancestors(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            cur = stack.pop()
            for p in self.parents(cur):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def root_causes_of(self, node_id: str) -> list[str]:
        """Root nodes reachable backwards from ``node_id``."""
        anc = self.ancestors(node_id)
        if not anc:
            return [node_id] if node_id in self.nodes else []
        roots = set(self.roots())
        found = sorted(anc & roots)
        return found or sorted(anc)

    def chains(self, max_len: int = 8) -> list[list[str]]:
        """All root-to-sink paths, as node-id lists."""
        sinks = {n for n in self.nodes if not self.children(n)}
        out: list[list[str]] = []

        def walk(node: str, path: list[str]) -> None:
            if len(path) > max_len:
                return
            if node in sinks:
                out.append(list(path))
                return
            for c in self.children(node):
                if c not in path:
                    walk(c, path + [c])

        for r in self.roots():
            walk(r, [r])
        return out

    def describe(self, node_id: str) -> str:
        a = self.nodes.get(node_id, {})
        if a.get("kind") == "criterion":
            return f"rubric:{a.get('criterion')}"
        return f"{a.get('mode', '?')}@step{a.get('event_index', '?')}"

    def chain_strings(self, max_len: int = 8) -> list[str]:
        return [" -> ".join(self.describe(n) for n in c) for c in self.chains(max_len)]

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "task": self.task,
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges],
            "roots": self.roots(),
            "chains": self.chain_strings(),
        }


@dataclass
class CriterionGraph:
    """Declared dependencies between rubric criteria: ``r1 -> r4``, ``r2,r3 -> r5``.

    Supplied by the benchmark, not inferred. When present it lets the attribution
    say "r4 failed only because r1 failed" instead of counting two deficits.
    """

    edges: dict[str, list[str]] = field(default_factory=dict)  # child -> parents

    def parents(self, criterion: str) -> list[str]:
        return list(self.edges.get(criterion, []))

    def roots_for(self, criterion: str, failed: set[str]) -> list[str]:
        """Failed ancestors of ``criterion`` that have no failed parents themselves."""
        seen: set[str] = set()
        stack = [criterion]
        while stack:
            cur = stack.pop()
            for p in self.parents(cur):
                if p in failed and p not in seen:
                    seen.add(p)
                    stack.append(p)
        if not seen:
            return [criterion]
        return sorted(p for p in seen if not (set(self.parents(p)) & failed))

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]]) -> "CriterionGraph":
        """Build from ``(parent, child)`` pairs, i.e. ``("r1", "r4")`` for r1 -> r4."""
        edges: dict[str, list[str]] = defaultdict(list)
        for parent, child in pairs:
            edges[child].append(parent)
        return cls(dict(edges))


def build_failure_graph(
    trajectory: Trajectory,
    *,
    failed_criteria: Optional[Sequence[str]] = None,
    criterion_graph: Optional[CriterionGraph] = None,
    labels: Optional[Sequence[FailureLabel]] = None,
    min_confidence: float = 0.3,
    min_edge_confidence: float = 0.3,
    reduce_transitive: bool = True,
) -> FailureGraph:
    """Build the causal DAG for one failed run.

    ``failed_criteria`` defaults to the criteria scoring below 1.0 in
    ``trajectory.rubric_scores``. ``min_edge_confidence`` drops edges justified
    by temporal and stage order alone, which are necessary-condition edges with
    no dependence evidence and which otherwise dominate the graph.
    """
    g = FailureGraph(trajectory.trajectory_id, trajectory.task)
    labels = list(labels) if labels is not None else classify_events(trajectory, min_confidence)

    by_id: dict[str, FailureLabel] = {}
    for lab in labels:
        nid = f"e{lab.event_index}:{lab.mode}"
        by_id[nid] = lab
        g.add_node(
            nid,
            kind="failure",
            mode=lab.mode,
            event_index=lab.event_index,
            stage=stage_of(lab.mode),
            confidence=lab.confidence,
            evidence=lab.evidence,
        )

    events = {e.index: e for e in trajectory.events}

    # -- event -> event edges ------------------------------------------------
    for sid, slab in by_id.items():
        for did, dlab in by_id.items():
            if sid == did:
                continue
            if slab.event_index > dlab.event_index:
                continue          # condition 1: temporal order
            if stage_of(slab.mode) >= stage_of(dlab.mode):
                continue          # condition 2: stage order
            se, de = events.get(slab.event_index), events.get(dlab.event_index)
            reason, conf = None, 0.0
            if se is not None and de is not None:
                if se.target and se.target == de.target:
                    reason, conf = "shared target (child reads what parent wrote)", 0.7
                elif se.target and de.target and se.target in str(de.tool_args):
                    reason, conf = "parent's target appears in child's arguments", 0.6
                elif set(se.state_features) & set(de.state_features):
                    reason, conf = "shared environment state variables", 0.4
                elif slab.event_index == dlab.event_index:
                    reason, conf = "same step, earlier pipeline stage", 0.5
                else:
                    reason, conf = "temporal and stage order only (weak)", 0.25
            if reason and conf * slab.confidence >= min_edge_confidence:
                g.add_edge(CausalEdge(sid, did, reason, conf * slab.confidence))

    # -- failure -> rubric criterion edges ----------------------------------
    if failed_criteria is None:
        failed_criteria = [k for k, v in trajectory.rubric_scores.items() if v < 1.0]
    failed_set = set(failed_criteria)

    for crit in sorted(failed_set):
        cid = f"rubric:{crit}"
        g.add_node(cid, kind="criterion", criterion=crit)
        # Prefer the criterion's own declared root causes when a graph is given.
        if criterion_graph is not None:
            for parent in criterion_graph.parents(crit):
                if parent in failed_set:
                    pid = f"rubric:{parent}"
                    g.add_node(pid, kind="criterion", criterion=parent)
                    g.add_edge(CausalEdge(pid, cid, "declared criterion dependency", 0.9))
        # Any failure event that touched this criterion's state, else the
        # earliest-stage failure in the run.
        touched = [
            nid for nid, lab in by_id.items()
            if crit in (events[lab.event_index].rubric_state or {})
        ]
        if touched:
            for nid in touched:
                g.add_edge(CausalEdge(nid, cid, "event updated this criterion", 0.8))
        elif by_id:
            earliest = min(
                by_id.items(), key=lambda kv: (stage_of(kv[1].mode), kv[1].event_index)
            )[0]
            g.add_edge(
                CausalEdge(earliest, cid, "earliest-stage failure in the run (weak)", 0.3)
            )
    return g.transitive_reduction() if reduce_transitive else g


def _failure_roots(graph: FailureGraph, node: str) -> list[str]:
    """Root causes of ``node``, preferring failure events over criterion nodes."""
    rc = graph.root_causes_of(node)
    failures = [r for r in rc if graph.nodes.get(r, {}).get("kind") == "failure"]
    return sorted(failures) if failures else sorted(rc)


def attribute_rubric_failures(
    graph: FailureGraph,
    criterion_graph: Optional[CriterionGraph] = None,
) -> dict:
    """Separate root-cause failures from downstream rubric symptoms.

    Returns the failed-criterion count, the number of *distinct* root causes,
    and the amplification factor between them. An amplification of 5.0 means one
    underlying defect is being counted as five capability failures.
    """
    crit_nodes = [n for n, a in graph.nodes.items() if a.get("kind") == "criterion"]
    failed = {graph.nodes[n]["criterion"] for n in crit_nodes}

    per_criterion: dict[str, list[str]] = {}
    roots: set[str] = set()
    for n in crit_nodes:
        crit = graph.nodes[n]["criterion"]
        # A criterion that failed only because a declared parent criterion failed
        # is a symptom, so resolve it to that parent -- and then keep resolving,
        # down to the underlying failure event. Counting "rubric:r4" as a root
        # cause would still overstate the deficit.
        target_node = n
        if criterion_graph is not None:
            declared = criterion_graph.roots_for(crit, failed)
            if declared != [crit]:
                rc: list[str] = []
                for d in declared:
                    dn = f"rubric:{d}"
                    rc.extend(_failure_roots(graph, dn))
                per_criterion[crit] = sorted(set(rc))
                roots.update(per_criterion[crit])
                continue
        rc = _failure_roots(graph, target_node)
        per_criterion[crit] = rc
        roots.update(rc)

    n_failed = len(failed)
    n_roots = len(roots)
    return {
        "n_failed_criteria": n_failed,
        "n_root_causes": n_roots,
        "amplification": (n_failed / n_roots) if n_roots else float("nan"),
        "roots": sorted(roots),
        "per_criterion": per_criterion,
        "root_descriptions": sorted(graph.describe(r) for r in roots),
    }


def aggregate_graphs(graphs: Sequence[FailureGraph]) -> dict:
    """Aggregate causal chains across many runs of a task.

    The point is recurrence: a chain seen in 6 of 7 failures is a systematic
    defect and a strong preference-training target; seven distinct chains across
    seven failures is diffuse incompetence and a poor one. This is the structural
    counterpart of the failure-entropy signal.
    """
    chain_counts: dict[str, int] = defaultdict(int)
    root_counts: dict[str, int] = defaultdict(int)
    root_stage: dict[str, int] = {}
    for g in graphs:
        for c in set(g.chain_strings()):
            chain_counts[c] += 1
        for r in g.roots():
            desc = g.describe(r)
            root_stage.setdefault(desc, g.nodes.get(r, {}).get("stage", 99))
        for desc in {g.describe(r) for r in g.roots()}:
            root_counts[desc] += 1
    n = max(len(graphs), 1)

    # Ties on recurrence are broken by pipeline stage: when two roots appear in
    # equally many runs, the earlier-stage one is the better causal candidate.
    def _root_key(desc: str) -> tuple[int, int]:
        return (-root_counts[desc], root_stage.get(desc, 99))
    return {
        "n_graphs": len(graphs),
        "chains": dict(sorted(chain_counts.items(), key=lambda kv: -kv[1])),
        "roots": dict(sorted(root_counts.items(), key=lambda kv: _root_key(kv[0]))),
        "dominant_chain": max(chain_counts, key=chain_counts.get) if chain_counts else None,
        "dominant_chain_share": (max(chain_counts.values()) / n) if chain_counts else 0.0,
        "dominant_root": min(root_counts, key=_root_key) if root_counts else None,
        "dominant_root_share": (max(root_counts.values()) / n) if root_counts else 0.0,
    }
