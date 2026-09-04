"""Failure taxonomy, causality graphs, failure entropy and survival analysis."""
import numpy as np
import pytest

from disteval.diagnosis.causality import (
    CriterionGraph,
    aggregate_graphs,
    attribute_rubric_failures,
    build_failure_graph,
)
from disteval.diagnosis.entropy import (
    dominant_mode_share,
    entropy_ci,
    failure_distribution,
    normalized_entropy,
    plugin_entropy,
)
from disteval.diagnosis.survival import (
    hazard_by_step,
    kaplan_meier,
    recovery_probability,
    run_survival_record,
    survival_summary,
    time_to_first_error,
)
from disteval.diagnosis.taxonomy import (
    TAXONOMY,
    FailureAttributor,
    FailureLabel,
    FailureMode,
    classify_events,
    classify_failure,
    register_mode,
    stage_of,
)
from disteval.trajectory.events import EventType, Trajectory, from_generic_steps


def _traj(steps, tid="t", success=False, task="task"):
    t = from_generic_steps(steps, tid, task, "m", score=1.0 if success else 0.0,
                           success=success)
    return t


class TestTaxonomy:
    def test_default_modes_are_stage_ordered(self):
        stages = [m.stage for m in TAXONOMY.values()]
        assert len(set(stages)) == len(stages), "stages must be distinct"
        assert stage_of("retrieval") < stage_of("synthesis")

    def test_unknown_mode_sorts_last(self):
        assert stage_of("does_not_exist") == 99

    def test_retrieval_failure_is_detected_and_is_primary(self):
        t = _traj([
            {"tool": "fetch", "event_type": "retrieval", "target": "x", "ok": False,
             "observation": "not found"},
            {"tool": "submit", "event_type": "final_answer", "ok": False},
        ])
        mode, labels = classify_failure(t)
        assert mode == "retrieval", "earliest-stage failure wins, not highest confidence"
        assert any(l.mode == "synthesis" for l in labels), "downstream still recorded"

    def test_repeated_identical_retries_flag_recovery(self):
        steps = [{"tool": "x", "args": {"a": 1}, "ok": False} for _ in range(3)]
        labels = classify_events(_traj(steps))
        assert any(l.mode == "recovery" for l in labels)

    def test_no_structural_trace_returns_unknown(self):
        mode, labels = classify_failure(_traj([{"tool": "a", "ok": True}]))
        assert mode == "unknown" and labels == []

    def test_one_event_can_carry_several_labels(self):
        t = _traj([{"tool": "fetch", "event_type": "retrieval", "ok": False}])
        labels = classify_events(t)
        assert len({l.mode for l in labels}) > 1

    def test_taxonomy_is_extensible(self):
        register_mode(FailureMode("demo_mode", 42, "test-only",
                                  lambda e, t: 1.0 if e.tool_name == "zzz" else 0.0))
        try:
            labels = classify_events(_traj([{"tool": "zzz"}]))
            assert any(l.mode == "demo_mode" for l in labels)
        finally:
            TAXONOMY.pop("demo_mode", None)

    def test_attributor_without_a_judge_returns_heuristics_only(self):
        t = _traj([{"tool": "f", "event_type": "retrieval", "ok": False}])
        assert FailureAttributor().label(t) == classify_events(t)

    def test_attributor_retains_heuristics_alongside_model_labels(self):
        t = _traj([{"tool": "f", "event_type": "retrieval", "ok": False}])
        judge = lambda tr, h: [FailureLabel("planning", 0, 0.9, "llm said so", "llm")]  # noqa: E731
        out = FailureAttributor(judge).label(t)
        assert any(l.source == "heuristic" for l in out)
        assert any(l.source == "llm" for l in out)


class TestCausality:
    def _run(self):
        t = _traj([
            {"tool": "fetch", "event_type": "retrieval", "target": "d.csv", "ok": False},
            {"tool": "edit", "event_type": "state_change", "target": "d.csv", "ok": False,
             "rubric_state": {"r4": False}},
            {"tool": "submit", "event_type": "final_answer", "ok": False},
        ])
        t.rubric_scores = {"r1": 1.0, "r4": 0.0, "r5": 0.0}
        return t

    def test_edges_carry_their_justification(self):
        g = build_failure_graph(self._run())
        assert g.edges
        assert all(e.reason and 0 < e.confidence <= 1 for e in g.edges)

    def test_weak_order_only_edges_are_dropped_by_default(self):
        g = build_failure_graph(self._run())
        assert not any("order only" in e.reason for e in g.edges)

    def test_transitive_reduction_keeps_chains_readable(self):
        full = build_failure_graph(self._run(), reduce_transitive=False)
        red = build_failure_graph(self._run(), reduce_transitive=True)
        assert len(red.edges) < len(full.edges)
        assert len(red.chain_strings()) < len(full.chain_strings())

    def test_root_cause_is_the_earliest_stage_failure(self):
        g = build_failure_graph(self._run())
        assert any("retrieval" in g.describe(r) for r in g.roots())

    def test_amplification_counts_one_root_for_two_criteria(self):
        cg = CriterionGraph.from_pairs([("r4", "r5")])
        g = build_failure_graph(self._run(), criterion_graph=cg)
        out = attribute_rubric_failures(g, cg)
        assert out["n_failed_criteria"] == 2
        assert out["n_root_causes"] == 1
        assert out["amplification"] == pytest.approx(2.0)

    def test_criterion_graph_resolves_transitively(self):
        cg = CriterionGraph.from_pairs([("a", "b"), ("b", "c")])
        assert cg.roots_for("c", {"a", "b", "c"}) == ["a"]

    def test_criterion_with_no_failing_parents_is_its_own_root(self):
        cg = CriterionGraph.from_pairs([("a", "b")])
        assert cg.roots_for("b", {"b"}) == ["b"]

    def test_aggregate_finds_recurring_chains(self):
        g = build_failure_graph(self._run())
        out = aggregate_graphs([g, g, g])
        assert out["n_graphs"] == 3
        assert out["dominant_chain_share"] == pytest.approx(1.0)

    def test_aggregate_breaks_root_ties_by_earliest_stage(self):
        g = build_failure_graph(self._run())
        assert "retrieval" in aggregate_graphs([g, g])["dominant_root"]

    def test_empty_aggregate(self):
        assert aggregate_graphs([])["n_graphs"] == 0

    def test_graph_serialises(self):
        d = build_failure_graph(self._run()).to_dict()
        assert {"nodes", "edges", "roots", "chains"} <= set(d)


class TestFailureEntropy:
    def test_concentrated_beats_diffuse(self):
        a = failure_distribution(["retrieval"] * 8)
        b = failure_distribution(["retrieval", "planning", "memory", "recovery",
                                  "synthesis", "reasoning", "verification", "memory"])
        assert a.concentration > b.concentration
        assert a.dominant_share > b.dominant_share

    def test_normalised_entropy_is_in_unit_interval(self):
        for modes in (["a"] * 5, ["a", "b", "c"], ["a", "a", "b"]):
            v = failure_distribution(modes, support=["a", "b", "c"]).normalized
            assert 0.0 <= v <= 1.0

    def test_uniform_over_the_support_is_maximal(self):
        d = failure_distribution(["a", "b", "c"] * 20, support=["a", "b", "c"])
        assert d.normalized > 0.98

    def test_no_failures_gives_nan_not_zero(self):
        d = failure_distribution([])
        assert d.n_failures == 0
        assert np.isnan(d.entropy) or np.isnan(d.normalized)

    def test_plugin_entropy_basics(self):
        assert plugin_entropy([1.0]) == pytest.approx(0.0)
        assert plugin_entropy([0.5, 0.5]) == pytest.approx(np.log(2))
        assert np.isnan(plugin_entropy([]))

    def test_miller_madow_raises_the_estimate(self):
        modes = ["a", "b", "a", "c"]
        with_c = failure_distribution(modes, support=list("abcd"), miller_madow=True)
        without = failure_distribution(modes, support=list("abcd"), miller_madow=False)
        assert with_c.entropy > without.entropy

    def test_default_alpha_does_not_swamp_the_data(self):
        d = failure_distribution(["a"] * 6, support=list("abcdefghij"))
        assert d.probs["a"] > 0.8, "one pseudo-observation total, not one per mode"

    def test_dominant_share_is_support_free(self):
        assert dominant_mode_share(["a", "a", "b"]) == pytest.approx(2 / 3)
        assert np.isnan(dominant_mode_share([]))

    def test_bootstrap_interval_brackets_and_widens_at_small_n(self):
        wide = entropy_ci(["a", "b", "c", "d"], n_boot=300)
        narrow = entropy_ci(["a"] * 40, n_boot=300)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_ci_needs_two_observations(self):
        assert all(np.isnan(x) for x in entropy_ci(["a"]))

    def test_normalized_entropy_helper(self):
        assert 0 <= normalized_entropy(["a", "b"], support=["a", "b"]) <= 1


class TestSurvival:
    def _runs(self, n=24, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n):
            k = int(rng.integers(6, 14))
            fail = bool(rng.random() < 0.5)
            steps = [{"tool": "a", "ok": True} for _ in range(k)]
            if fail:
                steps[int(k * 0.6)] = {"tool": "a", "ok": False}
            out.append(_traj(steps, f"r{i}", success=not fail))
        return out

    def test_successful_runs_are_censored_not_events(self):
        r = run_survival_record(_traj([{"tool": "a", "ok": True}] * 5, success=True))
        assert r.event is False
        assert r.time == 4

    def test_failed_run_event_is_its_last_error(self):
        t = _traj([{"tool": "a", "ok": False}, {"tool": "a", "ok": True},
                   {"tool": "a", "ok": False}], success=False)
        r = run_survival_record(t)
        assert r.event is True
        assert r.time == 2, "last error, not first"
        assert r.first_error_step == 0

    def test_failed_run_with_no_errors_fails_at_the_end(self):
        r = run_survival_record(_traj([{"tool": "a", "ok": True}] * 4, success=False))
        assert r.event is True and r.time == 3

    def test_custom_irrecoverability_rule_is_honoured(self):
        t = _traj([{"tool": "a", "ok": True}] * 5, success=True)
        r = run_survival_record(t, irrecoverable=lambda _t: 2)
        assert r.event is True and r.time == 2

    def test_recovered_from_error_flag(self):
        t = _traj([{"tool": "a", "ok": False}, {"tool": "a", "ok": True}], success=True)
        assert run_survival_record(t).recovered_from_error is True

    def test_kaplan_meier_is_monotone_non_increasing(self):
        c = kaplan_meier([run_survival_record(t) for t in self._runs()])
        assert np.all(np.diff(c.survival) <= 1e-12)
        assert np.all((c.survival >= 0) & (c.survival <= 1))

    def test_greenwood_interval_brackets_the_curve(self):
        c = kaplan_meier([run_survival_record(t) for t in self._runs()])
        lo, hi = c.ci()
        assert np.all(lo <= c.survival + 1e-9)
        assert np.all(hi >= c.survival - 1e-9)

    def test_censoring_is_handled_not_dropped(self):
        """All-success runs are censored, so survival must stay at 1."""
        recs = [run_survival_record(_traj([{"tool": "a", "ok": True}] * 5, success=True))
                for _ in range(6)]
        c = kaplan_meier(recs)
        assert c.n_events == 0
        assert c.at(10) == pytest.approx(1.0)

    def test_empty_curve(self):
        c = kaplan_meier([])
        assert c.n_runs == 0 and c.times.size == 0
        assert c.at(3) == 1.0

    def test_hazard_binning_pools_small_risk_sets(self):
        recs = [run_survival_record(t) for t in self._runs()]
        assert len(hazard_by_step(recs, n_bins=5)) == 5
        assert len(hazard_by_step(recs)) > 5

    def test_time_to_first_error_separates_clean_runs(self):
        out = time_to_first_error(self._runs())
        assert out["n"] + out["n_error_free"] == 24

    def test_recovery_probability_contrasts_the_two_populations(self):
        out = recovery_probability(self._runs())
        assert out["n_with_error"] + out["n_without_error"] == 24
        assert np.isfinite(out["error_cost"])

    def test_summary_is_complete_and_serialisable(self):
        s = survival_summary(self._runs())
        assert {"n_runs", "n_irrecoverable", "recovery", "time_to_first_error"} <= set(s)
        assert isinstance(s["hazard_table"], list)
