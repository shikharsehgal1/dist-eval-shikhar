"""Trajectory events, alignment, divergence, embeddings and counterfactuals."""
import numpy as np
import pytest

from disteval.trajectory.align import (
    align,
    dtw,
    event_type_similarity,
    exact_action_similarity,
    needleman_wunsch,
    structural_similarity,
)
from disteval.trajectory.counterfactual import (
    EditCosts,
    intervention_distance,
    minimal_intervention,
)
from disteval.trajectory.divergence import (
    EmbeddingComparator,
    NullComparator,
    analyse_divergence,
    divergence_matrix,
)
from disteval.trajectory.embed import (
    EmbeddingConfig,
    cluster_failures,
    embed_trajectories,
    fit_supervised_projection,
    neighbourhood_distance,
    project_2d,
    structural_features,
)
from disteval.trajectory.events import (
    EventType,
    Trajectory,
    TrajectoryEvent,
    TrajectorySet,
    from_generic_steps,
    register_event_type,
)


def _steps(*specs):
    """(tool, target, ok) triples -> loose step dicts."""
    return [{"tool": t, "args": {"path": g}, "target": g, "ok": ok} for t, g, ok in specs]


def _traj(tid, specs, success, task="task", model="m"):
    return from_generic_steps(_steps(*specs), tid, task, model,
                              score=1.0 if success else 0.0, success=success)


class TestEvents:
    def test_action_key_is_coarse_and_arg_key_is_fine(self):
        a = TrajectoryEvent(0, EventType.TOOL_CALL, tool_name="read", target="a.csv")
        b = TrajectoryEvent(0, EventType.TOOL_CALL, tool_name="read", target="b.csv")
        assert a.action_key == b.action_key
        assert a.arg_key != b.arg_key

    def test_arg_key_is_stable_under_argument_ordering(self):
        a = TrajectoryEvent(0, tool_name="f", tool_args={"x": 1, "y": 2})
        b = TrajectoryEvent(0, tool_name="f", tool_args={"y": 2, "x": 1})
        assert a.arg_key == b.arg_key

    def test_event_type_registry_is_extensible(self):
        name = register_event_type("spreadsheet_edit")
        from disteval.trajectory.events import EVENT_TYPES

        assert name in EVENT_TYPES
        with pytest.raises(ValueError):
            register_event_type("not an identifier")

    def test_generic_step_mapping_infers_types(self):
        t = from_generic_steps(
            [{"tool": "search", "args": {"q": "x"}, "output": "ok"},
             {"tool": "read", "error": "boom"},
             {"thought": "hmm", "observation": "thinking"}],
            "t1", "task", "m", score=0.0,
        )
        assert t.events[0].event_type == EventType.TOOL_CALL
        assert t.events[1].event_type == EventType.ERROR
        assert t.n_errors() == 1
        assert t.first_error().tool_name == "read"

    def test_roundtrip_through_dict(self):
        t = _traj("a", [("read", "x", True)], True)
        back = Trajectory.from_dict(t.to_dict())
        assert back.trajectory_id == t.trajectory_id
        assert [e.arg_key for e in back] == [e.arg_key for e in t]

    def test_from_dict_accepts_aliases_and_reports_missing_fields(self):
        t = Trajectory.from_dict({"id": "x", "task_id": "T", "agent": "m", "events": []})
        assert (t.trajectory_id, t.task, t.model) == ("x", "T", "m")
        with pytest.raises(ValueError, match="missing required field"):
            Trajectory.from_dict({"events": []})

    def test_trajectory_set_queries(self, tmp_path):
        ts = TrajectorySet([_traj("s1", [("a", "x", True)], True),
                            _traj("f1", [("a", "x", False)], False)])
        assert len(ts.successes("task")) == 1
        assert len(ts.failures("task")) == 1
        p = tmp_path / "t.jsonl"
        ts.to_jsonl(str(p))
        assert len(TrajectorySet.from_jsonl(str(p))) == 2

    def test_empty_trajectory_is_handled(self):
        t = Trajectory("e", "task", "m", events=[])
        assert len(t) == 0
        assert t.first_error() is None
        assert t.n_errors() == 0


class TestAlignment:
    def test_identical_trajectories_align_perfectly(self):
        a = _traj("a", [("s", "1", True), ("r", "2", True)], True)
        al = align(a, a)
        assert al.n_gaps == 0
        assert al.distance() == pytest.approx(0.0, abs=1e-9)
        assert al.common_prefix_length() == 2

    def test_inserted_step_becomes_a_gap_not_a_cascade(self):
        """The reason alignment exists: an extra step must not shift everything."""
        a = _traj("a", [("s", "1", True), ("r", "2", True), ("w", "3", True)], True)
        b = _traj("b", [("s", "1", True), ("retry", "1", False),
                        ("r", "2", True), ("w", "3", True)], False)
        al = align(a, b)
        assert al.n_gaps == 1
        matched = al.matched_indices()
        assert (1, 2) in matched and (2, 3) in matched

    def test_single_argument_difference_is_localised(self):
        a = _traj("a", [("s", "1", True), ("r", "good", True), ("w", "3", True)], True)
        b = _traj("b", [("s", "1", True), ("r", "bad", True), ("w", "3", True)], False)
        al = align(a, b)
        assert al.common_prefix_length() == 1
        assert al.n_gaps == 0

    def test_similarity_functions_are_bounded(self):
        a = TrajectoryEvent(0, EventType.TOOL_CALL, tool_name="x", target="p")
        b = TrajectoryEvent(0, EventType.RETRIEVAL, tool_name="y", target="q")
        for fn in (structural_similarity, exact_action_similarity, event_type_similarity):
            assert 0.0 <= fn(a, b) <= 1.0
            assert fn(a, a) == pytest.approx(1.0)

    def test_empty_inputs(self):
        a = _traj("a", [("s", "1", True)], True)
        empty = Trajectory("e", "task", "m", events=[])
        al = align(a, empty)
        assert al.n_matched == 0 and al.n_gaps == 1

    def test_all_methods_run_and_agree_on_identity(self):
        a = _traj("a", [("s", "1", True), ("r", "2", True)], True)
        for method in ("needleman_wunsch", "exact", "event_type", "dtw"):
            al = align(a, a, method=method)
            assert al.method in ("needleman_wunsch", "dtw")
            assert al.n_matched == 2

    def test_unknown_method_raises(self):
        a = _traj("a", [("s", "1", True)], True)
        with pytest.raises(ValueError):
            align(a, a, method="telepathy")

    def test_refuses_oversized_alignments(self):
        big = [TrajectoryEvent(i, tool_name="x") for i in range(3000)]
        with pytest.raises(ValueError, match="max_cells"):
            needleman_wunsch(big, big, max_cells=1000)

    def test_dtw_handles_many_to_one(self):
        a = [TrajectoryEvent(i, tool_name="x", state_features={"p": i / 3})
             for i in range(3)]
        b = [TrajectoryEvent(i, tool_name="x", state_features={"p": i / 6})
             for i in range(6)]
        al = dtw(a, b)
        assert al.n_matched >= 6

    def test_dtw_band_restricts_the_path(self):
        a = [TrajectoryEvent(i, tool_name="x", state_features={"p": i}) for i in range(12)]
        assert dtw(a, a, band=1).n_matched > 0

    def test_distance_is_in_unit_interval(self):
        a = _traj("a", [("s", "1", True), ("r", "2", True)], True)
        b = _traj("b", [("z", "9", False)], False)
        assert 0.0 <= align(a, b).distance() <= 1.0


class TestDivergence:
    def test_identifies_the_wrong_file(self):
        s = _traj("s", [("search", "idx", True), ("read", "q3.csv", True),
                        ("calc", "sheet", True)], True)
        f = _traj("f", [("search", "idx", True), ("read", "q2.csv", True),
                        ("calc", "sheet", True)], False)
        rep = analyse_divergence(s, f)
        assert rep.primary.kind == "first_argument_divergence"
        assert rep.primary.j == 1
        assert rep.common_prefix == 1

    def test_identifies_a_failed_execution(self):
        s = _traj("s", [("a", "1", True), ("b", "2", True)], True)
        f = _traj("f", [("a", "1", True), ("b", "2", False)], False)
        rep = analyse_divergence(s, f)
        assert rep.candidate("first_failed_execution") is not None

    def test_identical_runs_have_no_divergence(self):
        s = _traj("s", [("a", "1", True), ("b", "2", True)], True)
        f = _traj("f", [("a", "1", True), ("b", "2", True)], False)
        rep = analyse_divergence(s, f)
        assert rep.primary is None
        assert rep.consequence_strength == 0.0

    def test_consequence_strength_reflects_downstream_damage(self):
        s = _traj("s", [("a", "1", True), ("b", "2", True), ("c", "3", True),
                        ("d", "4", True)], True)
        f = _traj("f", [("a", "1", True), ("x", "9", True), ("y", "8", True),
                        ("z", "7", True)], False)
        rep = analyse_divergence(s, f)
        assert rep.consequence_strength > 0.5

    def test_action_realignment_alone_is_not_recovery(self):
        """A run that reads the wrong file then repeats the same later steps."""
        s = _traj("s", [("r", "good", True), ("c", "x", True), ("w", "y", True)], True)
        f = _traj("f", [("r", "bad", True), ("c", "x", True), ("w", "y", True)], False)
        f.events[2].rubric_state = {"r1": False}
        s.events[2].rubric_state = {"r1": True}
        rep = analyse_divergence(s, f)
        assert rep.recovered is False

    def test_null_comparator_declines_to_judge(self):
        assert NullComparator().compare("a", "b") == (0.0, 0.0)

    def test_embedding_comparator_uses_the_supplied_function(self):
        calls = []

        def embed(texts):
            calls.append(list(texts))
            return np.array([[1.0, 0.0]] * len(texts))

        c = EmbeddingComparator(embed)
        sim, conf = c.compare("x", "y")
        assert sim == pytest.approx(1.0)
        assert conf > 0
        c.compare("x", "y")  # cached
        assert len(calls) == 2, "each distinct string embedded once"

    def test_divergence_matrix_covers_every_pair(self):
        succ = [_traj(f"s{i}", [("a", "1", True)], True) for i in range(2)]
        fail = [_traj(f"f{i}", [("a", "2", True)], False) for i in range(3)]
        assert len(divergence_matrix(succ, fail)) == 6

    def test_report_is_serialisable(self):
        s = _traj("s", [("r", "good", True)], True)
        f = _traj("f", [("r", "bad", True)], False)
        d = analyse_divergence(s, f).to_dict()
        assert d["primary_kind"] == "first_argument_divergence"
        assert "primary_rule" in d


class TestCounterfactual:
    def _pair(self):
        s = [_traj(f"s{i}", [("search", "idx", True), ("read", "good", True),
                             ("calc", "x", True)], True) for i in range(2)]
        f = [_traj(f"f{i}", [("search", "idx", True), ("read", "bad", True),
                             ("calc", "x", True)], False) for i in range(2)]
        return s, f

    def test_one_retarget_is_the_minimal_fix(self):
        s, f = self._pair()
        est = minimal_intervention(f[0], s)
        assert est.n_edits == 1
        assert est.edits[0].kind == "retarget"
        assert est.dominant_edit == "retarget"

    def test_structurally_different_failure_costs_more(self):
        s, f = self._pair()
        odd = _traj("odd", [("guess", "z", True)], False)
        assert (minimal_intervention(odd, s).normalized_cost
                > minimal_intervention(f[0], s).normalized_cost)

    def test_undefined_without_any_success(self):
        _s, f = self._pair()
        est = minimal_intervention(f[0], [])
        assert est.is_bound is False
        assert np.isnan(est.total_cost)
        assert "undefined, not zero" in est.note

    def test_takes_the_minimum_over_all_references(self):
        s, f = self._pair()
        far = _traj("far", [("q", "1", True), ("w", "2", True), ("e", "3", True),
                            ("r", "4", True)], True)
        est = minimal_intervention(f[0], [far] + s)
        assert est.reference_id in {"s0", "s1"}
        assert est.n_references_considered == 3

    def test_costs_are_configurable(self):
        s, f = self._pair()
        cheap = minimal_intervention(f[0], s, costs=EditCosts(retarget=0.1))
        dear = minimal_intervention(f[0], s, costs=EditCosts(retarget=10.0))
        assert dear.total_cost > cheap.total_cost

    def test_task_level_consistency_detects_a_shared_fix(self):
        s, f = self._pair()
        out = intervention_distance(f, s)
        assert out["consistency"] == pytest.approx(1.0)
        assert out["dominant_edit"] == "retarget"

    def test_task_level_handles_no_successes(self):
        _s, f = self._pair()
        out = intervention_distance(f, [])
        assert out["n_estimated"] == 0
        assert np.isnan(out["mean_normalized_cost"])


class TestEmbedding:
    def _corpus(self, n_success=6, n_fail=6, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n_success):
            k = int(rng.integers(3, 6))
            out.append(_traj(f"s{i}", *[[("a", str(j), True) for j in range(k)]], True))
        for i in range(n_fail):
            k = int(rng.integers(3, 6))
            specs = [("a", str(j), True) for j in range(k)] + [("b", "x", False)]
            out.append(_traj(f"f{i}", specs, False))
        return out

    def test_structural_features_have_the_declared_length(self):
        from disteval.trajectory.embed import _STRUCTURAL_NAMES

        t = _traj("a", [("x", "1", True), ("y", "2", False)], False)
        assert structural_features(t).shape == (len(_STRUCTURAL_NAMES),)

    def test_embedding_shape_and_names(self):
        emb = embed_trajectories(self._corpus())
        assert emb.matrix.shape[0] == 12
        assert emb.matrix.shape[1] == len(emb.feature_names)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            embed_trajectories([])

    def test_text_family_requires_an_embedder(self):
        with pytest.raises(ValueError, match="text_embedder"):
            embed_trajectories(self._corpus(), EmbeddingConfig(text=True))

    def test_neighbourhood_distance_is_normalised(self):
        out = neighbourhood_distance(embed_trajectories(self._corpus()))
        assert out["n_success"] == 6 and out["n_failure"] == 6
        assert np.isfinite(out["normalized_distance"])

    def test_neighbourhood_needs_both_classes(self):
        out = neighbourhood_distance(embed_trajectories(self._corpus(n_fail=0)))
        assert np.isnan(out["normalized_distance"])
        assert "no comparison possible" in out["note"]

    def test_clustering_declines_below_four_failures(self):
        out = cluster_failures(embed_trajectories(self._corpus(n_fail=2)))
        assert out["k"] is None
        assert "fewer than 4" in out["note"]

    def test_clustering_handles_identical_failures(self):
        traj = [_traj(f"s{i}", [("a", "1", True)], True) for i in range(4)]
        traj += [_traj(f"f{i}", [("a", "2", True)], False) for i in range(5)]
        out = cluster_failures(embed_trajectories(traj))
        assert out["dominant_cluster_share"] == pytest.approx(1.0)

    def test_pca_projection_reports_explained_variance(self):
        emb = embed_trajectories(self._corpus())
        xy = project_2d(emb)
        assert xy.shape == (12, 2)
        assert len(emb.meta["explained_variance_ratio"]) == 2

    def test_unknown_projection_raises(self):
        with pytest.raises(ValueError):
            project_2d(embed_trajectories(self._corpus()), method="magic")

    def test_supervised_projection_reports_held_out_auc(self):
        out = fit_supervised_projection(embed_trajectories(self._corpus()))
        assert np.isnan(out["auc"]) or 0.0 <= out["auc"] <= 1.0

    def test_supervised_projection_with_one_class(self):
        out = fit_supervised_projection(embed_trajectories(self._corpus(n_fail=0)))
        assert np.isnan(out["auc"])
        assert "one outcome class" in out["note"]
