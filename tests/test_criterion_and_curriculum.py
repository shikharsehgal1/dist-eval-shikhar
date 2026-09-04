"""Criterion-level gap estimation and the six curriculum strategies."""
import numpy as np
import pytest

from disteval.reliability.classify import ReliabilityThresholds
from disteval.reliability.criterion import (
    capability_reliability_gap,
    empirical_bayes_criteria,
    fit_criterion_model,
    gap_profile,
    gap_profiles,
)
from disteval.reliability.recoverability import incremental_validity
from disteval.selection.export import (
    VIEWS,
    dataset_cost,
    export_views,
    to_listwise,
    to_pairwise,
    to_scalar_reward,
    to_weighted_sft,
)
from disteval.selection.selectors import CURRICULUM_STRATEGIES, make_selector
from disteval.reliability.classify import diagnose
from disteval.reliability.posterior import posterior_from_scores
from disteval.selection.pairs import build_dataset
from disteval.trajectory.events import from_generic_steps

TH = ReliabilityThresholds()


def _runs(pattern, n=8):
    """pattern: {criterion: probability-of-satisfaction as a repeating list}."""
    return [{c: float(vals[i % len(vals)]) for c, vals in pattern.items()}
            for i in range(n)]


class TestCriterionGap:
    def test_stable_task_has_a_small_gap(self):
        p = gap_profile("t", _runs({"a": [1], "b": [1], "c": [1]}), thresholds=TH)
        assert p.gap < 0.3
        assert p.reliability > 0.7

    def test_one_unstable_criterion_concentrates_the_gap(self):
        """Nine stable criteria and one coin flip: the gap must localise."""
        pattern = {f"c{i}": [1] for i in range(9)}
        pattern["flaky"] = [1, 0]
        p = gap_profile("t", _runs(pattern), thresholds=TH)
        assert p.dominant_criterion == "flaky"
        # Compared against a uniform spread (1/J), not against 0.5: nine stable
        # criteria still carry a small residual gap each, because P(p>0.9) is
        # only 0.81 after 8/8 successes, and that dilutes the raw share.
        assert p.gap_concentration_ratio > 3.0
        assert p.n_unstable() == 1

    def test_uniformly_mediocre_task_spreads_the_gap(self):
        pattern = {f"c{i}": [1, 0] for i in range(10)}
        p = gap_profile("t", _runs(pattern), thresholds=TH)
        assert p.gap_concentration_ratio == pytest.approx(1.0, abs=0.15)
        assert p.n_unstable() == 10

    def test_the_two_are_distinguished_despite_equal_pass_rates(self):
        """The motivating case: same mean score, different structure."""
        one_flaky = {f"c{i}": [1] for i in range(9)}
        one_flaky["flaky"] = [1, 0]
        all_half = {f"c{i}": [1, 0] for i in range(10)}
        a = gap_profile("a", _runs(one_flaky), thresholds=TH)
        b = gap_profile("b", _runs(all_half), thresholds=TH)
        assert a.gap_concentration_ratio > b.gap_concentration_ratio
        assert a.n_unstable() < b.n_unstable()

    def test_gap_is_bounded_and_uses_no_maximum(self):
        for pattern in ({"a": [1]}, {"a": [0]}, {"a": [1, 0]}, {"a": [1, 0, 0, 0]}):
            p = gap_profile("t", _runs(pattern), thresholds=TH)
            assert 0.0 <= p.capability <= 1.0
            assert 0.0 <= p.reliability <= 1.0
            assert -1e-9 <= p.gap <= 1.0

    def test_gap_does_not_grow_with_run_count(self):
        """The property the observed maximum fails: stability in n."""
        pattern = {"a": [1, 0, 0, 0]}
        gaps = [gap_profile("t", _runs(pattern, n=n), thresholds=TH).gap
                for n in (4, 8, 16, 32, 64)]
        assert max(gaps) - min(gaps) < 0.35
        assert gaps[-1] == pytest.approx(gaps[-2], abs=0.1)

    def test_joint_reliability_is_estimated_not_multiplied(self):
        """Criteria each within reach but never satisfied together."""
        runs = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}] * 4
        p = gap_profile("t", runs, thresholds=TH)
        assert p.capability > 0.8, "each criterion is individually achievable"
        assert p.joint_reliability < 0.2, "but never both at once"
        assert p.joint_capability < 0.5

    def test_joint_reliability_is_high_when_task_is_solved(self):
        p = gap_profile("t", _runs({"a": [1], "b": [1]}), thresholds=TH)
        assert p.joint_reliability > 0.85

    def test_empty_criteria(self):
        p = gap_profile("t", [{}], thresholds=TH)
        assert p.n_criteria == 0
        assert np.isnan(p.gap_concentration)
        assert p.dominant_criterion is None

    def test_missing_criteria_are_unobserved_not_failed(self):
        runs = [{"a": 1.0, "b": 1.0}, {"a": 1.0}]
        p = gap_profile("t", runs, thresholds=TH)
        b = [c for c in p.criteria if c.criterion == "b"][0]
        assert b.n_runs == 1 and b.n_satisfied == 1

    def test_serialises(self):
        d = gap_profile("t", _runs({"a": [1, 0]}), thresholds=TH).to_dict()
        assert {"gap", "gap_concentration", "joint_reliability", "capability"} <= set(d)


class TestPooledEstimators:
    def _corpus(self, n_tasks=12, seed=0):
        rng = np.random.default_rng(seed)
        out = {}
        for i in range(n_tasks):
            base = rng.uniform(0.2, 0.9)
            out[f"t{i}"] = [
                {"a": float(rng.random() < base), "b": float(rng.random() < base * 0.7),
                 "c": float(rng.random() < 0.95)}
                for _ in range(8)
            ]
        return out

    def test_hierarchical_fit_has_one_cell_per_task_criterion(self):
        corpus = self._corpus()
        fit = fit_criterion_model(corpus)
        assert len(fit.cells) == len(corpus) * 3
        assert ("t0", "a") in fit.cells

    def test_empirical_bayes_pools_by_criterion(self):
        posts = empirical_bayes_criteria(self._corpus())
        assert len(posts) == 36
        assert all(0 < p.mean < 1 for p in posts.values())

    def test_empirical_bayes_ranking_tracks_the_unpooled_one(self):
        from scipy.stats import spearmanr

        corpus = self._corpus(n_tasks=20)
        a = capability_reliability_gap(corpus, estimator="independent")
        b = capability_reliability_gap(corpus, estimator="empirical_bayes")
        ks = sorted(a)
        assert spearmanr([a[k] for k in ks], [b[k] for k in ks]).statistic > 0.6

    def test_gap_is_non_monotone_in_performance(self):
        """The property that makes the estimator choice substantive.

        G_t is an inverted U in p: near zero for never-solved and for
        always-solved criteria, maximal in between. This is why shrinkage can
        reorder a gap ranking rather than merely compress it, and it is
        documented on the module rather than left for a user to discover.
        """
        gaps = [
            gap_profile("t", _runs({"a": [1] * k + [0] * (8 - k)}, n=8),
                        thresholds=TH).gap
            for k in range(9)
        ]
        assert gaps[0] < max(gaps) and gaps[-1] < max(gaps)
        peak = int(np.argmax(gaps))
        assert 0 < peak < 8, "the maximum must be interior, not at an endpoint"

    def test_pooling_reduces_the_spread_of_estimates(self):
        corpus = self._corpus(n_tasks=20)
        indep = list(capability_reliability_gap(corpus, estimator="independent").values())
        pooled = list(capability_reliability_gap(corpus, estimator="hierarchical").values())
        assert np.std(pooled) <= np.std(indep) + 1e-9

    def test_falls_back_below_four_cells(self):
        tiny = {"t0": [{"a": 1.0}, {"a": 0.0}]}
        profs = gap_profiles(tiny, estimator="hierarchical")
        assert len(profs) == 1 and np.isfinite(profs[0].gap)

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError):
            fit_criterion_model({})


class TestCurriculumSelectors:
    def _diags(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n):
            p = float(rng.random())
            s = list((rng.random(8) < p).astype(float))
            out.append(diagnose(posterior_from_scores(s), task=f"t{i}", model="m",
                                scores=s))
        return out

    def test_all_six_are_registered(self):
        assert len(CURRICULUM_STRATEGIES) == 6
        for name in CURRICULUM_STRATEGIES:
            assert make_selector(name) is not None

    @pytest.mark.parametrize("name", CURRICULUM_STRATEGIES)
    def test_each_respects_the_budget_and_is_reproducible(self, name):
        d = self._diags()
        a = make_selector(name, seed=3).select(d, 10)
        b = make_selector(name, seed=3).select(d, 10)
        assert a.delivered <= 10 and a.tasks == b.tasks

    def test_legacy_names_still_work(self):
        for name in ("random", "hardest", "lowest_mean", "highest_variance",
                     "success_failure", "recoverability", "uncertainty_aware"):
            assert make_selector(name).select(self._diags(), 5).delivered <= 5

    def test_uniform_and_random_behave_identically(self):
        d = self._diags()
        assert (make_selector("uniform", seed=1).select(d, 8).tasks
                == make_selector("random", seed=1).select(d, 8).tasks)

    def test_difficulty_and_hardest_behave_identically(self):
        d = self._diags()
        assert (make_selector("difficulty").select(d, 8).tasks
                == make_selector("hardest").select(d, 8).tasks)

    def test_uncertainty_picks_the_widest_posteriors(self):
        d = self._diags()
        chosen = make_selector("uncertainty").select(d, 5).tasks
        sds = {x.task: x.posterior_sd for x in d}
        assert min(sds[t] for t in chosen) >= max(
            v for k, v in sds.items() if k not in chosen
        ) - 1e-9

    def test_learning_progress_uses_history_when_given(self):
        d = self._diags(n=10)
        history = {f"t{i}": [0.1, 0.1, 0.1] for i in range(10)}
        history["t7"] = [0.1, 0.1, 0.9, 0.9]  # the only task actually moving
        r = make_selector("learning_progress", history=history, window=2).select(d, 3)
        assert r.params["used_history"] is True
        assert "t7" in r.tasks

    def test_learning_progress_falls_back_and_says_so(self):
        r = make_selector("learning_progress").select(self._diags(), 5)
        assert r.params["used_history"] is False
        assert "learnability proxy" in r.note

    def test_learnability_proxy_peaks_at_mid_performance(self):
        from disteval.selection.selectors import LearningProgressSelector

        mid = diagnose(posterior_from_scores([1, 0] * 8), task="mid")
        solved = diagnose(posterior_from_scores([1] * 16), task="solved")
        hopeless = diagnose(posterior_from_scores([0] * 16), task="hopeless")
        f = LearningProgressSelector._learnability
        assert f(mid) > f(solved) and f(mid) > f(hopeless)

    def test_gap_selector_uses_criterion_profiles(self):
        d = self._diags(n=10)
        profiles = {
            p.task: p for p in gap_profiles(
                {f"t{i}": _runs({"a": [1, 0] if i < 5 else [1]}) for i in range(10)},
                thresholds=TH,
            )
        }
        r = make_selector("capability_reliability_gap", gap=profiles).select(d, 5)
        assert all("criterion-level gap" in r.reasons[t] for t in r.tasks)
        assert set(r.tasks) <= {f"t{i}" for i in range(5)}

    def test_gap_selector_reports_task_level_fallback(self):
        d = self._diags(n=10)
        r = make_selector("capability_reliability_gap", gap={}).select(d, 5)
        assert r.params["n_task_level_fallback"] == 5
        assert "task-level posterior gap" in r.note

    def test_joint_capability_gate_filters(self):
        d = self._diags(n=6)
        # Each criterion achievable, never both together -> low joint capability.
        never = {f"t{i}": [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}] * 4
                 for i in range(3)}
        always = {f"t{i}": _runs({"a": [1, 0], "b": [1, 0]}) for i in range(3, 6)}
        profiles = {p.task: p for p in gap_profiles({**never, **always}, thresholds=TH)}
        gated = make_selector("capability_reliability_gap", gap=profiles,
                              require_joint_capability=0.5).select(d, 6)
        assert gated.delivered < 6

    def test_gap_plus_structure_uses_structure(self):
        d = self._diags(n=10)
        gap = {f"t{i}": 0.5 for i in range(10)}
        structure = {f"t{i}": {"failure_concentration": 1.0 if i < 3 else 0.0}
                     for i in range(10)}
        r = make_selector("gap_plus_structure", gap=gap, structure=structure).select(d, 3)
        assert set(r.tasks) == {"t0", "t1", "t2"}

    def test_structure_weight_zero_recovers_the_plain_gap(self):
        d = self._diags(n=12)
        gap = {f"t{i}": float(i) / 12 for i in range(12)}
        structure = {f"t{i}": {"failure_concentration": 1.0 - i / 12} for i in range(12)}
        plain = make_selector("capability_reliability_gap", gap=gap).select(d, 5)
        ablated = make_selector("gap_plus_structure", gap=gap, structure=structure,
                                gap_weight=1.0, structure_weight=0.0).select(d, 5)
        assert plain.tasks == ablated.tasks

    def test_intervention_cost_is_inverted(self):
        d = self._diags(n=6)
        gap = {f"t{i}": 0.5 for i in range(6)}
        structure = {f"t{i}": {"intervention": 0.1 if i < 2 else 5.0} for i in range(6)}
        r = make_selector("gap_plus_structure", gap=gap, structure=structure).select(d, 2)
        assert set(r.tasks) == {"t0", "t1"}, "cheaper interventions rank higher"

    def test_missing_structure_is_reported_not_imputed(self):
        d = self._diags(n=6)
        r = make_selector("gap_plus_structure", gap={f"t{i}": 0.5 for i in range(6)},
                          structure={}).select(d, 4)
        assert r.params["n_with_structure"] == 0
        assert "no trajectory-structure signals" in r.note


class TestOptimiserAgnosticExport:
    def _dataset(self):
        trajs = {}
        for t in ("a", "b"):
            trajs[t] = [
                from_generic_steps(
                    [{"tool": "s", "args": {"q": "x"}, "output": "ok"},
                     {"tool": "r", "args": {"p": "good" if i < 2 else f"bad{i}"},
                      "target": "good" if i < 2 else f"bad{i}"}],
                    f"{t}_{i}", t, "m", score=1.0 if i < 2 else 0.0, success=i < 2)
                for i in range(5)
            ]
        return build_dataset(["a", "b"], trajs, strategy="gap_plus_structure",
                             target_pairs=10)

    def test_every_view_renders(self):
        ds = self._dataset()
        for fn in (to_pairwise, to_listwise, to_weighted_sft, to_scalar_reward):
            rows = fn(ds)
            assert rows and all(isinstance(r, dict) for r in rows)

    def test_views_come_from_one_selection(self):
        ds = self._dataset()
        tasks = {r["task_id"] for r in to_listwise(ds)}
        assert tasks == {r["task_id"] for r in to_scalar_reward(ds)}
        assert tasks <= set(ds.selected_tasks)

    def test_weighted_sft_keeps_only_successes_and_weights_them(self):
        rows = to_weighted_sft(self._dataset())
        assert all(r["score"] == 1.0 for r in rows)
        assert all(r["weight"] > 0 for r in rows)

    def test_scalar_reward_keeps_both_outcomes(self):
        rewards = {r["reward"] for r in to_scalar_reward(self._dataset())}
        assert rewards == {0.0, 1.0}

    def test_cost_accounting_is_populated(self):
        c = dataset_cost(self._dataset())
        assert c.n_examples > 0 and c.n_trajectories > 0
        assert c.n_events > 0 and c.n_tasks == 2

    def test_export_writes_every_view_and_a_manifest(self, tmp_path):
        import json

        export_views(self._dataset(), tmp_path)
        for v in VIEWS:
            assert (tmp_path / f"{v}.jsonl").exists()
        m = json.loads((tmp_path / "manifest.json").read_text())
        assert set(m["views"]) == set(VIEWS)
        assert m["cost"]["n_examples"] > 0

    def test_unknown_view_raises(self, tmp_path):
        from disteval.selection.export import export_view

        with pytest.raises(ValueError):
            export_view(self._dataset(), "telepathy", tmp_path / "x.jsonl")


class TestIncrementalValidity:
    def _features(self, n, rng, struct):
        return {f"t{i}": {"difficulty": float(rng.random()),
                          "learning_progress": float(rng.random()),
                          "gap": float(struct[i])} for i in range(n)}

    def test_detects_genuine_incremental_signal(self):
        rng = np.random.default_rng(0)
        n = 300
        struct = rng.random(n)
        f = self._features(n, rng, struct)
        y = {t: 0.5 * f[t]["difficulty"] + 0.3 * f[t]["learning_progress"]
                + 0.6 * f[t]["gap"] + float(rng.normal(0, 0.2)) for t in f}
        out = incremental_validity(f, y, added=("gap",))
        assert out["supported"] is True
        assert out["delta_r2"] > 0 and out["delta_ci_lo"] > 0

    def test_rejects_pure_noise(self):
        rng = np.random.default_rng(1)
        n = 300
        f = self._features(n, rng, rng.random(n))
        y = {t: 0.5 * f[t]["difficulty"] + 0.3 * f[t]["learning_progress"]
                + float(rng.normal(0, 0.2)) for t in f}
        out = incremental_validity(f, y, added=("gap",))
        assert out["supported"] is False
        assert "no evidence" in out["verdict"]

    def test_reports_insufficient_data_rather_than_guessing(self):
        f = {f"t{i}": {"difficulty": 0.1 * i, "learning_progress": 0.2,
                       "gap": 0.3} for i in range(6)}
        out = incremental_validity(f, dict.fromkeys(f, 0.5))
        assert out["verdict"] == "insufficient data"

    def test_constant_added_feature_is_not_testable(self):
        rng = np.random.default_rng(2)
        n = 60
        f = {f"t{i}": {"difficulty": float(rng.random()),
                       "learning_progress": float(rng.random()),
                       "gap": 0.5} for i in range(n)}
        out = incremental_validity(f, {t: float(rng.random()) for t in f},
                                   added=("gap",))
        assert out["verdict"] == "not testable"

    def test_uses_out_of_fold_scores(self):
        rng = np.random.default_rng(3)
        n = 200
        f = self._features(n, rng, rng.random(n))
        out = incremental_validity(f, {t: float(rng.random()) for t in f},
                                   added=("gap",))
        assert out["r2_base"] < 0.5, "out-of-fold R^2 on noise must be near zero"
        assert "out-of-fold" in out["note"]


class TestCLIExperimentWiring:
    """The flagship config must not compare two identical selectors.

    Without trajectories and per-criterion scores, the criterion-level gap falls
    back to its task-level form and the structure signals are empty, which makes
    `gap_plus_structure` numerically identical to `capability_reliability_gap`.
    The comparison the config exists to run would then be vacuous by
    construction, so the simulator path must supply both.
    """

    def test_simulator_path_populates_gap_and_structure(self):
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        out = subprocess.run(
            [sys.executable, "-m", "disteval", "experiment",
             "configs/curriculum_baselines.yaml", "-o", "/tmp/_disteval_wiring_test",
             "--n-tasks", "120", "--coupling", "weak", "--overwrite"],
            capture_output=True, text=True, cwd=str(root), timeout=600,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        assert "trajectories per task" in out.stdout, (
            "the simulator path must generate trajectories, or the gap and "
            "structure selectors collapse to the same thing"
        )

        rows = {}
        for line in out.stdout.splitlines():
            parts = line.split()
            if len(parts) > 4 and parts[0] in (
                "gap_plus_structure", "capability_reliability_gap"
            ):
                rows[parts[0]] = parts[2]
        assert len(rows) == 2, f"expected both gap selectors in the summary, got {rows}"
        assert rows["gap_plus_structure"] != rows["capability_reliability_gap"], (
            "gap_plus_structure and capability_reliability_gap produced identical "
            "results, which means the structure signals were not populated"
        )
