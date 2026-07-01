"""Tests for the deferred research items: IRT, PPI, and Bradley-Terry ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from disteval import compare, irt, ppi


# ---------------------------------------------------------------------------
# IRT (item response theory)
# ---------------------------------------------------------------------------

def _simulate_2pl(rng, R, n_items):
    theta = rng.normal(0, 1, R)
    a = rng.uniform(0.7, 2.0, n_items)
    b = rng.uniform(-2, 2, n_items)
    p = 1.0 / (1.0 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))
    X = (rng.uniform(size=(R, n_items)) < p).astype(float)
    return X, theta, a, b


class TestIRT:
    def test_fit_recovers_difficulty_and_ability_ordering(self):
        rng = np.random.default_rng(0)
        X, theta, a, b = _simulate_2pl(rng, 300, 20)
        fit = irt.fit_2pl(X)
        # Difficulty and ability are recovered up to standardization -> rank corr.
        assert stats.spearmanr(fit["difficulty"], b).statistic > 0.8
        assert stats.spearmanr(fit["ability"], theta).statistic > 0.8
        assert np.all(fit["discrimination"] > 0)

    def test_fit_recovers_discrimination_trend(self):
        rng = np.random.default_rng(3)
        X, theta, a, b = _simulate_2pl(rng, 400, 20)
        fit = irt.fit_2pl(X)
        assert stats.spearmanr(fit["discrimination"], a).statistic > 0.3

    def test_1pl_has_equal_discriminations(self):
        rng = np.random.default_rng(1)
        X, *_ = _simulate_2pl(rng, 100, 10)
        fit = irt.fit_2pl(X, model="1pl")
        assert np.allclose(fit["discrimination"], fit["discrimination"][0])

    def test_item_information_peaks_at_difficulty(self):
        a, b = np.array([1.5]), np.array([0.0])
        assert irt.item_information(a, b, 0.0) > irt.item_information(a, b, 2.0)

    def test_select_items_prefers_discriminating(self):
        a = np.array([0.1, 2.0, 0.1, 1.5])
        b = np.zeros(4)
        sel = irt.select_items(a, b, 2)
        assert set(sel.tolist()) == {1, 3}

    def test_responses_from_frame(self):
        df = pd.DataFrame({
            "model": ["m1", "m1", "m2", "m2"],
            "task": ["t1", "t2", "t1", "t2"],
            "success": [1, 0, 1, 1],
        })
        mat, models, tasks = irt.responses_from_frame(df)
        assert mat.shape == (2, 2)
        assert set(models) == {"m1", "m2"} and set(tasks) == {"t1", "t2"}

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            irt.fit_2pl(np.zeros((1, 3)))
        with pytest.raises(ValueError):
            irt.fit_2pl(np.zeros(5))


# ---------------------------------------------------------------------------
# PPI (prediction-powered inference)
# ---------------------------------------------------------------------------

class TestPPI:
    def _informative(self, rng, n, N, true_p=0.4, bias=0.3, slope=0.4, noise=0.1):
        gold = (rng.uniform(size=n) < true_p).astype(float)
        judge_gold = np.clip(bias + slope * gold + rng.normal(0, noise, n), 0, 1)
        y_un = (rng.uniform(size=N) < true_p).astype(float)
        judge_un = np.clip(bias + slope * y_un + rng.normal(0, noise, N), 0, 1)
        return gold, judge_gold, judge_un

    def test_informative_judge_tightens_ci(self):
        rng = np.random.default_rng(0)
        gold, jg, ju = self._informative(rng, 60, 3000)
        res = ppi.ppi_mean(gold, jg, ju)
        assert abs(res["point"] - 0.4) < 0.1
        assert res["se"] <= res["gold_only_se"]
        assert 0.0 <= res["lambda"] <= 1.0
        assert res["lo"] < res["point"] < res["hi"]

    def test_uninformative_judge_lambda_near_zero(self):
        rng = np.random.default_rng(2)
        gold = (rng.uniform(size=80) < 0.5).astype(float)
        jg = rng.uniform(0, 1, 80)          # independent of gold
        ju = rng.uniform(0, 1, 2000)
        res = ppi.ppi_mean(gold, jg, ju)
        assert res["lambda"] < 0.2
        assert res["point"] == pytest.approx(gold.mean(), abs=0.1)

    def test_no_unlabeled_falls_back_to_gold(self):
        res = ppi.ppi_mean([0.0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], [])
        assert res["lambda"] == 0.0
        assert res["point"] == pytest.approx(0.5)

    def test_optimal_lambda_bounds(self):
        rng = np.random.default_rng(5)
        gold, jg, ju = self._informative(rng, 40, 1000)
        lam = ppi.optimal_lambda(gold, jg, ju)
        assert 0.0 <= lam <= 1.0


# ---------------------------------------------------------------------------
# Bradley-Terry
# ---------------------------------------------------------------------------

class TestBradleyTerry:
    def test_recovers_clear_ordering(self):
        # System 0 beats 1 beats 2.
        W = np.array([[0, 8, 9], [2, 0, 7], [1, 3, 0]], dtype=float)
        res = compare.bradley_terry(W)
        assert list(res["ranking"]) == [0, 1, 2]
        s = res["strengths"]
        assert s[0] > s[1] > s[2]
        assert res["probs"].sum() == pytest.approx(1.0)

    def test_symmetric_is_tie(self):
        W = np.array([[0, 5, 5], [5, 0, 5], [5, 5, 0]], dtype=float)
        res = compare.bradley_terry(W)
        assert np.allclose(res["strengths"], 0.0, atol=0.05)

    def test_win_matrix_from_pairs(self):
        pairs = [("a", "b", 1), ("a", "b", 1), ("b", "c", 1), ("a", "c", 0.5)]
        W, items = compare.win_matrix_from_pairs(pairs)
        assert items == ["a", "b", "c"]
        i = {n: k for k, n in enumerate(items)}
        assert W[i["a"], i["b"]] == 2.0
        assert W[i["a"], i["c"]] == 0.5 and W[i["c"], i["a"]] == 0.5

    def test_ci_bounds_and_stability_flag(self):
        W = np.array([[0, 8, 9], [2, 0, 7], [1, 3, 0]], dtype=float)
        res = compare.bradley_terry(W, ci=0.9, n_boot=300, seed=1)
        assert "strength_lo" in res and "strength_hi" in res
        for i in range(3):
            assert res["strength_lo"][i] <= res["strengths"][i] <= res["strength_hi"][i]
        assert isinstance(res["ranking_unstable"], bool)
        assert 0.0 <= res["rank_flip_prob"] <= 1.0

    def test_degenerate_all_wins_stays_finite(self):
        # System 0 wins everything; reg keeps strengths finite.
        W = np.array([[0, 10, 10], [0, 0, 5], [0, 5, 0]], dtype=float)
        res = compare.bradley_terry(W)
        assert np.all(np.isfinite(res["strengths"]))
        assert res["ranking"][0] == 0

    def test_requires_square_and_two_systems(self):
        with pytest.raises(ValueError):
            compare.bradley_terry(np.zeros((2, 3)))
        with pytest.raises(ValueError):
            compare.bradley_terry(np.zeros((1, 1)))


# ---------------------------------------------------------------------------
# training_sim: hump-shaped overoptimization model
# ---------------------------------------------------------------------------

class TestOveroptimization:
    def test_hump_rises_then_falls(self):
        from disteval.training_sim import optimal_training_amount, overoptimized_gain
        t_star = optimal_training_amount(alpha=0.6, kl_coef=0.15)["t_star"]
        before = overoptimized_gain(1.0, 0.5 * t_star)
        peak = overoptimized_gain(1.0, t_star)
        after = overoptimized_gain(1.0, 2.0 * t_star)
        assert before < peak
        assert after < peak

    def test_peak_matches_numeric_argmax(self):
        from disteval.training_sim import optimal_training_amount, overoptimized_gain
        opt = optimal_training_amount(alpha=0.6, kl_coef=0.15)
        ts = np.linspace(0, 5, 2001)
        gains = overoptimized_gain(1.0, ts)
        assert abs(ts[int(np.argmax(gains))] - opt["t_star"]) < 0.05
        assert gains.max() == pytest.approx(opt["peak_frac"], abs=1e-3)

    def test_gain_scales_with_gap(self):
        from disteval.training_sim import overoptimized_gain
        assert overoptimized_gain(0.5, 1.3) == pytest.approx(0.5 * overoptimized_gain(1.0, 1.3))

    def test_no_overoptimization_is_monotone(self):
        from disteval.training_sim import optimal_training_amount, overoptimized_gain
        assert optimal_training_amount(kl_coef=0.0)["t_star"] == float("inf")
        assert overoptimized_gain(1.0, 5.0, kl_coef=0.0) >= overoptimized_gain(1.0, 1.0, kl_coef=0.0)

    def test_array_input(self):
        from disteval.training_sim import overoptimized_gain
        g = overoptimized_gain(1.0, np.array([0.0, 1.0, 2.0]))
        assert g.shape == (3,)


# ---------------------------------------------------------------------------
# self_engine: anti-collapse accumulation + diversity filter
# ---------------------------------------------------------------------------

def _mk_pair(task, reinforce_path, contrast_path):
    from disteval.self_engine import TrainingPair
    return TrainingPair(
        task=task, reinforce_traj_path=reinforce_path, contrast_traj_path=contrast_path,
        reinforce_score=1.0, contrast_score=0.0, gap=1.0, structural_divergence_step=0,
    )


def _bare_engine(**kwargs):
    from disteval.records import RecordStore
    from disteval.self_engine import SelfEngine
    from disteval.trajectory_memory import TrajectoryMemory
    return SelfEngine(
        store=RecordStore(), job_dirs=kwargs.pop("job_dirs", []), agent_name="a", model_name="m",
        monitor=object(), memory=TrajectoryMemory(), **kwargs,
    )


class TestDiversityFilter:
    def test_drops_near_duplicate_reinforce(self):
        from types import SimpleNamespace
        eng = _bare_engine(diversity_threshold=0.9)
        eng._traj_records = [
            SimpleNamespace(traj_path="p1", tool_sequence=["read", "write", "exec"]),
            SimpleNamespace(traj_path="p2", tool_sequence=["read", "write", "exec"]),
        ]
        filtered = eng._diversity_filter([_mk_pair("t", "p1", "c1"), _mk_pair("t", "p2", "c2")])
        assert len(filtered) == 1

    def test_keeps_distinct_reinforce(self):
        from types import SimpleNamespace
        eng = _bare_engine(diversity_threshold=0.9)
        eng._traj_records = [
            SimpleNamespace(traj_path="p1", tool_sequence=["read", "read"]),
            SimpleNamespace(traj_path="p2", tool_sequence=["exec", "exec"]),
        ]
        filtered = eng._diversity_filter([_mk_pair("t", "p1", "c1"), _mk_pair("t", "p2", "c2")])
        assert len(filtered) == 2

    def test_keeps_pairs_without_tool_data(self):
        eng = _bare_engine(diversity_threshold=0.9)
        eng._traj_records = []  # no tool data -> cannot assess similarity -> keep all
        filtered = eng._diversity_filter([_mk_pair("t", "p1", "c1"), _mk_pair("t", "p2", "c2")])
        assert len(filtered) == 2

    def test_disabled_by_default(self):
        eng = _bare_engine()
        assert eng.diversity_threshold is None


class TestReloadAccumulation:
    def _engine_with(self, episodes, monkeypatch_new_store=None, monkeypatch=None):
        from disteval.records import EpisodeRecord, RecordStore
        store = RecordStore()
        for ep, score in episodes:
            store.add(EpisodeRecord(run_id="r0", model="m", task="t", episode=ep,
                                    score=score, success=score >= 0.99))
        eng = _bare_engine(job_dirs=["dummy"])
        eng.store = store
        eng._traj_records = []
        return eng

    def test_accumulate_merges_and_dedups(self, monkeypatch):
        from disteval.records import EpisodeRecord, RecordStore
        from disteval.self_engine import SelfEngine

        eng = self._engine_with([(0, 1.0), (1, 0.0)])

        new_store = RecordStore()
        new_store.add(EpisodeRecord(run_id="r0", model="m", task="t", episode=0, score=1.0, success=True))  # dup
        new_store.add(EpisodeRecord(run_id="r0", model="m", task="t", episode=2, score=0.5, success=False))  # new
        fake = _bare_engine(job_dirs=["dummy"])
        fake.store = new_store
        fake._traj_records = []
        monkeypatch.setattr(SelfEngine, "from_job_dirs", classmethod(lambda cls, *a, **k: fake))

        eng.reload(accumulate=True)
        assert len(eng.store) == 3  # episodes 0,1 (old) + 2 (new); dup 0 removed

    def test_no_accumulate_replaces(self, monkeypatch):
        from disteval.records import EpisodeRecord, RecordStore
        from disteval.self_engine import SelfEngine

        eng = self._engine_with([(0, 1.0), (1, 0.0)])
        new_store = RecordStore()
        new_store.add(EpisodeRecord(run_id="r0", model="m", task="t", episode=9, score=0.5, success=False))
        fake = _bare_engine(job_dirs=["dummy"])
        fake.store = new_store
        fake._traj_records = []
        monkeypatch.setattr(SelfEngine, "from_job_dirs", classmethod(lambda cls, *a, **k: fake))

        eng.reload(accumulate=False)
        assert len(eng.store) == 1
