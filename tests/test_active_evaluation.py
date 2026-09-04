"""Value of information, adaptive allocation, and sequential stopping.

The load-bearing property here is the budget guarantee: no policy, and no
composition of policy and stopping rule, may ever exceed the configured budget.
"""
import numpy as np
import pytest

from disteval.active.allocation import (
    GreedyValuePolicy,
    StratifiedUniform,
    TaskState,
    ThompsonAllocation,
    UniformAllocation,
    make_policy,
)
from disteval.active.evaluator import AdaptiveEvaluator, compare_allocation
from disteval.active.stopping import (
    AllOf,
    AnyOf,
    ConfidentLabel,
    IntervalWidth,
    MaxRuns,
    MinRuns,
    NoValue,
    StoppingLedger,
    default_stopping,
)
from disteval.active.value import (
    bald,
    expected_entropy_reduction,
    expected_runs_to_decide,
    flip_probability,
    label_of,
    rank_instability,
)
from disteval.reliability.classify import ReliabilityThresholds
from disteval.reliability.posterior import binary_posterior

TH = ReliabilityThresholds()


class TestValueOfInformation:
    def test_entropy_reduction_is_non_negative(self):
        for s, n in [(0, 1), (3, 8), (8, 8), (0, 20), (10, 40)]:
            assert expected_entropy_reduction(binary_posterior(s, n)) >= -1e-12

    def test_entropy_reduction_falls_as_evidence_accumulates(self):
        few = expected_entropy_reduction(binary_posterior(2, 4))
        many = expected_entropy_reduction(binary_posterior(20, 40))
        assert many < few

    def test_bald_is_non_negative_and_bounded(self):
        for s, n in [(0, 4), (2, 4), (4, 4), (16, 32)]:
            v = bald(binary_posterior(s, n))
            assert 0.0 <= v <= np.log(2) + 1e-9

    def test_bald_prefers_epistemic_over_aleatoric_uncertainty(self):
        """A well-established fair coin has an uncertain outcome, nothing to learn."""
        known_coin = bald(binary_posterior(200, 400))
        unknown = bald(binary_posterior(2, 4))
        assert unknown > known_coin

    def test_flip_probability_is_a_probability(self):
        for s, n in [(0, 6), (3, 6), (8, 8), (0, 20), (1, 2)]:
            v = flip_probability(binary_posterior(s, n), TH)
            assert 0.0 <= v <= 1.0

    def test_multi_step_horizon_sees_what_one_step_cannot(self):
        """0/6 is UNCERTAIN and no single run can change that; two can."""
        p = binary_posterior(0, 6)
        assert label_of(p, TH) == "UNCERTAIN"
        assert flip_probability(p, TH, horizon=1) == pytest.approx(0.0, abs=1e-9)
        assert flip_probability(p, TH, horizon=2) > 0.5

    def test_horizon_is_extended_to_reach_min_runs(self):
        """A brand-new task must not score zero and be starved."""
        assert flip_probability(binary_posterior(0, 0), TH) > 0.0

    def test_horizon_must_be_positive(self):
        with pytest.raises(ValueError):
            flip_probability(binary_posterior(1, 2), TH, horizon=0)

    def test_settled_tasks_have_no_value(self):
        assert flip_probability(binary_posterior(4, 8), TH, horizon=2) == pytest.approx(0.0)

    def test_expected_runs_to_decide(self):
        assert expected_runs_to_decide(binary_posterior(4, 8), TH) == 0.0
        assert expected_runs_to_decide(binary_posterior(0, 6), TH) == 2.0
        assert expected_runs_to_decide(binary_posterior(0, 0), TH) >= 1

    def test_rank_instability_flags_the_boundary_not_the_extremes(self):
        posts = [binary_posterior(i, 8) for i in range(9)]
        inst = rank_instability(posts, top_k=3, n_samples=200, seed=0)
        assert inst.shape == (9,)
        assert np.all((inst >= 0) & (inst <= 1))
        assert inst[8] < inst[5] or inst[0] < inst[5]

    def test_rank_instability_on_empty(self):
        assert rank_instability([]).size == 0


class TestAllocationBudget:
    """The budget guarantee, checked exhaustively across policies and budgets."""

    def _states(self):
        return [
            TaskState("solid", 8, 8.0), TaskState("stuck", 8, 0.0),
            TaskState("recov", 8, 3.0), TaskState("unclear", 2, 1.0),
            TaskState("fresh", 0, 0.0), TaskState("mid", 5, 2.0),
        ]

    @pytest.mark.parametrize("name", ["uniform", "greedy_voi", "thompson",
                                      "stratified_uniform"])
    @pytest.mark.parametrize("budget", [0, 1, 3, 7, 20, 61])
    def test_never_exceeds_the_budget(self, name, budget):
        r = make_policy(name).allocate(self._states(), budget, seed=0)
        assert r.spent <= budget
        assert sum(r.allocation.values()) == r.spent

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError):
            UniformAllocation().allocate(self._states(), -1)

    def test_stopped_tasks_receive_nothing(self):
        states = self._states()
        states[0].stopped = True
        r = make_policy("greedy_voi").allocate(states, 30, seed=0)
        assert "solid" not in r.allocation

    def test_all_stopped_spends_nothing(self):
        states = self._states()
        for s in states:
            s.stopped = True
        assert UniformAllocation().allocate(states, 50).spent == 0

    def test_uniform_is_balanced(self):
        r = UniformAllocation().allocate(self._states(), 12)
        assert set(r.allocation.values()) == {2}

    def test_greedy_skips_settled_tasks(self):
        r = GreedyValuePolicy(thresholds=TH).allocate(self._states(), 24, seed=0)
        assert r.allocation.get("recov", 0) == 0, "a settled task is not worth runs"
        assert r.allocation.get("fresh", 0) > 0, "a fresh task must be sampled"

    def test_greedy_does_not_fixate_on_one_task(self):
        """The bug that motivated sampling the rollout instead of its expectation."""
        r = GreedyValuePolicy(thresholds=TH).allocate(self._states(), 24, seed=0)
        assert max(r.allocation.values()) < 24

    def test_greedy_is_reproducible(self):
        a = GreedyValuePolicy(thresholds=TH).allocate(self._states(), 15, seed=3)
        b = GreedyValuePolicy(thresholds=TH).allocate(self._states(), 15, seed=3)
        assert a.allocation == b.allocation

    def test_greedy_rejects_unknown_criterion(self):
        with pytest.raises(ValueError):
            GreedyValuePolicy(criterion="vibes")

    def test_stratified_covers_every_stratum(self):
        states = [TaskState(f"t{i}", 4, 2.0, domain=f"d{i % 3}") for i in range(9)]
        r = StratifiedUniform().allocate(states, 18, seed=0)
        assert len({s.domain for s in states if r.allocation.get(s.task)}) == 3

    def test_thompson_explores_several_tasks(self):
        r = ThompsonAllocation(thresholds=TH).allocate(self._states(), 30, seed=0)
        assert len(r.allocation) >= 2

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError):
            make_policy("crystal_ball")

    def test_result_serialises(self):
        d = UniformAllocation().allocate(self._states(), 6).to_dict()
        assert d["spent"] == 6 and "allocation" in d


class TestStopping:
    def test_confident_solid_and_stuck(self):
        rule = ConfidentLabel(TH, confidence=0.95)
        assert rule.should_stop(TaskState("a", 24, 24.0))[0] is True
        assert rule.should_stop(TaskState("b", 24, 0.0))[0] is True

    def test_recoverable_is_held_to_a_stricter_standard(self):
        """2/4 supports RECOVERABLE but is far too imprecise to rank."""
        rule = ConfidentLabel(TH, confidence=0.95)
        assert rule.should_stop(TaskState("mid", 4, 2.0))[0] is False
        assert rule.should_stop(TaskState("mid", 30, 12.0))[0] is True

    def test_interval_width_rule(self):
        assert IntervalWidth(0.2).should_stop(TaskState("a", 100, 50.0))[0] is True
        assert IntervalWidth(0.2).should_stop(TaskState("a", 4, 2.0))[0] is False

    def test_max_and_min_runs(self):
        assert MaxRuns(10).should_stop(TaskState("a", 10, 5.0))[0] is True
        assert MaxRuns(10).should_stop(TaskState("a", 9, 5.0))[0] is False
        assert MinRuns(3).should_stop(TaskState("a", 3, 1.0))[0] is True
        assert MinRuns(3).should_stop(TaskState("a", 2, 1.0))[0] is False

    def test_no_value_horizon_matters(self):
        st = TaskState("a", 4, 2.0)
        assert NoValue(0.02, TH, horizon=2).should_stop(st)[0] is True
        assert NoValue(0.02, TH, horizon=8).should_stop(st)[0] is False

    def test_composition_operators(self):
        rule = MinRuns(3) & MaxRuns(100)
        assert isinstance(rule, AllOf)
        assert (MaxRuns(1) | MinRuns(99)).should_stop(TaskState("a", 5, 2.0))[0] is True

    def test_any_of_and_all_of_semantics(self):
        never = MaxRuns(10**9)
        always = MaxRuns(0)
        assert AnyOf([never, always]).should_stop(TaskState("a", 1, 0.0))[0] is True
        assert AllOf([never, always]).should_stop(TaskState("a", 1, 0.0))[0] is False

    def test_every_stop_reports_a_reason(self):
        rule = default_stopping(TH)
        stop, why = rule.should_stop(TaskState("a", 24, 24.0))
        assert stop and why

    def test_default_respects_the_floor_and_the_cap(self):
        rule = default_stopping(TH, min_runs=3, max_runs=12)
        assert rule.should_stop(TaskState("a", 2, 2.0))[0] is False
        assert rule.should_stop(TaskState("a", 12, 6.0))[0] is True

    def test_ledger_records_and_summarises(self):
        led = StoppingLedger()
        led.record(TaskState("a", 8, 4.0), "because")
        assert led.summary()["n_stopped"] == 1
        assert len(led.to_frame()) == 1
        assert StoppingLedger().summary()["n_stopped"] == 0


class TestAdaptiveEvaluator:
    def _world(self, seed=0):
        rng = np.random.default_rng(seed)
        true_p = {f"t{i}": float(rng.choice([0.02, 0.4, 0.5, 0.98])) for i in range(24)}

        def run_fn(task, i):
            return float(rng.random() < true_p[task])

        return list(true_p), run_fn, true_p

    def test_never_exceeds_the_budget(self):
        tasks, run_fn, _ = self._world()
        for budget in (0, 5, 50, 240):
            tr = AdaptiveEvaluator(UniformAllocation(TH), thresholds=TH).run(
                tasks, run_fn, budget, warmup=1, seed=0
            )
            assert tr.n_executions <= budget

    def test_warmup_is_uniform(self):
        tasks, run_fn, _ = self._world()
        tr = AdaptiveEvaluator(GreedyValuePolicy(thresholds=TH), thresholds=TH).run(
            tasks, run_fn, len(tasks) * 2, warmup=2, seed=0
        )
        assert {s.n_runs for s in tr.states.values()} == {2}

    def test_stopping_can_leave_budget_unspent(self):
        tasks, run_fn, _ = self._world()
        tr = AdaptiveEvaluator(UniformAllocation(TH), thresholds=TH).run(
            tasks, run_fn, 10_000, warmup=2, seed=0
        )
        assert tr.n_executions < 10_000, "stopping rules must actually save runs"
        assert tr.ledger.summary()["n_stopped"] > 0

    def test_trace_reports_labels_and_a_frame(self):
        tasks, run_fn, _ = self._world()
        tr = AdaptiveEvaluator(thresholds=TH).run(tasks, run_fn, 200, warmup=2, seed=0)
        assert set(tr.labels(TH)) == set(tasks)
        df = tr.to_frame()
        assert len(df) == len(tasks)
        assert {"n_runs", "ci_width", "stopped", "stop_reason"} <= set(df.columns)
        assert tr.summary()["n_executions"] == tr.n_executions

    def test_zero_budget_runs_nothing(self):
        tasks, run_fn, _ = self._world()
        tr = AdaptiveEvaluator(thresholds=TH).run(tasks, run_fn, 0, seed=0)
        assert tr.n_executions == 0

    def test_compare_allocation_against_a_reference(self):
        tasks, run_fn, true_p = self._world()
        ref = {t: ("SOLID" if p > 0.9 else "STUCK" if p < 0.1 else "RECOVERABLE")
               for t, p in true_p.items()}
        df = compare_allocation(tasks, run_fn, ref, budget=len(tasks) * 6,
                                warmup=2, thresholds=TH, seed=0)
        assert {"policy", "executions", "agreement", "recoverable_recall"} <= set(df.columns)
        assert (df["executions"] <= len(tasks) * 6).all()
        assert df["agreement"].between(0, 1).all()
