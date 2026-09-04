"""Classification, recoverability estimators, and ranking.

The behavioural contract these lock in: the README's canonical example must come
out as SOLID / RECOVERABLE / STUCK, a single lucky success must NOT produce
RECOVERABLE, and every discrete label must travel with its continuous scores.
"""
import numpy as np
import pytest

from disteval.reliability.classify import (
    RECOVERABLE,
    SOLID,
    STUCK,
    UNCERTAIN,
    ReliabilityThresholds,
    diagnose,
    diagnose_many,
    evidence_weighted_gap,
    expected_headroom,
    posterior_gap,
    rank_by_recoverability,
    tie_diagnostics,
)
from disteval.reliability.posterior import binary_posterior, posterior_from_scores


class TestThresholds:
    def test_ordering_is_enforced(self):
        with pytest.raises(ValueError):
            ReliabilityThresholds(tau_cap=0.9, tau_rel=0.5)
        with pytest.raises(ValueError):
            ReliabilityThresholds(tau_stuck=0.5, tau_cap=0.2)

    def test_confidence_must_exceed_a_half(self):
        for c in (0.4, 0.5, 1.0):
            with pytest.raises(ValueError):
                ReliabilityThresholds(confidence=c)

    def test_min_runs_must_be_positive(self):
        with pytest.raises(ValueError):
            ReliabilityThresholds(min_runs=0)


class TestCanonicalExample:
    """The three run patterns the README uses."""

    def test_task_a_all_success_is_solid(self):
        d = diagnose(posterior_from_scores([1] * 8), scores=[1] * 8)
        assert d.label == SOLID
        assert d.capability > 0.9 and d.reliability > 0.8

    def test_task_b_intermittent_is_recoverable(self):
        s = [1, 0, 1, 0, 0, 1, 0, 0]
        d = diagnose(posterior_from_scores(s), scores=s)
        assert d.label == RECOVERABLE
        assert d.capability > 0.8
        assert d.reliability < 0.2

    def test_task_c_all_failure_is_stuck(self):
        d = diagnose(posterior_from_scores([0] * 8), scores=[0] * 8)
        assert d.label == STUCK
        assert d.stuck_evidence > 0.8

    def test_one_lucky_success_in_two_is_not_recoverable(self):
        """The explicit requirement: do not claim recoverability from one run."""
        d = diagnose(posterior_from_scores([1, 0]), scores=[1, 0])
        assert d.label == UNCERTAIN

    def test_single_run_is_uncertain_either_way(self):
        assert diagnose(posterior_from_scores([1]), scores=[1]).label == UNCERTAIN
        assert diagnose(posterior_from_scores([0]), scores=[0]).label == UNCERTAIN


class TestLabelsCarryUncertainty:
    def test_every_diagnosis_reports_continuous_scores(self):
        d = diagnose(posterior_from_scores([1, 0, 1, 0]), scores=[1, 0, 1, 0])
        for f in ("capability", "reliability", "stuck_evidence", "posterior_mean",
                  "posterior_sd", "ci_lo", "ci_hi", "posterior_entropy"):
            assert np.isfinite(getattr(d, f))

    def test_observed_max_is_kept_but_separate_from_the_estimate(self):
        s = [0, 0, 0, 1]
        d = diagnose(posterior_from_scores(s), scores=s)
        assert d.observed_max == 1.0
        assert d.posterior_mean < 0.6, "the max must not become the estimate"

    def test_ci_contains_the_posterior_mean(self):
        for s in ([1] * 8, [0] * 8, [1, 0] * 4, [1, 1, 1, 0]):
            d = diagnose(posterior_from_scores(s), scores=s)
            assert d.ci_lo <= d.posterior_mean <= d.ci_hi

    def test_more_runs_narrow_the_interval(self):
        few = diagnose(posterior_from_scores([1, 0, 1]), scores=[1, 0, 1])
        many = diagnose(posterior_from_scores([1, 0] * 16), scores=[1, 0] * 16)
        assert many.ci_width < few.ci_width


class TestRecoverabilityEstimators:
    def test_posterior_gap_bounds_and_monotonicity(self):
        assert posterior_gap(1.0, 0.0) == 1.0
        assert posterior_gap(1.0, 1.0) == 0.0
        assert posterior_gap(0.0, 0.0) == 0.0
        assert posterior_gap(0.9, 0.1) > posterior_gap(0.5, 0.1)

    def test_headroom_is_low_for_solid_and_for_stuck(self):
        th = ReliabilityThresholds()
        solid = expected_headroom(binary_posterior(30, 30), th)
        stuck = expected_headroom(binary_posterior(0, 30), th)
        mid = expected_headroom(binary_posterior(12, 30), th)
        assert mid > solid
        assert mid > stuck

    def test_headroom_is_in_range(self):
        th = ReliabilityThresholds()
        for s, n in [(0, 8), (3, 8), (8, 8), (1, 2), (0, 1)]:
            v = expected_headroom(binary_posterior(s, n), th)
            assert 0.0 <= v <= 1.0

    def test_evidence_weighting_discounts_low_sample_tasks(self):
        low = evidence_weighted_gap(0.9, 0.05, n_obs=2)
        high = evidence_weighted_gap(0.9, 0.05, n_obs=32)
        assert high > low
        assert evidence_weighted_gap(0.9, 0.05, n_obs=0) == 0.0

    def test_well_evidenced_task_outranks_a_lucky_one(self):
        """The brief's requirement, on the estimator that claims to fix it."""
        lucky = diagnose(posterior_from_scores([1, 0]), scores=[1, 0])
        solid = diagnose(posterior_from_scores([1, 1, 1, 1, 0, 0, 0, 0]),
                         scores=[1, 1, 1, 1, 0, 0, 0, 0])
        assert solid.recoverability_evidence > lucky.recoverability_evidence
        assert solid.recoverability_headroom > lucky.recoverability_headroom

    def test_unknown_estimator_raises(self):
        with pytest.raises(ValueError):
            diagnose(posterior_from_scores([1, 0]), primary="nonsense")

    def test_primary_selects_the_reported_estimator(self):
        s = [1, 0, 1, 0]
        for name, field in [("gap", "recoverability_gap"),
                            ("headroom", "recoverability_headroom"),
                            ("evidence", "recoverability_evidence")]:
            d = diagnose(posterior_from_scores(s), scores=s, primary=name)
            assert d.recoverability == pytest.approx(getattr(d, field))


class TestRanking:
    def _diags(self):
        patterns = {
            "solid": [1] * 8, "stuck": [0] * 8, "half": [1, 0] * 4,
            "mostly": [1, 1, 1, 1, 1, 1, 0, 0], "rare": [1, 0, 0, 0, 0, 0, 0, 0],
        }
        return [diagnose(posterior_from_scores(v), task=k, scores=v)
                for k, v in patterns.items()]

    def test_ranking_is_descending(self):
        r = rank_by_recoverability(self._diags(), labels=None)
        vals = [d.recoverability_headroom for d in r]
        assert vals == sorted(vals, reverse=True)

    def test_ranking_is_deterministic_under_input_order(self):
        d = self._diags()
        a = [x.task for x in rank_by_recoverability(d, labels=None)]
        b = [x.task for x in rank_by_recoverability(list(reversed(d)), labels=None)]
        assert a == b

    def test_label_filter_restricts_the_pool(self):
        r = rank_by_recoverability(self._diags(), labels=(RECOVERABLE,))
        assert all(d.label == RECOVERABLE for d in r)

    def test_tie_diagnostics_detects_the_structural_tie(self):
        """With n binary runs there are at most n+1 distinct scores."""
        diags = [
            diagnose(binary_posterior(i % 9, 8), task=f"t{i}") for i in range(60)
        ]
        t = tie_diagnostics(diags, top_n=20)
        assert t["n_distinct_scores"] <= 9
        assert t["largest_tier_size"] > 1
        assert t["cutoff_is_arbitrary"] is True
        assert t["note"]

    def test_tie_diagnostics_on_empty_input(self):
        assert tie_diagnostics([])["n_tasks"] == 0


class TestDiagnoseMany:
    def test_keys_are_model_task_pairs(self):
        posts = {("m", "a"): binary_posterior(4, 8), ("m", "b"): binary_posterior(1, 8)}
        out = diagnose_many(posts, domains={"a": "d1", "b": "d2"})
        assert {(d.model, d.task) for d in out} == set(posts)
        assert {d.domain for d in out} == {"d1", "d2"}

    def test_unequal_run_counts_are_fine(self):
        posts = {("m", "a"): binary_posterior(2, 3), ("m", "b"): binary_posterior(9, 20)}
        out = diagnose_many(posts)
        assert {d.n_runs for d in out} == {3, 20}

    def test_to_dict_is_flat_and_serialisable(self):
        d = diagnose(posterior_from_scores([1, 0, 1]), task="t").to_dict()
        assert d["task"] == "t"
        assert isinstance(d["label"], str)


class TestConfigurability:
    def test_raising_tau_rel_makes_solid_harder(self):
        s = [1] * 10
        lenient = diagnose(posterior_from_scores(s), scores=s,
                           thresholds=ReliabilityThresholds(tau_rel=0.6))
        strict = diagnose(posterior_from_scores(s), scores=s,
                          thresholds=ReliabilityThresholds(tau_rel=0.99))
        assert lenient.reliability > strict.reliability
        assert lenient.label == SOLID and strict.label != SOLID

    def test_raising_confidence_produces_more_uncertain(self):
        s = [1, 0, 1, 0, 0, 1, 0, 0]
        loose = diagnose(posterior_from_scores(s), scores=s,
                         thresholds=ReliabilityThresholds(confidence=0.55))
        tight = diagnose(posterior_from_scores(s), scores=s,
                         thresholds=ReliabilityThresholds(confidence=0.99))
        assert loose.label == RECOVERABLE
        assert tight.label == UNCERTAIN

    def test_min_runs_gate(self):
        s = [1, 0, 1]
        assert diagnose(posterior_from_scores(s), scores=s,
                        thresholds=ReliabilityThresholds(min_runs=5)).label == UNCERTAIN
