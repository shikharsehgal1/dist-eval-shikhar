"""Tests for disteval.metrics – every assertion uses hand-verified expected values."""
import math

import numpy as np
import pandas as pd
import pytest

from disteval.metrics import iqm, cvar, var_at, pass_at_k, pass_hat_k, summarize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_df(task_successes: dict[str, list[bool]], base_score: float = 0.5) -> pd.DataFrame:
    """Build a minimal DataFrame with 'task', 'success', and 'score' columns."""
    rows = []
    for task, successes in task_successes.items():
        for ep, s in enumerate(successes):
            rows.append({"task": task, "episode": ep, "success": s, "score": base_score})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# iqm
# ---------------------------------------------------------------------------

class TestIQM:
    def test_uniform_eight_elements(self):
        # [1,2,3,4,5,6,7,8]: Q1=2.75, Q3=6.25 → mid=[3,4,5,6] → mean=4.5
        x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
        assert iqm(x) == pytest.approx(4.5)

    def test_uniform_ten_elements(self):
        # [0,1,...,9]: Q1=2.25, Q3=6.75 → mid=[3,4,5,6] → mean=4.5
        x = np.arange(10, dtype=float)
        assert iqm(x) == pytest.approx(4.5)

    def test_bimodal_symmetric(self):
        # [1,1,1,1,10,10,10,10]: Q1=1.0, Q3=10.0 → all values in mid → mean=5.5
        x = np.array([1.0, 1, 1, 1, 10, 10, 10, 10])
        assert iqm(x) == pytest.approx(5.5)

    def test_constant_array(self):
        x = np.full(10, 7.0)
        assert iqm(x) == pytest.approx(7.0)

    def test_single_element(self):
        assert iqm(np.array([42.0])) == pytest.approx(42.0)

    def test_empty_returns_nan(self):
        result = iqm(np.array([]))
        assert math.isnan(result)

    def test_iqm_ignores_extreme_outliers(self):
        # Insert extreme outliers; IQM of the middle 50% should be unaffected
        x = np.array([-1000.0, 1, 2, 3, 4, 5, 6, 7, 8, 1000.0])
        # without outliers median and IQM cluster around 4-5
        result = iqm(x)
        assert 3.0 <= result <= 6.0

    def test_unsorted_input_matches_sorted(self):
        x = np.array([5.0, 1, 8, 3, 7, 2, 6, 4])
        assert iqm(x) == pytest.approx(iqm(np.sort(x)))

    def test_list_input_accepted(self):
        assert iqm([1.0, 2.0, 3.0, 4.0]) == pytest.approx(iqm(np.array([1.0, 2, 3, 4])))


# ---------------------------------------------------------------------------
# var_at
# ---------------------------------------------------------------------------

class TestVarAt:
    def test_lower_quantile(self):
        # [0..99]: 5th-percentile (alpha=0.05) = 4.95
        x = np.arange(100, dtype=float)
        assert var_at(x, alpha=0.05, tail="lower") == pytest.approx(4.95)

    def test_upper_quantile(self):
        # [0..99]: upper alpha=0.1 → 90th-percentile = 89.1
        x = np.arange(100, dtype=float)
        assert var_at(x, alpha=0.1, tail="upper") == pytest.approx(89.1)

    def test_lower_default_tail(self):
        # default is tail='lower'
        x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert var_at(x, alpha=0.1) == pytest.approx(var_at(x, alpha=0.1, tail="lower"))

    def test_median_is_50th_percentile(self):
        x = np.arange(1, 12, dtype=float)   # odd-length for exact median
        assert var_at(x, alpha=0.5, tail="lower") == pytest.approx(np.median(x))

    def test_min_is_0th_percentile(self):
        x = np.array([3.0, 1, 4, 1, 5, 9, 2, 6])
        assert var_at(x, alpha=0.0, tail="lower") == pytest.approx(x.min())

    def test_max_is_0th_percentile_upper(self):
        x = np.array([3.0, 1, 4, 1, 5, 9, 2, 6])
        assert var_at(x, alpha=0.0, tail="upper") == pytest.approx(x.max())


# ---------------------------------------------------------------------------
# cvar
# ---------------------------------------------------------------------------

class TestCVaR:
    def test_lower_tail(self):
        # [1..10], alpha=0.2: VaR=2.8, tail=[1,2] → CVaR=1.5
        x = np.arange(1, 11, dtype=float)
        assert cvar(x, alpha=0.2, tail="lower") == pytest.approx(1.5)

    def test_upper_tail(self):
        # [1..10], alpha=0.2: upper VaR=8.2, tail=[9,10] → CVaR=9.5
        x = np.arange(1, 11, dtype=float)
        assert cvar(x, alpha=0.2, tail="upper") == pytest.approx(9.5)

    def test_cvar_leq_var_lower(self):
        # CVaR (mean of worst tail) should be ≤ VaR for lower tail
        x = np.array([0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert cvar(x, alpha=0.1, tail="lower") <= var_at(x, alpha=0.1, tail="lower") + 1e-10

    def test_cvar_geq_var_upper(self):
        # CVaR (mean of best tail) should be ≥ VaR for upper tail
        x = np.array([0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert cvar(x, alpha=0.1, tail="upper") >= var_at(x, alpha=0.1, tail="upper") - 1e-10

    def test_constant_array(self):
        x = np.full(10, 5.0)
        assert cvar(x, alpha=0.1) == pytest.approx(5.0)

    def test_empty_returns_nan(self):
        result = cvar(np.array([]))
        assert math.isnan(result)

    def test_default_tail_is_lower(self):
        x = np.arange(1, 11, dtype=float)
        assert cvar(x, alpha=0.2) == pytest.approx(cvar(x, alpha=0.2, tail="lower"))


# ---------------------------------------------------------------------------
# pass_at_k
# ---------------------------------------------------------------------------

class TestPassAtK:
    def test_all_succeed_k1(self):
        # Single task, all 4 trials succeed → pass@1 = 1.0
        df = _task_df({"t1": [True, True, True, True]})
        assert pass_at_k(df, k=1) == pytest.approx(1.0)

    def test_none_succeed_k1(self):
        # No successes → pass@k = 0.0
        df = _task_df({"t1": [False, False, False, False]})
        assert pass_at_k(df, k=1) == pytest.approx(0.0)

    def test_partial_success_k2(self):
        # task_A: (n=4, c=4), task_B: (n=4, c=2)
        # task_A pass@2 = 1.0
        # task_B pass@2 = 1 - C(2,2)/C(4,2) = 1 - 1/6 ≈ 0.8333
        # average = 0.9166...
        df = _task_df({"t_A": [True]*4, "t_B": [True, True, False, False]})
        assert pass_at_k(df, k=2) == pytest.approx((1.0 + 5/6) / 2, rel=1e-6)

    def test_two_tasks_k1_average(self):
        # task_A: 4/4, task_B: 2/4
        # pass@1 task_A = 1.0, task_B = 1 - C(2,1)/C(4,1) = 0.5  → avg=0.75
        df = _task_df({"t_A": [True]*4, "t_B": [True, True, False, False]})
        assert pass_at_k(df, k=1) == pytest.approx(0.75)

    def test_fallback_when_n_lt_k(self):
        # When n < k the function falls back to float(c > 0)
        # task with 1 trial that succeeds → pass@2 (fallback) = 1.0
        df = _task_df({"t1": [True]})
        assert pass_at_k(df, k=2) == pytest.approx(1.0)

    def test_fallback_fail(self):
        # 1 trial, no success, k=2 → fallback = 0.0
        df = _task_df({"t1": [False]})
        assert pass_at_k(df, k=2) == pytest.approx(0.0)

    def test_pass_at_k_increases_with_k(self):
        # More tries can't decrease pass@k when there is any chance of success
        df = _task_df({"t1": [True, False, True, False, False]})
        assert pass_at_k(df, k=1) <= pass_at_k(df, k=2) + 1e-9


# ---------------------------------------------------------------------------
# pass_hat_k (pass^k)
# ---------------------------------------------------------------------------

class TestPassHatK:
    def test_all_succeed_k1(self):
        df = _task_df({"t1": [True, True, True, True]})
        assert pass_hat_k(df, k=1) == pytest.approx(1.0)

    def test_none_succeed_k1(self):
        df = _task_df({"t1": [False, False, False, False]})
        assert pass_hat_k(df, k=1) == pytest.approx(0.0)

    def test_partial_success_k2(self):
        # task_A: (n=4, c=4), pass^2 = C(4,2)/C(4,2) = 1.0
        # task_B: (n=4, c=2), pass^2 = C(2,2)/C(4,2) = 1/6
        # average = (1.0 + 1/6)/2 = 7/12
        df = _task_df({"t_A": [True]*4, "t_B": [True, True, False, False]})
        assert pass_hat_k(df, k=2) == pytest.approx(7 / 12, rel=1e-6)

    def test_two_tasks_k1_average(self):
        # task_A: 4/4, task_B: 2/4
        # pass^1 task_A = C(4,1)/C(4,1)=1.0, task_B = C(2,1)/C(4,1)=0.5 → avg=0.75
        df = _task_df({"t_A": [True]*4, "t_B": [True, True, False, False]})
        assert pass_hat_k(df, k=1) == pytest.approx(0.75)

    def test_reliability_gap_exists(self):
        # pass@k >= pass^k always; the gap captures reliability
        df = _task_df({"t1": [True, True, False, False, False]})
        assert pass_at_k(df, k=2) >= pass_hat_k(df, k=2) - 1e-9

    def test_fallback_all_succeed(self):
        # n < k, all trials succeed → pass^k fallback = 1.0
        df = _task_df({"t1": [True]})
        assert pass_hat_k(df, k=2) == pytest.approx(1.0)

    def test_fallback_partial_success(self):
        # n < k, not all succeed → fallback = 0.0
        df = _task_df({"t1": [True, False]})
        assert pass_hat_k(df, k=3) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

class TestSummarize:
    def _make_df(self):
        return pd.DataFrame({
            "task": ["t1"] * 4,
            "score": [1.0, 2.0, 3.0, 4.0],
            "success": [True, True, False, False],
        })

    def test_keys_present(self):
        df = self._make_df()
        out = summarize(df)
        required_keys = {"n_episodes", "mean", "iqm", "median", "std", "success_rate"}
        assert required_keys.issubset(out.keys())

    def test_n_episodes(self):
        df = self._make_df()
        assert summarize(df)["n_episodes"] == 4

    def test_mean(self):
        df = self._make_df()
        assert summarize(df)["mean"] == pytest.approx(2.5)

    def test_success_rate(self):
        df = self._make_df()
        assert summarize(df)["success_rate"] == pytest.approx(0.5)

    def test_ks_keys_present(self):
        df = self._make_df()
        out = summarize(df, ks=(1,))
        assert "pass@1" in out
        assert "pass^1" in out
