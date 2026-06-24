"""Tests for disteval.failure – failure_distribution, criterion_failure_rates."""
import pandas as pd
import numpy as np
import pytest

from disteval.failure import failure_distribution, criterion_failure_rates, top_failing_criteria


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows):
    """Build a DataFrame from a list of dicts; add default 'score' column if missing."""
    df = pd.DataFrame(rows)
    if "score" not in df.columns:
        df["score"] = 0.0
    return df


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

class TestFailureDistributionBasic:
    def test_all_success_returns_empty(self):
        df = _make_df([
            {"success": True, "failure_mode": None},
            {"success": True, "failure_mode": None},
        ])
        result = failure_distribution(df)
        assert result.empty

    def test_empty_result_has_correct_columns(self):
        df = _make_df([{"success": True, "failure_mode": None}])
        result = failure_distribution(df)
        assert "failure_mode" in result.columns
        assert "n" in result.columns
        assert "share_of_failures" in result.columns

    def test_empty_result_with_by_has_by_columns(self):
        df = _make_df([{"success": True, "failure_mode": None, "s_diff": "easy"}])
        result = failure_distribution(df, by=["s_diff"])
        assert "s_diff" in result.columns

    def test_single_failure_mode(self):
        df = _make_df([
            {"success": False, "failure_mode": "timeout"},
            {"success": False, "failure_mode": "timeout"},
            {"success": True,  "failure_mode": None},
        ])
        result = failure_distribution(df)
        assert len(result) == 1
        assert result.iloc[0]["failure_mode"] == "timeout"
        assert result.iloc[0]["n"] == 2
        assert result.iloc[0]["share_of_failures"] == pytest.approx(1.0)

    def test_two_equal_failure_modes(self):
        df = _make_df([
            {"success": False, "failure_mode": "timeout"},
            {"success": False, "failure_mode": "timeout"},
            {"success": False, "failure_mode": "crash"},
            {"success": False, "failure_mode": "crash"},
            {"success": True,  "failure_mode": None},
            {"success": True,  "failure_mode": None},
        ])
        result = failure_distribution(df)
        assert len(result) == 2
        # Both failure modes appear equally often (2 each out of 4 failures → 0.5)
        for share in result["share_of_failures"]:
            assert share == pytest.approx(0.5)

    def test_shares_sum_to_one(self):
        df = _make_df([
            {"success": False, "failure_mode": "timeout"},
            {"success": False, "failure_mode": "crash"},
            {"success": False, "failure_mode": "hallucination"},
        ])
        result = failure_distribution(df)
        assert result["share_of_failures"].sum() == pytest.approx(1.0)

    def test_n_column_counts_failures(self):
        df = _make_df([
            {"success": False, "failure_mode": "timeout"},
            {"success": False, "failure_mode": "timeout"},
            {"success": False, "failure_mode": "timeout"},
            {"success": False, "failure_mode": "crash"},
        ])
        result = failure_distribution(df)
        timeout_row = result[result["failure_mode"] == "timeout"]
        crash_row   = result[result["failure_mode"] == "crash"]
        assert timeout_row["n"].iloc[0] == 3
        assert crash_row["n"].iloc[0] == 1

    def test_missing_success_column_raises(self):
        df = pd.DataFrame({"failure_mode": ["timeout", "crash"]})
        with pytest.raises(ValueError, match="'success' column"):
            failure_distribution(df)


# ---------------------------------------------------------------------------
# Unlabeled / NaN failure modes
# ---------------------------------------------------------------------------

class TestFailureDistributionUnlabeled:
    def test_none_failure_mode_becomes_unlabeled(self):
        df = _make_df([
            {"success": False, "failure_mode": None},
            {"success": False, "failure_mode": None},
        ])
        result = failure_distribution(df)
        assert len(result) == 1
        assert result.iloc[0]["failure_mode"] == "unlabeled"

    def test_mixed_labeled_and_unlabeled(self):
        df = _make_df([
            {"success": False, "failure_mode": "timeout"},
            {"success": False, "failure_mode": None},
        ])
        result = failure_distribution(df)
        modes = set(result["failure_mode"].tolist())
        assert "timeout" in modes
        assert "unlabeled" in modes
        assert result["share_of_failures"].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Stratified (by= parameter)
# ---------------------------------------------------------------------------

class TestFailureDistributionStratified:
    def _stratified_df(self):
        """Three failures: 2 easy-timeout, 1 hard-crash; 2 successes."""
        return _make_df([
            {"success": False, "failure_mode": "timeout",  "s_difficulty": "easy"},
            {"success": False, "failure_mode": "timeout",  "s_difficulty": "easy"},
            {"success": False, "failure_mode": "crash",    "s_difficulty": "hard"},
            {"success": True,  "failure_mode": None,       "s_difficulty": "easy"},
            {"success": True,  "failure_mode": None,       "s_difficulty": "hard"},
        ])

    def test_by_stratum_creates_stratum_column(self):
        result = failure_distribution(self._stratified_df(), by=["s_difficulty"])
        assert "s_difficulty" in result.columns

    def test_by_stratum_correct_rows(self):
        result = failure_distribution(self._stratified_df(), by=["s_difficulty"])
        # easy → 2 timeout; hard → 1 crash
        assert len(result) == 2

    def test_by_stratum_shares_per_group_sum_to_one(self):
        df = _make_df([
            {"success": False, "failure_mode": "timeout",      "group": "A"},
            {"success": False, "failure_mode": "crash",        "group": "A"},
            {"success": False, "failure_mode": "hallucination","group": "B"},
            {"success": False, "failure_mode": "crash",        "group": "B"},
        ])
        result = failure_distribution(df, by=["group"])
        for grp, sub in result.groupby("group"):
            assert sub["share_of_failures"].sum() == pytest.approx(1.0), (
                f"Shares for group {grp} do not sum to 1: {sub}"
            )

    def test_by_stratum_easy_timeout_share_is_one(self):
        result = failure_distribution(self._stratified_df(), by=["s_difficulty"])
        easy_row = result[result["s_difficulty"] == "easy"]
        assert easy_row["share_of_failures"].iloc[0] == pytest.approx(1.0)

    def test_by_stratum_hard_crash_share_is_one(self):
        result = failure_distribution(self._stratified_df(), by=["s_difficulty"])
        hard_row = result[result["s_difficulty"] == "hard"]
        assert hard_row["share_of_failures"].iloc[0] == pytest.approx(1.0)

    def test_global_shares_without_by(self):
        """Without 'by', shares are global fractions of total failures."""
        df = _make_df([
            {"success": False, "failure_mode": "timeout"},
            {"success": False, "failure_mode": "timeout"},
            {"success": False, "failure_mode": "crash"},
        ])
        result = failure_distribution(df)
        timeout_row = result[result["failure_mode"] == "timeout"]
        crash_row   = result[result["failure_mode"] == "crash"]
        assert timeout_row["share_of_failures"].iloc[0] == pytest.approx(2 / 3)
        assert crash_row["share_of_failures"].iloc[0]   == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestFailureDistributionReturnType:
    def test_returns_dataframe(self):
        df = _make_df([{"success": False, "failure_mode": "timeout"}])
        assert isinstance(failure_distribution(df), pd.DataFrame)

    def test_n_column_is_integer_type(self):
        df = _make_df([{"success": False, "failure_mode": "timeout"}])
        result = failure_distribution(df)
        assert result["n"].dtype in (np.int64, np.int32, int)

    def test_share_column_is_float_type(self):
        df = _make_df([{"success": False, "failure_mode": "timeout"}])
        result = failure_distribution(df)
        assert result["share_of_failures"].dtype == float

    def test_sorted_by_n_descending(self):
        df = _make_df([
            {"success": False, "failure_mode": "timeout"},
            {"success": False, "failure_mode": "timeout"},
            {"success": False, "failure_mode": "timeout"},
            {"success": False, "failure_mode": "crash"},
        ])
        result = failure_distribution(df)
        # Most frequent failure mode should come first (sort descending by n)
        assert result.iloc[0]["failure_mode"] == "timeout"
        assert result.iloc[1]["failure_mode"] == "crash"


# ---------------------------------------------------------------------------
# criterion_failure_rates
# ---------------------------------------------------------------------------

class TestCriterionFailureRates:
    def _episodes(self):
        return [
            {"criteria": {"cost_ok": True,  "format_ok": False, "output_correct": True},  "difficulty": "easy"},
            {"criteria": {"cost_ok": False, "format_ok": False, "output_correct": False}, "difficulty": "hard"},
            {"criteria": {"cost_ok": True,  "format_ok": True,  "output_correct": True},  "difficulty": "easy"},
            {"criteria": {"cost_ok": False, "format_ok": False, "output_correct": True},  "difficulty": "hard"},
        ]

    def test_empty_returns_empty_df(self):
        df = criterion_failure_rates([])
        assert df.empty
        assert "failure_rate" in df.columns

    def test_basic_failure_rates(self):
        eps = self._episodes()
        df = criterion_failure_rates(eps)
        assert set(df.columns) == {"criterion", "n_episodes", "n_failed", "failure_rate"}
        # format_ok failed 3/4 times → highest failure rate
        top = df.iloc[0]
        assert top["criterion"] == "format_ok"
        assert top["failure_rate"] == pytest.approx(0.75)

    def test_failure_rate_sorted_descending(self):
        eps = self._episodes()
        df = criterion_failure_rates(eps)
        rates = df["failure_rate"].tolist()
        assert rates == sorted(rates, reverse=True)

    def test_stratified_by_difficulty(self):
        eps = self._episodes()
        df = criterion_failure_rates(eps, by=["difficulty"])
        assert "difficulty" in df.columns
        # Each stratum is independent
        for diff in ["easy", "hard"]:
            sub = df[df["difficulty"] == diff]
            assert not sub.empty

    def test_episode_without_criteria_is_skipped(self):
        eps = [
            {"criteria": {"cost_ok": False}},
            {"difficulty": "hard"},           # no criteria key
            {"criteria": {}},                  # empty criteria
        ]
        df = criterion_failure_rates(eps)
        assert len(df) == 1
        assert df.iloc[0]["criterion"] == "cost_ok"

    def test_all_pass_gives_zero_failure_rate(self):
        eps = [{"criteria": {"x": True, "y": True}} for _ in range(5)]
        df = criterion_failure_rates(eps)
        assert (df["failure_rate"] == 0.0).all()

    def test_all_fail_gives_one_failure_rate(self):
        eps = [{"criteria": {"x": False}} for _ in range(3)]
        df = criterion_failure_rates(eps)
        assert df.iloc[0]["failure_rate"] == pytest.approx(1.0)

    def test_n_episodes_correct(self):
        eps = [
            {"criteria": {"a": True}},
            {"criteria": {"a": False}},
            {"criteria": {"a": True}},
        ]
        df = criterion_failure_rates(eps)
        assert df.iloc[0]["n_episodes"] == 3
        assert df.iloc[0]["n_failed"] == 1


class TestTopFailingCriteria:
    def test_returns_top_n(self):
        eps = [
            {"criteria": {"a": False, "b": True,  "c": False, "d": False, "e": False, "f": True}},
            {"criteria": {"a": False, "b": False, "c": False, "d": True,  "e": False, "f": True}},
        ]
        df = top_failing_criteria(eps, n=3)
        assert len(df) == 3
        # all returned criteria have failure_rate >= those not returned
        all_df = criterion_failure_rates(eps)
        assert df["failure_rate"].min() >= all_df.iloc[3]["failure_rate"]

    def test_n_larger_than_criteria_returns_all(self):
        eps = [{"criteria": {"x": False, "y": True}}]
        df = top_failing_criteria(eps, n=10)
        assert len(df) == 2  # only 2 criteria exist
