"""Tests for disteval.compare – wasserstein, ks, prob_improvement, stochastic_dominance."""
import numpy as np
import pytest

from disteval.compare import wasserstein, ks, prob_improvement, stochastic_dominance


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def identical_arrays():
    return np.array([1.0, 2, 3, 4, 5])


@pytest.fixture()
def low_array():
    return np.array([1.0, 2, 3, 4, 5])


@pytest.fixture()
def high_array():
    return np.array([10.0, 20, 30, 40, 50])


# ---------------------------------------------------------------------------
# wasserstein
# ---------------------------------------------------------------------------

class TestWasserstein:
    def test_identical_arrays_is_zero(self, identical_arrays):
        assert wasserstein(identical_arrays, identical_arrays) == pytest.approx(0.0)

    def test_symmetry(self, low_array, high_array):
        assert wasserstein(low_array, high_array) == pytest.approx(
            wasserstein(high_array, low_array)
        )

    def test_returns_float(self, low_array, high_array):
        result = wasserstein(low_array, high_array)
        assert isinstance(result, float)

    def test_non_negative(self, low_array, high_array):
        assert wasserstein(low_array, high_array) >= 0.0

    def test_shifted_distribution(self):
        # Wasserstein distance between [0,1,2,3,4] and [1,2,3,4,5] should be 1.0
        a = np.arange(5, dtype=float)
        b = np.arange(1, 6, dtype=float)
        assert wasserstein(a, b) == pytest.approx(1.0)

    def test_large_shift_larger_distance(self, low_array, high_array):
        # high_array is much further from low_array than a 1-unit shift
        assert wasserstein(low_array, high_array) > wasserstein(low_array, low_array + 1)

    def test_list_input_accepted(self):
        assert wasserstein([1.0, 2, 3], [1.0, 2, 3]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ks
# ---------------------------------------------------------------------------

class TestKS:
    def test_identical_arrays_D_is_zero(self, identical_arrays):
        result = ks(identical_arrays, identical_arrays)
        assert result["D"] == pytest.approx(0.0)

    def test_identical_arrays_p_is_one(self, identical_arrays):
        result = ks(identical_arrays, identical_arrays)
        assert result["p"] == pytest.approx(1.0)

    def test_returns_dict_with_D_and_p(self, low_array, high_array):
        result = ks(low_array, high_array)
        assert "D" in result
        assert "p" in result

    def test_completely_separated_D_is_one(self, low_array, high_array):
        """[1-5] vs [10-50]: CDFs never overlap → D = 1.0."""
        result = ks(low_array, high_array)
        assert result["D"] == pytest.approx(1.0)

    def test_completely_separated_p_is_small(self, low_array, high_array):
        result = ks(low_array, high_array)
        assert result["p"] < 0.05

    def test_D_is_float(self, low_array, high_array):
        result = ks(low_array, high_array)
        assert isinstance(result["D"], float)
        assert isinstance(result["p"], float)

    def test_D_in_zero_one(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 30)
        b = rng.normal(1, 1, 30)
        result = ks(a, b)
        assert 0.0 <= result["D"] <= 1.0
        assert 0.0 <= result["p"] <= 1.0

    def test_symmetry(self, low_array, high_array):
        assert ks(low_array, high_array)["D"] == pytest.approx(
            ks(high_array, low_array)["D"]
        )


# ---------------------------------------------------------------------------
# prob_improvement
# ---------------------------------------------------------------------------

class TestProbImprovement:
    def test_identical_arrays_is_half(self, identical_arrays):
        """P(A > B) for identical distributions = 0.5 (ties broken at 0.5)."""
        pi = prob_improvement(identical_arrays, identical_arrays)
        assert pi == pytest.approx(0.5)

    def test_a_always_greater_than_b(self, low_array, high_array):
        """high_array always beats low_array → P(A > B) = 1.0."""
        pi = prob_improvement(high_array, low_array)
        assert pi == pytest.approx(1.0)

    def test_b_always_greater_than_a(self, low_array, high_array):
        """low_array always loses to high_array → P(A > B) = 0.0."""
        pi = prob_improvement(low_array, high_array)
        assert pi == pytest.approx(0.0)

    def test_returns_float(self, low_array, high_array):
        assert isinstance(prob_improvement(low_array, high_array), float)

    def test_bounded_zero_one(self):
        rng = np.random.default_rng(42)
        a = rng.normal(1.0, 0.5, 20)
        b = rng.normal(0.0, 0.5, 20)
        pi = prob_improvement(a, b)
        assert 0.0 <= pi <= 1.0

    def test_complementary(self, low_array, high_array):
        """P(A > B) + P(B > A) should equal 1 when there are no ties (distinct arrays)."""
        pi_ab = prob_improvement(high_array, low_array)
        pi_ba = prob_improvement(low_array, high_array)
        assert pi_ab + pi_ba == pytest.approx(1.0)

    def test_partial_overlap(self):
        a = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
        b = np.array([3.0, 4, 5, 6, 7, 8, 9, 10])
        # b is on average higher than a → P(A > B) < 0.5
        assert prob_improvement(a, b) < 0.5


# ---------------------------------------------------------------------------
# stochastic_dominance
# ---------------------------------------------------------------------------

class TestStochasticDominance:
    def test_returns_required_keys(self, low_array, high_array):
        result = stochastic_dominance(low_array, high_array)
        assert "FSD_A_dominates_B" in result
        assert "FSD_B_dominates_A" in result
        assert "SSD_A_dominates_B" in result
        assert "SSD_B_dominates_A" in result

    def test_identical_arrays_both_fsd(self, identical_arrays):
        """Identical distributions: both FSD and both SSD hold (trivially)."""
        result = stochastic_dominance(identical_arrays, identical_arrays)
        assert result["FSD_A_dominates_B"] is True
        assert result["FSD_B_dominates_A"] is True
        assert result["SSD_A_dominates_B"] is True
        assert result["SSD_B_dominates_A"] is True

    def test_a_fsd_over_b_when_a_always_higher(self):
        """A = [5-10], B = [1-6] overlap at 5-6 but A's CDF is always <= B's CDF."""
        A = np.array([5.0, 6, 7, 8, 9, 10])
        B = np.array([1.0, 2, 3, 4, 5, 6])
        result = stochastic_dominance(A, B)
        assert result["FSD_A_dominates_B"] is True
        assert result["FSD_B_dominates_A"] is False

    def test_a_ssd_over_b_when_a_always_higher(self):
        A = np.array([5.0, 6, 7, 8, 9, 10])
        B = np.array([1.0, 2, 3, 4, 5, 6])
        result = stochastic_dominance(A, B)
        assert result["SSD_A_dominates_B"] is True

    def test_b_does_not_fsd_over_a_when_a_dominates(self):
        A = np.array([5.0, 6, 7, 8, 9, 10])
        B = np.array([1.0, 2, 3, 4, 5, 6])
        result = stochastic_dominance(A, B)
        assert result["FSD_B_dominates_A"] is False

    def test_no_dominance_for_crossing_cdfs(self):
        """When CDFs cross neither FSD should hold for each direction."""
        # A is spread low and high; B concentrated in middle
        A = np.array([0.0, 0, 0, 5, 10, 10, 10])
        B = np.array([4.0, 5, 5, 5, 5, 6, 6])
        result = stochastic_dominance(A, B)
        # At least one of the FSD flags must be False (CDFs cross)
        assert not (result["FSD_A_dominates_B"] and result["FSD_B_dominates_A"])

    def test_returns_bool_values(self, low_array, high_array):
        result = stochastic_dominance(low_array, high_array)
        for key in ("FSD_A_dominates_B", "FSD_B_dominates_A",
                    "SSD_A_dominates_B", "SSD_B_dominates_A"):
            assert isinstance(result[key], bool)

    def test_list_input_accepted(self):
        result = stochastic_dominance([1.0, 2, 3], [4.0, 5, 6])
        assert result["FSD_B_dominates_A"] is True

    def test_fsd_implies_ssd(self):
        """If A FSD-dominates B then A should also SSD-dominate B."""
        A = np.array([5.0, 6, 7, 8])
        B = np.array([1.0, 2, 3, 4])
        result = stochastic_dominance(A, B)
        if result["FSD_A_dominates_B"]:
            assert result["SSD_A_dominates_B"] is True
