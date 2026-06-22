"""Tests for disteval.test_suite_parser."""
from __future__ import annotations

import pytest

from disteval.test_suite_parser import (
    CheckpointSpec,
    checkpoint_weights,
    parse_all_tasks,
    parse_test_suite,
)


def test_parse_medium_2():
    specs = parse_test_suite("tasks/medium-2/tests/test.sh")
    assert len(specs) == 5
    assert [s.score_increment for s in specs] == [10, 25, 25, 20, 20]
    assert [s.reward_weight for s in specs] == [0.10, 0.25, 0.25, 0.20, 0.20]
    assert sum(s.reward_weight for s in specs) == pytest.approx(1.0)
    assert specs[0].task_name == "medium-2"
    assert specs[0].checkpoint_id == "medium-2::phase-0"
    assert specs[2].checkpoint_id == "medium-2::phase-2"
    assert "total_score" in specs[1].description.lower() or "eligible" in specs[1].description.lower()


def test_parse_easy_1():
    specs = parse_test_suite("tasks/easy-1/tests/test.sh")
    assert len(specs) == 3
    assert [s.score_increment for s in specs] == [34, 33, 33]
    assert sum(s.reward_weight for s in specs) == pytest.approx(1.0)


def test_parse_easy_2():
    specs = parse_test_suite("tasks/easy-2/tests/test.sh")
    assert len(specs) == 4
    assert [s.score_increment for s in specs] == [25, 25, 25, 25]
    assert [s.reward_weight for s in specs] == [0.25, 0.25, 0.25, 0.25]


def test_parse_hard_1():
    specs = parse_test_suite("tasks/hard-1/tests/test.sh")
    assert len(specs) == 6
    assert [s.score_increment for s in specs] == [10, 15, 20, 20, 20, 15]
    assert sum(s.reward_weight for s in specs) == pytest.approx(1.0)


def test_parse_hard_2():
    specs = parse_test_suite("tasks/hard-2/tests/test.sh")
    assert len(specs) == 3
    assert [s.score_increment for s in specs] == [10, 40, 50]
    assert sum(s.reward_weight for s in specs) == pytest.approx(1.0)


def test_parse_medium_1():
    specs = parse_test_suite("tasks/medium-1/tests/test.sh")
    assert len(specs) == 5
    assert [s.score_increment for s in specs] == [10, 20, 30, 20, 20]
    assert sum(s.reward_weight for s in specs) == pytest.approx(1.0)


def test_parse_all_tasks():
    result = parse_all_tasks("tasks")
    assert "medium-2" in result
    assert "easy-1" in result
    assert "easy-2" in result
    assert "hard-1" in result
    assert "hard-2" in result
    assert "medium-1" in result


def test_checkpoint_weights():
    specs = [
        CheckpointSpec(0, "t", "a", 0.2, 20, 100, "", "t::phase-0"),
        CheckpointSpec(1, "t", "b", 0.8, 80, 100, "", "t::phase-1"),
    ]
    assert checkpoint_weights(specs) == [0.2, 0.8]


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_test_suite("tasks/nonexistent/tests/test.sh")


def test_total_score_derived_from_increments():
    # If a test script has no final print($SCORE / N) but does have increments,
    # total_score should still be derived from the increments.
    specs = parse_test_suite("tasks/medium-2/tests/test.sh")
    assert specs[0].total_score == 100
