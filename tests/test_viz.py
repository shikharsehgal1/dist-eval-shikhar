"""Tests for disteval.viz – plot generation."""
import os
import tempfile

import pytest

from disteval.records import EpisodeRecord, RecordStore
from disteval.viz import (
    plot_performance_profile,
    plot_empirical_cdf,
    generate_all_plots,
)


@pytest.fixture()
def store():
    s = RecordStore()
    for task, scores in {
        "t_easy": [0.9, 0.95, 0.85],
        "t_medium": [0.6, 0.5, 0.7],
        "t_hard": [0.2, 0.1, 0.3],
    }.items():
        diff = task.split("_")[1]
        for i, score in enumerate(scores):
            s.add(EpisodeRecord(
                run_id="r0", model="m", task=task, episode=i,
                score=score, success=score >= 0.99,
                strata={"difficulty": diff},
            ))
    return s


class TestPlotPerformanceProfile:
    def test_creates_file(self, store):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_performance_profile(store, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


class TestPlotEmpiricalCDF:
    def test_creates_file(self, store):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_empirical_cdf(store, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


class TestGenerateAllPlots:
    def test_generates_expected_files(self, store):
        with tempfile.TemporaryDirectory() as d:
            paths = generate_all_plots(store, d, boot_width=0.05, repeat_width=0.15)
            assert "performance_profile" in paths
            assert "empirical_cdf" in paths
            assert "mean_vs_cvar" in paths
            assert "pass_reliability" in paths
            assert "eval_reliability" in paths
            for p in paths.values():
                assert os.path.exists(p)
