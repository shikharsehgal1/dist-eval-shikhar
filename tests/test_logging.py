"""Tests for disteval.logging — cycle observability."""
import json
import os
import tempfile

import pytest

from disteval.logging import CycleLogger


class TestCycleLogger:
    def test_full_cycle_roundtrip(self):
        logger = CycleLogger(agent_name="agent-A", model_name="m")
        logger.log_cycle_start(1, n_tasks=10, kappa=0.65)
        logger.log_taxonomy(n_solid=5, n_recoverable=3, n_stuck=2,
                            recoverable_score_left=0.45)
        logger.log_task_improvement(
            "task-1", "recoverable", gap=0.3, priority_score=0.15,
            difficulty="hard", n_training_pairs=2, divergence_step=5,
            predicted_gain=0.12,
        )
        logger.log_cycle_end(1, kappa_new=0.72, delta_kappa=0.07,
                            plateau_detected=False, predicted_total_gain=0.25,
                            cycle_complete=False, recursion_enabled=True,
                            n_decomposed=4)

        data = logger.to_dict()
        assert data["agent_name"] == "agent-A"
        assert data["n_cycles"] == 1
        assert data["cycles"][0]["consistency_index"] == pytest.approx(0.72)
        assert data["cycles"][0]["tasks"][0]["task"] == "task-1"

    def test_export_json(self):
        logger = CycleLogger()
        logger.log_cycle_start(1, n_tasks=5, kappa=0.5)
        logger.log_cycle_end(1, kappa_new=0.5, delta_kappa=0.0,
                            plateau_detected=True)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            logger.export_json(path)
            assert os.path.exists(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["n_cycles"] == 1
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_export_csv(self):
        logger = CycleLogger()
        logger.log_cycle_start(1, n_tasks=5, kappa=0.5)
        logger.log_task_improvement("t1", "recoverable", gap=0.2, priority_score=0.1)
        logger.log_cycle_end(1, kappa_new=0.55, delta_kappa=0.05,
                            plateau_detected=False)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            logger.export_csv(path)
            assert os.path.exists(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2  # header + one task row
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_export_csv_empty_is_noop(self):
        logger = CycleLogger()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            logger.export_csv(path)
            assert os.path.getsize(path) == 0
        finally:
            if os.path.exists(path):
                os.unlink(path)
