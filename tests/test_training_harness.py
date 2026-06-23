"""Tests for disteval.training_harness."""
import os
import tempfile

import pytest

from disteval.training_harness import (
    NoOpTrainer,
    SimulatedTrainer,
    TRLReferenceTrainer,
    AxolotlReferenceTrainer,
    run_training,
)


def make_plan():
    return {
        "agent_name": "a",
        "model_name": "m",
        "cycle": 1,
        "curriculum": [
            {
                "task": "task-1",
                "current_q_star": 0.8,
                "predicted_gain": 0.1,
                "training_pairs": [
                    {
                        "reinforce_traj_path": "/tmp/reinforce.json",
                        "contrast_traj_path": "/tmp/contrast.json",
                        "reinforce_score": 0.8,
                        "contrast_score": 0.4,
                    }
                ],
            }
        ],
    }


class TestNoOpTrainer:
    def test_writes_dataset_and_returns_scores(self):
        with tempfile.TemporaryDirectory() as d:
            scores = NoOpTrainer(improved_score=0.99).train(make_plan(), d)
            assert scores == {"task-1": 0.99}
            assert os.path.exists(os.path.join(d, "dpo_dataset.jsonl"))


class TestSimulatedTrainer:
    def test_returns_capped_scores(self):
        scores = SimulatedTrainer().train(make_plan(), "out")
        assert scores["task-1"] == pytest.approx(0.9)


class TestAxolotlReferenceTrainer:
    def test_writes_config(self):
        with tempfile.TemporaryDirectory() as d:
            scores = AxolotlReferenceTrainer("base/model").train(make_plan(), d)
            assert "task-1" in scores
            assert os.path.exists(os.path.join(d, "dpo_dataset.jsonl"))
            assert os.path.exists(os.path.join(d, "axolotl_config.yaml"))


class TestTRLReferenceTrainer:
    def test_raises_without_trl(self):
        trainer = TRLReferenceTrainer("model")
        with pytest.raises(ImportError, match="trl"):
            trainer.train(make_plan(), "out")


class TestRunTraining:
    def test_entrypoint(self):
        with tempfile.TemporaryDirectory() as d:
            scores = run_training(make_plan(), NoOpTrainer(improved_score=1.0), d)
            assert scores == {"task-1": 1.0}
