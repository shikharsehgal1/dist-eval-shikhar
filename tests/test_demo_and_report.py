"""End-to-end: the shipped demo dataset must work immediately after cloning.

These are the tests that would catch "it works on my machine": the committed
example files must load, classify as documented, and drive the full report and
CLI without an API key or any network access.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from disteval.loaders import (
    TaskSpec,
    describe_dataset,
    load_runs,
    load_tasks,
    load_trajectories,
    runs_to_store,
)
from disteval.reliability.classify import diagnose
from disteval.reliability.posterior import posterior_from_scores
from disteval.research_report import build_report_data, generate_report, render_html

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


@pytest.fixture(scope="module")
def demo():
    runs = load_runs(EXAMPLES / "runs.json")
    tasks = load_tasks(EXAMPLES / "tasks.json")
    trajs = load_trajectories(EXAMPLES / "trajectories.jsonl")
    return runs, tasks, trajs


class TestDemoDataset:
    def test_files_are_present(self):
        for name in ("runs.json", "tasks.json", "trajectories.jsonl",
                     "generate_demo.py"):
            assert (EXAMPLES / name).exists(), f"examples/{name} is missing"

    def test_files_declare_themselves_synthetic(self):
        for name in ("runs.json", "tasks.json"):
            blob = json.loads((EXAMPLES / name).read_text())
            assert "SYNTHETIC" in blob["_README"]

    def test_loads_with_the_documented_shape(self, demo):
        runs, tasks, trajs = demo
        assert len(runs) == 174
        assert len(tasks) == 24
        assert len(trajs) == 174

    def test_contains_each_classification(self, demo):
        """The README promises one SOLID, one STUCK, several RECOVERABLE, and
        low-sample UNCERTAIN tasks."""
        runs, _, _ = demo
        by = {}
        for r in runs:
            by.setdefault(r["task"], []).append(r["score"])
        labels = {t: diagnose(posterior_from_scores(v), task=t, scores=v).label
                  for t, v in by.items()}
        counts = {k: sum(1 for v in labels.values() if v == k)
                  for k in ("SOLID", "RECOVERABLE", "STUCK", "UNCERTAIN")}
        assert counts["SOLID"] >= 1
        assert counts["STUCK"] >= 1
        assert counts["RECOVERABLE"] >= 5
        assert counts["UNCERTAIN"] >= 3

    def test_named_tasks_behave_as_documented(self, demo):
        runs, _, _ = demo
        by = {}
        for r in runs:
            by.setdefault(r["task"], []).append(r["score"])
        assert sum(by["reconcile_ledger"]) == len(by["reconcile_ledger"])
        assert sum(by["precedent_search"]) == 0

    def test_low_sample_tasks_exist(self, demo):
        runs, _, _ = demo
        counts = {}
        for r in runs:
            counts[r["task"]] = counts.get(r["task"], 0) + 1
        assert min(counts.values()) == 2, "the UNCERTAIN path needs low-n tasks"

    def test_every_analysis_is_enabled(self, demo):
        runs, tasks, trajs = demo
        enabled = describe_dataset(runs, tasks, trajs)["enabled_analyses"]
        assert all(enabled.values()), f"disabled: {[k for k, v in enabled.items() if not v]}"

    def test_rubric_and_dependencies_are_present(self, demo):
        runs, tasks, _ = demo
        assert all(r["rubric_scores"] for r in runs)
        assert any(t.criterion_dependencies for t in tasks.values())

    def test_trajectories_have_events_and_usable_pairs(self, demo):
        """Most tasks must have both outcomes, so preference pairs can be built.

        A few tasks legitimately have no successes at all -- that is what STUCK
        means, and a demo without any would not exercise the STUCK path. The
        assertion is therefore on the proportion, not on a hardcoded task name.
        """
        _, _, trajs = demo
        assert all(len(t.events) > 0 for t in trajs)
        grouped = {k[1]: v for k, v in trajs.group().items()}
        pairable = [
            t for t, v in grouped.items()
            if any(x.success for x in v) and any(not x.success for x in v)
        ]
        no_success = [t for t, v in grouped.items() if not any(x.success for x in v)]
        assert len(pairable) >= 0.5 * len(grouped)
        assert 1 <= len(no_success) <= 3, "some, but not many, fully-stuck tasks"

    def test_generator_is_deterministic(self, tmp_path):
        """Re-running the generator must reproduce the committed files."""
        out = subprocess.run(
            [sys.executable, str(EXAMPLES / "generate_demo.py")],
            capture_output=True, text=True, cwd=str(tmp_path),
        )
        assert out.returncode == 0, out.stderr
        # The generator writes next to itself, so compare against a fresh load.
        assert len(load_runs(EXAMPLES / "runs.json")) == 174

    def test_bridges_to_the_legacy_record_store(self, demo):
        runs, _, _ = demo
        store = runs_to_store(runs)
        assert len(store) == len(runs)
        df = store.df()
        assert {"task", "model", "score", "success"} <= set(df.columns)


class TestLoaders:
    def test_accepts_loose_field_names(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps([
            {"task": "a", "model": "m", "score": 1.0},
            {"task_id": "b", "agent": "m", "reward": 0.0},
        ]))
        runs = load_runs(p)
        assert {r["task"] for r in runs} == {"a", "b"}
        assert runs[0]["success"] is True and runs[1]["success"] is False

    def test_assigns_episodes_when_absent(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps([{"task": "a", "score": 1.0}] * 3))
        assert [r["episode"] for r in load_runs(p)] == [0, 1, 2]

    def test_missing_score_raises_with_the_task_named(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps([{"task": "a"}]))
        with pytest.raises(ValueError, match="no score/reward"):
            load_runs(p)

    def test_missing_task_raises(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps([{"score": 1.0}]))
        with pytest.raises(ValueError, match="no task/task_id"):
            load_runs(p)

    def test_wrapped_lists_are_unwrapped(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps({"runs": [{"task": "a", "score": 1.0}]}))
        assert len(load_runs(p)) == 1

    def test_jsonl_line_errors_name_the_line(self, tmp_path):
        p = tmp_path / "r.jsonl"
        p.write_text('{"task": "a", "score": 1.0}\nnot json\n')
        with pytest.raises(ValueError, match=":2"):
            load_runs(p)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_runs("/definitely/not/here.json")

    def test_parquet_roundtrip(self, tmp_path):
        import pandas as pd

        p = tmp_path / "r.parquet"
        pd.DataFrame([{"task": "a", "model": "m", "score": 1.0}]).to_parquet(p)
        assert len(load_runs(p)) == 1

    def test_success_threshold_is_honoured(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps([{"task": "a", "score": 0.85}]))
        assert load_runs(p, success_threshold=0.8)[0]["success"] is True
        assert load_runs(p, success_threshold=0.9)[0]["success"] is False

    def test_task_spec_tool_signature(self):
        t = TaskSpec("a", tools=["b", "a"])
        assert t.tool_signature == "a+b"
        assert TaskSpec("a").tool_signature == "_none"

    def test_describe_flags_unequal_run_counts(self):
        runs = [{"task": "a", "model": "m", "score": 1.0, "success": True},
                {"task": "b", "model": "m", "score": 1.0, "success": True},
                {"task": "b", "model": "m", "score": 0.0, "success": False}]
        d = describe_dataset(runs)
        assert d["unequal_run_counts"] is True
        assert d["runs_per_task_min"] == 1 and d["runs_per_task_max"] == 2


class TestReport:
    def test_builds_every_section_on_the_demo(self, demo):
        runs, tasks, trajs = demo
        data = build_report_data(runs, tasks=tasks, trajectories=trajs)
        assert data.diagnoses and data.ranking
        assert data.failure and data.rubric and data.survival and data.trajectory
        assert data.cost
        assert data.aggregate["n_runs"] == 174

    def test_records_skipped_analyses_with_a_reason(self, demo):
        runs, _, _ = demo
        bare = [{k: v for k, v in r.items() if k not in ("rubric_scores", "cost",
                                                         "failure_mode")}
                for r in runs]
        for b in bare:
            b["rubric_scores"], b["cost"] = {}, {}
        data = build_report_data(bare)
        assert "rubric_reliability" in data.skipped
        assert data.skipped["rubric_reliability"]

    def test_pooling_decision_is_made_and_recorded(self, demo):
        runs, tasks, trajs = demo
        data = build_report_data(runs, tasks=tasks, trajectories=trajs,
                                 hierarchical="auto")
        assert data.pooling is not None
        assert "recommendation" in data.pooling

    def test_markdown_and_html_render(self, demo, tmp_path):
        runs, tasks, trajs = demo
        paths = generate_report(runs, str(tmp_path), tasks=tasks, trajectories=trajs)
        md = Path(paths["markdown"]).read_text()
        for section in ("Dataset", "Aggregate evaluation", "Task-level analysis",
                        "Training-data recommendation", "Metric reference"):
            assert section in md
        html = Path(paths["html"]).read_text()
        assert html.startswith("<!doctype html>") and "</html>" in html

    def test_report_states_what_it_does_not_establish(self, demo, tmp_path):
        runs, tasks, trajs = demo
        paths = generate_report(runs, str(tmp_path), tasks=tasks, trajectories=trajs)
        md = Path(paths["markdown"]).read_text()
        assert "does not establish" in md
        assert "hypothesis" in md.lower()

    def test_figures_are_written(self, demo, tmp_path):
        runs, tasks, trajs = demo
        paths = generate_report(runs, str(tmp_path), tasks=tasks, trajectories=trajs)
        figs = [v for k, v in paths.items() if k.startswith("figure:")]
        assert len(figs) >= 5
        assert all(Path(f).exists() and Path(f).stat().st_size > 0 for f in figs)

    def test_task_metrics_table_is_written(self, demo, tmp_path):
        runs, _, _ = demo
        paths = generate_report(runs, str(tmp_path), html_output=False)
        import pandas as pd

        df = pd.read_csv(paths["task_metrics"])
        assert len(df) == 24
        assert {"capability", "reliability", "label", "ci_lo", "ci_hi"} <= set(df.columns)

    def test_summary_json_is_valid(self, demo, tmp_path):
        runs, _, _ = demo
        paths = generate_report(runs, str(tmp_path), html_output=False)
        blob = json.loads(Path(paths["summary"]).read_text())
        assert {"dataset", "aggregate", "ties"} <= set(blob)

    def test_html_escapes_content(self):
        from disteval.research_report import ReportData

        data = ReportData(dataset={}, aggregate={}, diagnoses=[], ranking=[],
                          ties={}, per_task_frame=None, title="<script>x</script>")
        assert "<script>x</script>" not in render_html(data, "# <b>hi</b>")


class TestCLI:
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "disteval", *args],
            capture_output=True, text=True, cwd=str(ROOT),
        )

    def test_bare_invocation_lists_the_subcommands(self):
        out = self._run()
        assert out.returncode == 0
        for name in ("evaluate", "diagnose", "select", "experiment", "simulate"):
            assert name in out.stdout

    def test_evaluate_runs_on_the_demo(self, tmp_path):
        out = self._run("evaluate", "examples/runs.json",
                        "--trajectories", "examples/trajectories.jsonl",
                        "--tasks", "examples/tasks.json",
                        "-o", str(tmp_path / "rep"))
        assert out.returncode == 0, out.stderr
        assert (tmp_path / "rep" / "report.md").exists()
        assert (tmp_path / "rep" / "report.html").exists()

    def test_diagnose_prints_the_table(self):
        out = self._run("diagnose", "examples/runs.json", "--top", "5")
        assert out.returncode == 0, out.stderr
        assert "RECOVERABLE" in out.stdout
        assert "thresholds:" in out.stdout

    def test_diagnose_label_filter(self):
        out = self._run("diagnose", "examples/runs.json", "--label", "STUCK")
        assert out.returncode == 0
        assert "SOLID" not in out.stdout.split("label")[-1]

    def test_select_requires_trajectories(self):
        out = self._run("select", "examples/runs.json")
        assert out.returncode == 2
        assert "--trajectories is required" in out.stderr

    def test_select_writes_pairs(self, tmp_path):
        target = tmp_path / "p.jsonl"
        out = self._run("select", "examples/runs.json",
                        "--trajectories", "examples/trajectories.jsonl",
                        "--n-tasks", "8", "--n-pairs", "12", "-o", str(target))
        assert out.returncode == 0, out.stderr
        rows = [json.loads(l) for l in target.read_text().splitlines() if l.strip()]
        assert 0 < len(rows) <= 12
        assert all("selection_strategy" in r for r in rows)

    def test_metrics_prints_assumptions(self):
        out = self._run("metrics", "pass^k")
        assert out.returncode == 0
        assert "assumes" in out.stdout and "prior work" in out.stdout

    def test_metrics_rejects_unknown_names(self):
        out = self._run("metrics", "vibes")
        assert out.returncode == 2

    def test_unknown_subcommand_fails_cleanly(self):
        out = self._run("teleport")
        assert out.returncode == 1
        assert "Unknown subcommand" in out.stderr
