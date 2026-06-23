"""Tests for disteval.__main__ CLI dispatcher."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

from disteval.__main__ import main


JOBS_ROOT = Path(__file__).parent.parent / "jobs"


class TestCLIDispatcher:
    def test_no_args_prints_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["disteval"]
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "report" in captured.out
        assert "compare" in captured.out

    def test_unknown_subcommand_exits_with_error(self, capsys):
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["disteval", "nope"]
            main()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "Unknown subcommand" in captured.err

    def test_help_flag_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["disteval", "--help"]
            main()
        assert exc.value.code == 0

    def test_engine_help_flag_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["disteval", "engine", "--help"]
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "--enable-recursion" in captured.out


class TestReportSubcommand:
    def test_report_routes_to_report_main(self, monkeypatch, capsys):
        calls = []
        def fake_main():
            calls.append("report")
        monkeypatch.setattr("disteval.report.main", fake_main)
        sys.argv = ["disteval", "report"]
        main()
        assert calls == ["report"]


class TestCompareSubcommand:
    def test_compare_routes_to_compare_report_main(self, monkeypatch, capsys):
        calls = []
        def fake_main():
            calls.append("compare")
        monkeypatch.setattr("disteval.compare_report.main", fake_main)
        sys.argv = ["disteval", "compare"]
        main()
        assert calls == ["compare"]


class TestSimSubcommand:
    def test_sim_routes_to_training_sim_main(self, monkeypatch, capsys):
        calls = []
        def fake_main():
            calls.append("sim")
        monkeypatch.setattr("disteval.training_sim.main", fake_main)
        sys.argv = ["disteval", "sim"]
        main()
        assert calls == ["sim"]


class TestEngineSubcommand:
    @pytest.mark.integration
    @pytest.mark.requires_harbor
    def test_engine_runs_on_real_job_dir(self):
        job_dir = str(JOBS_ROOT / "run_A" / "disteval-run-A")
        if not os.path.isdir(job_dir):
            pytest.skip("Harbor job directory not present")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            sys.argv = [
                "disteval", "engine", job_dir,
                "--agent", "agent-A",
                "--model", "test-model",
                "--output", out_path,
                "--cycle", "1",
            ]
            main()
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)
