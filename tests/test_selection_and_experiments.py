"""Selection strategies, preference pairs, config, splits, and the pipeline."""
import json

import numpy as np
import pytest

from disteval.experiments.config import ExperimentConfig, load_config, save_config
from disteval.experiments.pipeline import run_experiment
from disteval.experiments.splits import make_split
from disteval.experiments.sweep import SweepSpec, aggregate_sweep, expand_sweep, run_sweep
from disteval.experiments.tracking import ExperimentRun
from disteval.experiments.training import (
    ExportOnlyBackend,
    SimulatedBackend,
    make_backend,
)
from disteval.reliability.classify import diagnose
from disteval.reliability.posterior import posterior_from_scores
from disteval.selection.pairs import (
    PairConfig,
    PreferenceDataset,
    build_dataset,
    build_pairs_for_task,
    to_dpo_jsonl,
    to_ranking_jsonl,
)
from disteval.selection.selectors import SELECTORS, make_selector
from disteval.sim.world import SimulatedWorld, WorldConfig
from disteval.trajectory.events import from_generic_steps


def _diags(n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = float(rng.choice([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]))
        s = list((rng.random(8) < p).astype(float))
        out.append(diagnose(posterior_from_scores(s), task=f"t{i}", model="m", scores=s,
                            domain=f"d{i % 4}"))
    return out


def _trajs(tasks, n_success=3, n_fail=3):
    out = {}
    for t in tasks:
        runs = []
        for i in range(n_success):
            runs.append(from_generic_steps(
                [{"tool": "s", "args": {"q": "a"}}, {"tool": "r", "args": {"p": "good"},
                                                     "target": "good"}],
                f"{t}_s{i}", t, "m", score=1.0, success=True))
        for i in range(n_fail):
            runs.append(from_generic_steps(
                [{"tool": "s", "args": {"q": "a"}}, {"tool": "r", "args": {"p": f"bad{i}"},
                                                     "target": f"bad{i}"}],
                f"{t}_f{i}", t, "m", score=0.0, success=False))
        out[t] = runs
    return out


class TestSelectors:
    @pytest.mark.parametrize("name", [k for k in SELECTORS if k != "oracle"])
    def test_returns_at_most_the_requested_count(self, name):
        r = make_selector(name).select(_diags(), 10)
        assert r.delivered <= 10
        assert len(set(r.tasks)) == r.delivered, "no task selected twice"

    @pytest.mark.parametrize("name", [k for k in SELECTORS if k != "oracle"])
    def test_is_reproducible_under_a_seed(self, name):
        a = make_selector(name, seed=5).select(_diags(), 12)
        b = make_selector(name, seed=5).select(_diags(), 12)
        assert a.tasks == b.tasks

    def test_shortfall_is_reported_not_backfilled(self):
        r = make_selector("recoverability", restrict_to_recoverable=True).select(_diags(), 999)
        assert r.shortfall > 0
        assert "shortfall is reported rather than backfilled" in r.note

    def test_hardest_picks_the_lowest_posterior_means(self):
        d = _diags()
        r = make_selector("hardest").select(d, 5)
        chosen = [x for x in d if x.task in r.tasks]
        rest = [x for x in d if x.task not in r.tasks]
        assert max(x.posterior_mean for x in chosen) <= min(x.posterior_mean for x in rest) + 1e-9

    def test_variance_picks_the_most_inconsistent(self):
        d = _diags()
        r = make_selector("highest_variance").select(d, 5)
        chosen = [x for x in d if x.task in r.tasks]
        assert all(0 < x.n_success < x.n_runs for x in chosen)

    def test_success_failure_requires_both_outcomes(self):
        d = _diags()
        r = make_selector("success_failure").select(d, 20)
        chosen = {x.task: x for x in d if x.task in r.tasks}
        assert all(0 < x.n_success < x.n_runs for x in chosen.values())

    def test_recoverability_ranks_by_the_named_estimator(self):
        d = _diags()
        r = make_selector("recoverability", estimator="gap").select(d, 6)
        vals = [r.scores[t] for t in r.tasks]
        assert vals == sorted(vals, reverse=True)

    def test_uncertainty_aware_demotes_low_sample_tasks(self):
        s_low = [1.0, 0.0]
        s_high = [1.0, 0.0] * 8
        d = [diagnose(posterior_from_scores(s_low), task="low", scores=s_low),
             diagnose(posterior_from_scores(s_high), task="high", scores=s_high)]
        r = make_selector("uncertainty_aware").select(d, 2)
        assert r.tasks[0] == "high"

    def test_oracle_ranks_by_true_benefit(self):
        d = _diags()
        benefit = {x.task: float(i) for i, x in enumerate(d)}
        r = make_selector("oracle", benefit=benefit).select(d, 3)
        assert r.tasks == [d[-1].task, d[-2].task, d[-3].task]

    def test_every_selection_carries_a_reason(self):
        r = make_selector("recoverability").select(_diags(), 5)
        assert all(r.reasons[t] for t in r.tasks)

    def test_unknown_selector_raises(self):
        with pytest.raises(ValueError):
            make_selector("wishful_thinking")

    def test_empty_pool(self):
        assert make_selector("random").select([], 5).delivered == 0


class TestPreferencePairs:
    def test_pairs_are_built_within_a_task(self):
        tr = _trajs(["a"])["a"]
        pairs = build_pairs_for_task([t for t in tr if t.success],
                                     [t for t in tr if not t.success], task="a")
        assert all(p.task_id == "a" for p in pairs)
        assert all(p.chosen_score > p.rejected_score for p in pairs)

    def test_margin_threshold_filters(self):
        s = from_generic_steps([{"tool": "a"}], "s", "t", "m", score=1.0, success=True)
        f = from_generic_steps([{"tool": "a"}], "f", "t", "m", score=0.9, success=False)
        assert build_pairs_for_task([s], [f], config=PairConfig(min_margin=0.5)) == []
        assert build_pairs_for_task([s], [f], config=PairConfig(min_margin=0.05))

    def test_cap_per_task_is_enforced(self):
        tr = _trajs(["a"], n_success=4, n_fail=4)["a"]
        pairs = build_pairs_for_task([t for t in tr if t.success],
                                     [t for t in tr if not t.success],
                                     config=PairConfig(max_pairs_per_task=2,
                                                       deduplicate_by_divergence=False))
        assert len(pairs) == 2

    def test_deduplication_collapses_identical_divergences(self):
        tr = _trajs(["a"], n_success=3, n_fail=3)["a"]
        succ = [t for t in tr if t.success]
        fail = [t for t in tr if not t.success]
        deduped = build_pairs_for_task(succ, fail, config=PairConfig(
            max_pairs_per_task=99, deduplicate_by_divergence=True))
        raw = build_pairs_for_task(succ, fail, config=PairConfig(
            max_pairs_per_task=99, deduplicate_by_divergence=False))
        assert len(deduped) <= len(raw)

    def test_environment_matching_can_reject_a_pair(self):
        s = from_generic_steps([{"tool": "a"}], "s", "t", "m", score=1.0, success=True)
        f = from_generic_steps([{"tool": "a"}], "f", "t", "m", score=0.0, success=False)
        s.meta = {"environment": "v1"}
        f.meta = {"environment": "v2"}
        assert build_pairs_for_task([s], [f], config=PairConfig(match_environment=True)) == []
        assert build_pairs_for_task([s], [f], config=PairConfig(match_environment=False))

    def test_pairs_carry_full_provenance(self):
        tr = _trajs(["a"])["a"]
        p = build_pairs_for_task([t for t in tr if t.success],
                                 [t for t in tr if not t.success],
                                 task="a", recoverability=0.7, strategy="demo")[0]
        d = p.to_dict()
        assert d["selection_strategy"] == "demo"
        assert d["recoverability_score"] == pytest.approx(0.7)
        assert d["metadata"]["chosen_id"] and d["metadata"]["rejected_id"]
        assert d["margin"] == pytest.approx(1.0)

    def test_dataset_respects_the_pair_budget(self):
        tasks = [f"t{i}" for i in range(8)]
        ds = build_dataset(tasks, _trajs(tasks), strategy="x", target_pairs=5)
        assert len(ds) <= 5

    def test_dataset_reports_a_shortfall(self):
        tasks = [f"t{i}" for i in range(2)]
        ds = build_dataset(tasks, _trajs(tasks, 1, 1), strategy="x", target_pairs=100)
        assert ds.shortfall > 0
        assert "not backfilled" in ds.note

    def test_dataset_skips_tasks_without_both_outcomes(self):
        tr = _trajs(["a"], n_success=2, n_fail=0)
        ds = build_dataset(["a"], tr, strategy="x", target_pairs=10)
        assert len(ds) == 0 and ds.tasks_with_pairs == []

    def test_exports_are_valid_jsonl(self, tmp_path):
        tasks = [f"t{i}" for i in range(4)]
        ds = build_dataset(tasks, _trajs(tasks), strategy="x", target_pairs=8)
        dpo, rank = tmp_path / "d.jsonl", tmp_path / "r.jsonl"
        n1, n2 = to_dpo_jsonl(ds, str(dpo)), to_ranking_jsonl(ds, str(rank))
        for path, n in ((dpo, n1), (rank, n2)):
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            assert len(rows) == n
        first = json.loads(dpo.read_text().splitlines()[0])
        assert {"task_id", "chosen", "rejected", "selection_strategy"} <= set(first)

    def test_summary_reports_the_accounting(self):
        tasks = [f"t{i}" for i in range(4)]
        s = build_dataset(tasks, _trajs(tasks), strategy="x", target_pairs=8).summary()
        assert {"n_pairs", "shortfall", "mean_margin", "pairs_per_task"} <= set(s)


class TestConfig:
    def test_fingerprint_is_stable_and_sensitive(self):
        a, b = ExperimentConfig(), ExperimentConfig()
        assert a.experiment_id == b.experiment_id
        c = ExperimentConfig.from_dict({"evaluation": {"runs_per_task": 16}})
        assert c.experiment_id != a.experiment_id

    def test_unknown_keys_and_sections_raise(self):
        with pytest.raises(ValueError, match="unknown configuration key"):
            ExperimentConfig.from_dict({"evaluation": {"typo": 1}})
        with pytest.raises(ValueError, match="unknown top-level"):
            ExperimentConfig.from_dict({"nope": {}})

    def test_threshold_ordering_is_enforced(self):
        with pytest.raises(ValueError):
            ExperimentConfig.from_dict({"reliability": {"tau_cap": 0.95}}).validate()

    def test_warnings_for_weak_designs(self):
        cfg = ExperimentConfig.from_dict({
            "experiment": {"n_seeds": 1}, "selection": {"n_pairs": 3},
            "split": {"strategy": "task"},
        })
        w = " ".join(cfg.validate())
        assert "variance estimate" in w and "n_pairs" in w and "domain_holdout" in w

    def test_yaml_roundtrip(self, tmp_path):
        cfg = ExperimentConfig.from_dict({"experiment": {"name": "x", "seed": 9}})
        p = tmp_path / "c.yaml"
        save_config(cfg, p)
        assert load_config(p).experiment_id == cfg.experiment_id

    def test_thresholds_are_derived_from_the_config(self):
        cfg = ExperimentConfig.from_dict({"reliability": {"tau_rel": 0.75}})
        assert cfg.thresholds().tau_rel == pytest.approx(0.75)


class TestSplits:
    def test_domain_holdout_is_group_disjoint(self):
        tasks = [f"t{i}" for i in range(40)]
        dom = {t: f"d{i % 5}" for i, t in enumerate(tasks)}
        s = make_split(tasks, strategy="domain_holdout", domains=dom, seed=0)
        assert s.is_group_disjoint
        assert set(s.train) | set(s.test) == set(tasks)
        assert not (set(s.train) & set(s.test))

    def test_task_split_declares_that_it_is_in_distribution(self):
        s = make_split([f"t{i}" for i in range(20)], strategy="task", seed=0)
        assert "does not demonstrate transfer" in s.note

    def test_named_holdout_groups(self):
        tasks = [f"t{i}" for i in range(20)]
        dom = {t: f"d{i % 4}" for i, t in enumerate(tasks)}
        s = make_split(tasks, strategy="domain_holdout", domains=dom,
                       holdout_groups=["d1"])
        assert s.test_groups == ["d1"]

    def test_missing_holdout_group_raises(self):
        with pytest.raises(ValueError, match="not present"):
            make_split(["a"], strategy="domain_holdout", domains={"a": "d0"},
                       holdout_groups=["nope"])

    def test_refuses_to_hold_out_everything(self):
        tasks = [f"t{i}" for i in range(10)]
        with pytest.raises(ValueError, match="nothing would be left"):
            make_split(tasks, strategy="domain_holdout",
                       domains={t: f"d{i % 3}" for i, t in enumerate(tasks)},
                       test_fraction=0.99)

    def test_too_few_groups_is_flagged(self):
        tasks = [f"t{i}" for i in range(10)]
        s = make_split(tasks, strategy="domain_holdout",
                       domains={t: f"d{i % 2}" for i, t in enumerate(tasks)})
        assert "single observation" in s.note

    def test_missing_grouping_map_raises(self):
        with pytest.raises(ValueError, match="needs a grouping map"):
            make_split(["a", "b"], strategy="environment")

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            make_split(["a"], strategy="vibes", domains={"a": "d"})


class TestTraining:
    def test_export_only_claims_nothing(self, tmp_path):
        ds = PreferenceDataset([], "x", 0, [], [])
        r = ExportOnlyBackend().train(ds, str(tmp_path))
        assert r.is_real_training is False and r.is_simulated is False
        assert "no training was performed" in r.note

    def test_simulated_effects_come_from_true_benefit_not_the_selector(self, tmp_path):
        world = SimulatedWorld(WorldConfig(n_tasks=20, seed=0))
        tasks = world.tasks()[:5]
        ds = build_dataset(tasks, _trajs(tasks), strategy="x", target_pairs=15)
        b = SimulatedBackend(world.true_benefit(),
                             {k: v.domain for k, v in world.truth.items()}, seed=0)
        r = b.train(ds, str(tmp_path), all_tasks=world.tasks())
        assert r.is_simulated is True and r.is_real_training is False
        trained = [r.task_effects[t] for t in tasks]
        held = [r.task_effects[t] for t in world.tasks() if t not in tasks]
        assert np.mean(trained) > np.mean(held), "transfer must be weaker than direct"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            make_backend("gradient_fairy")


class TestPipeline:
    def _setup(self, **overrides):
        cfg = ExperimentConfig.from_dict({
            "experiment": {"name": "t", "seed": 0, "n_seeds": 2},
            "evaluation": {"runs_per_task": 8},
            "selection": {"n_tasks": 15, "n_pairs": 30, "method": "all"},
            **overrides,
        })
        world = SimulatedWorld(WorldConfig(n_tasks=60, n_domains=4,
                                           benefit_coupling="strong", seed=0))
        rng = np.random.default_rng(0)
        return cfg, world, (lambda t, i: float(world.run(t, rng)["score"]))

    def test_runs_end_to_end(self):
        cfg, world, run_fn = self._setup()
        res = run_experiment(world.tasks(), run_fn, cfg,
                             domains={k: v.domain for k, v in world.truth.items()},
                             true_benefit=world.true_benefit())
        assert res.phase_a.n_executions == 60 * 8
        assert len(res.outcomes) == 2 * 7
        s = res.summary()
        assert {"strategy", "heldout_delta", "heldout_delta_se"} <= set(s.columns)

    def test_held_out_tasks_are_never_selected_into(self):
        cfg, world, run_fn = self._setup()
        res = run_experiment(world.tasks(), run_fn, cfg,
                             domains={k: v.domain for k, v in world.truth.items()},
                             true_benefit=world.true_benefit())
        assert res.split.is_group_disjoint

    def test_simulated_backend_demands_ground_truth(self):
        cfg, world, run_fn = self._setup()
        with pytest.raises(ValueError, match="circular"):
            run_experiment(world.tasks(), run_fn, cfg, true_benefit=None)

    def test_config_selects_which_strategies_run(self):
        cfg, world, run_fn = self._setup(selection={"n_tasks": 10, "n_pairs": 20,
                                                    "method": "recoverability"})
        res = run_experiment(world.tasks(), run_fn, cfg,
                             domains={k: v.domain for k, v in world.truth.items()},
                             true_benefit=world.true_benefit())
        assert {o.strategy for o in res.outcomes} == {"recoverability"}

    def test_unknown_method_raises(self):
        cfg, world, run_fn = self._setup(selection={"method": "telepathy"})
        with pytest.raises(ValueError, match="not a known strategy"):
            run_experiment(world.tasks(), run_fn, cfg,
                           true_benefit=world.true_benefit())

    def test_unequal_pair_counts_produce_a_warning(self):
        cfg, world, run_fn = self._setup()
        res = run_experiment(world.tasks(), run_fn, cfg,
                             domains={k: v.domain for k, v in world.truth.items()},
                             true_benefit=world.true_benefit())
        if len({o.n_pairs for o in res.outcomes}) > 1:
            assert any("different pair counts" in w for w in res.warnings)


class TestTrackingAndSweep:
    def test_run_directory_is_self_describing(self, tmp_path):
        cfg = ExperimentConfig.from_dict({"experiment": {"name": "trk"}})
        with ExperimentRun.create(cfg, tmp_path) as run:
            run.log(metric=1.0)
        root = tmp_path / cfg.experiment_id
        assert (root / "config.yaml").exists()
        meta = json.loads((root / "metadata.json").read_text())
        assert meta["status"] == "ok"
        assert "environment" in meta and "git_commit" in meta["environment"]
        assert (root / "COMPLETED").exists()

    def test_refuses_to_clobber_a_completed_run(self, tmp_path):
        cfg = ExperimentConfig.from_dict({"experiment": {"name": "dup"}})
        with ExperimentRun.create(cfg, tmp_path):
            pass
        with pytest.raises(FileExistsError):
            ExperimentRun.create(cfg, tmp_path)
        ExperimentRun.create(cfg, tmp_path, overwrite=True)

    def test_errors_are_recorded_not_swallowed(self, tmp_path):
        cfg = ExperimentConfig.from_dict({"experiment": {"name": "err"}})
        with pytest.raises(RuntimeError):
            with ExperimentRun.create(cfg, tmp_path):
                raise RuntimeError("boom")
        meta = json.loads((tmp_path / cfg.experiment_id / "metadata.json").read_text())
        assert meta["status"] == "error" and "boom" in meta["error"]

    def test_sweep_expands_the_cross_product(self):
        spec = SweepSpec(base={"experiment": {"name": "s"}},
                         sweep={"evaluation.runs_per_task": [2, 4],
                                "reliability.prior": ["jeffreys", "uniform"]})
        cells = expand_sweep(spec)
        assert spec.n_cells == 4 and len(cells) == 4
        assert len({c.experiment_id for _, c in cells}) == 4

    def test_sweep_records_failures_without_aborting(self):
        spec = SweepSpec(base={}, sweep={"evaluation.runs_per_task": [2, 4]})
        calls = []

        def run_one(cfg):
            calls.append(cfg)
            if len(calls) == 1:
                raise RuntimeError("cell failed")

            class R:
                warnings = []

                def summary(self):
                    import pandas as pd

                    return pd.DataFrame([{"strategy": "x", "heldout_delta": 0.1}])

            return R()

        index = run_sweep(spec, run_one)
        assert len(index) == 2
        assert set(index["status"]) == {"error", "ok"}
        table = aggregate_sweep(index)
        assert len(table) == 2

    def test_sweep_validates_axes_eagerly(self, tmp_path):
        import yaml

        from disteval.experiments.sweep import load_sweep

        bad = tmp_path / "s.yaml"
        bad.write_text(yaml.safe_dump({"base": {}, "sweep": {"no_dot": [1]}}))
        with pytest.raises(ValueError, match="dotted config path"):
            load_sweep(bad)
        bad.write_text(yaml.safe_dump({"base": {}, "sweep": {"a.b": "not a list"}}))
        with pytest.raises(ValueError, match="non-empty list"):
            load_sweep(bad)
