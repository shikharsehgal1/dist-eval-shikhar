"""Metamorphic task generation: relations, targeting, gates, diversity."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from disteval.reliability.criterion import gap_profiles
from disteval.reliability.frontier import (
    MATH_CODE_BAND,
    SWE_BAND,
    FrontierBand,
    band_decision,
    band_information,
    band_probability,
    expected_runs_to_resolve_band,
    plugin_band_error,
)
from disteval.reliability.posterior import binary_posterior
from disteval.taskgen import (
    GATES,
    RELATIONS,
    CallableRelation,
    RelationKind,
    SeedTask,
    TaskGenerator,
    ValidationConfig,
    batch_diversity,
    get_relation,
    register_relation,
    relations_stressing,
    validate_batch,
    validate_task,
)

ROOT = Path(__file__).resolve().parent.parent


def _payload(i=0):
    return {
        "instruction": f"Reconcile ledger {i} for Acme{i}",
        "files": [f"a{i}.csv", f"b{i}.md", "notes.txt"],
        "entities": [f"Acme{i}"],
        "quantities": {"total": 100.0 * i + 50},
        "required_inputs": [f"a{i}.csv"],
    }


def _seeds(n=6):
    return [SeedTask(f"task_{i}", _payload(i), verifier=f"verify_{i}.sh",
                     domain="finance") for i in range(n)]


class TestRelations:
    def test_every_relation_declares_a_valid_kind(self):
        for rel in RELATIONS.values():
            assert rel.kind in (RelationKind.INVARIANT, RelationKind.EQUIVARIANT,
                                RelationKind.REFUTED)
            assert rel.description and rel.name

    def test_equivariant_relations_must_supply_answer_maps(self):
        """Without them the verifier cannot transfer, which is the whole point."""
        from disteval.taskgen.relations import MetamorphicRelation

        with pytest.raises(ValueError, match="answer_map"):
            MetamorphicRelation(
                name="bad", kind=RelationKind.EQUIVARIANT, description="x",
                transform=lambda p, r: (dict(p), {}),
            )

    def test_unknown_kind_raises(self):
        from disteval.taskgen.relations import MetamorphicRelation

        with pytest.raises(ValueError, match="unknown relation kind"):
            MetamorphicRelation(name="b", kind="vibes", description="x",
                                transform=lambda p, r: (dict(p), {}))

    @pytest.mark.parametrize("name", sorted(RELATIONS))
    def test_each_relation_changes_the_payload(self, name):
        rng = np.random.default_rng(0)
        out, evidence = get_relation(name).apply(_payload(1), rng)
        assert out != _payload(1), f"{name} was a no-op"
        assert evidence, f"{name} recorded no evidence of what it changed"

    def test_invariant_relations_preserve_the_entity_set(self):
        """An invariant relation must not change what the task is about."""
        rng = np.random.default_rng(0)
        base = _payload(1)
        for name in ("reorder_files", "add_distractor", "paraphrase_instruction"):
            out, _ = get_relation(name).apply(base, rng)
            assert out.get("entities") == base["entities"]
            assert out.get("quantities") == base["quantities"]

    def test_reorder_preserves_the_file_set(self):
        rng = np.random.default_rng(0)
        out, _ = get_relation("reorder_files").apply(_payload(1), rng)
        assert sorted(out["files"]) == sorted(_payload(1)["files"])
        assert out["files"] != _payload(1)["files"]

    def test_add_distractor_only_adds(self):
        rng = np.random.default_rng(0)
        base = _payload(1)
        out, ev = get_relation("add_distractor").apply(base, rng)
        assert set(base["files"]) <= set(out["files"])
        assert len(out["files"]) > len(base["files"])
        assert out["distractors"] == ev["added"]

    def test_rename_is_consistent_across_every_field(self):
        rng = np.random.default_rng(0)
        out, ev = get_relation("rename_entity").apply(_payload(1), rng)
        old, new = ev["old"], ev["new"]
        assert new in out["entities"] and old not in out["entities"]
        assert old not in out["instruction"] or new in out["instruction"]

    def test_scale_multiplies_every_quantity_by_one_factor(self):
        rng = np.random.default_rng(0)
        base = {"instruction": "x", "quantities": {"a": 2.0, "b": 5.0}}
        out, ev = get_relation("scale_quantities").apply(base, rng)
        k = ev["factor"]
        assert out["quantities"] == {"a": 2.0 * k, "b": 5.0 * k}

    def test_refuted_relation_removes_a_required_input(self):
        rng = np.random.default_rng(0)
        out, ev = get_relation("remove_required_input").apply(_payload(1), rng)
        assert ev["removed"] not in out["required_inputs"]
        assert get_relation("remove_required_input").kind == RelationKind.REFUTED

    def test_relations_raise_when_not_applicable(self):
        rng = np.random.default_rng(0)
        for name, bad in [
            ("reorder_files", {"files": ["only.csv"]}),
            ("rename_entity", {"instruction": "x"}),
            ("scale_quantities", {"instruction": "x"}),
            ("paraphrase_instruction", {"instruction": ""}),
            ("remove_required_input", {"instruction": "x"}),
        ]:
            with pytest.raises(ValueError):
                get_relation(name).apply(bad, rng)

    def test_stresses_lookup_matches_both_directions(self):
        assert relations_stressing("retrieval")
        assert relations_stressing("r_retrieve")
        assert relations_stressing("nonexistent_thing") == []

    def test_registry_is_extensible(self):
        from disteval.taskgen.relations import MetamorphicRelation

        rel = MetamorphicRelation(
            name="_test_rel", kind=RelationKind.INVARIANT, description="t",
            transform=lambda p, r: ({**p, "marker": 1}, {"did": "it"}),
            stresses=("planning",),
        )
        register_relation(rel)
        try:
            assert get_relation("_test_rel") is rel
            assert rel in relations_stressing("planning")
        finally:
            RELATIONS.pop("_test_rel", None)

    def test_unknown_relation_raises(self):
        with pytest.raises(ValueError):
            get_relation("telepathy")

    def test_callable_relation_flags_model_generated(self):
        rel = CallableRelation("llm_para", lambda s: s + " (reworded)",
                               stresses=("planning",))
        out, _ = rel.apply({"instruction": "do the thing"}, np.random.default_rng(0))
        assert out["_model_generated"] is True

    def test_callable_relation_rejects_a_no_op_model(self):
        rel = CallableRelation("noop", lambda s: s)
        with pytest.raises(ValueError, match="no change"):
            rel.apply({"instruction": "x"}, np.random.default_rng(0))


class TestGenerator:
    def test_requires_a_verifier_by_default(self):
        with pytest.raises(ValueError, match="cannot be graded"):
            TaskGenerator([SeedTask("x", _payload(0))])

    def test_dry_run_can_skip_the_verifier_requirement(self):
        g = TaskGenerator([SeedTask("x", _payload(0))], require_verifier=False)
        assert len(g.generate(3).tasks) > 0

    def test_rejects_an_empty_seed_set(self):
        with pytest.raises(ValueError, match="no seed"):
            TaskGenerator([])

    def test_every_variant_inherits_its_seed_verifier(self):
        rep = TaskGenerator(_seeds(), seed=0).generate(12)
        for t in rep.tasks:
            assert t.verifier == f"verify_{t.seed_id.split('_')[1]}.sh"
            assert t.verifier_note, "must say how to apply the verifier"

    def test_verifier_note_matches_the_relation_kind(self):
        rep = TaskGenerator(_seeds(), seed=0).generate(20)
        for t in rep.tasks:
            if t.relation_kind == RelationKind.REFUTED:
                assert "decline" in t.verifier_note
            elif t.relation_kind == RelationKind.EQUIVARIANT:
                assert "invert" in t.verifier_note
            else:
                assert "unchanged" in t.verifier_note

    def test_respects_the_per_seed_cap(self):
        rep = TaskGenerator(_seeds(3), seed=0).generate(30, max_per_seed=2)
        counts = {}
        for t in rep.tasks:
            counts[t.seed_id] = counts.get(t.seed_id, 0) + 1
        assert max(counts.values()) <= 2

    def test_respects_the_per_relation_cap(self):
        rep = TaskGenerator(_seeds(), seed=0).generate(30, max_per_relation=2)
        assert max(rep.relations_used.values()) <= 2

    def test_never_exceeds_the_requested_count(self):
        for n in (0, 1, 5, 40):
            assert len(TaskGenerator(_seeds(), seed=0).generate(n).tasks) <= n

    def test_shortfall_is_reported_not_papered_over(self):
        rep = TaskGenerator(_seeds(1), seed=0).generate(50, max_per_seed=2)
        assert len(rep.tasks) < 50
        assert any("shortfall is reported" in w for w in rep.warnings)

    def test_is_reproducible_under_a_seed(self):
        a = TaskGenerator(_seeds(), seed=7).generate(10)
        b = TaskGenerator(_seeds(), seed=7).generate(10)
        assert [t.task_id for t in a.tasks] == [t.task_id for t in b.tasks]
        assert [t.relation for t in a.tasks] == [t.relation for t in b.tasks]

    def test_ids_are_unique_and_traceable(self):
        rep = TaskGenerator(_seeds(), seed=0).generate(20)
        ids = [t.task_id for t in rep.tasks]
        assert len(set(ids)) == len(ids)
        for t in rep.tasks:
            assert t.seed_id in t.task_id and t.relation in t.task_id

    def test_warns_on_a_shared_model_family(self):
        rep = TaskGenerator(_seeds(), generator_id="gpt-4o-synth",
                            evaluated_model="gpt-4o-agent", seed=0).generate(5)
        assert any("share a model family" in w for w in rep.warnings)

    def test_no_family_warning_when_independent(self):
        rep = TaskGenerator(_seeds(), generator_id="structural",
                            evaluated_model="claude-agent", seed=0).generate(5)
        assert not any("share a model family" in w for w in rep.warnings)

    def test_model_generated_variants_are_flagged(self):
        rel = CallableRelation("llm_para", lambda s: s + " (v2)")
        rep = TaskGenerator(_seeds(2), relations=[rel], seed=0).generate(4)
        assert all(t.model_generated for t in rep.tasks)
        assert any("weaker guarantee" in w for w in rep.warnings)

    def test_inapplicable_transforms_are_recorded_as_rejections(self):
        bare = [SeedTask("bare", {"instruction": "x"}, verifier="v.sh")]
        rep = TaskGenerator(bare, seed=0).generate(10)
        assert rep.rejected
        assert all("not applicable" in r["reason"] for r in rep.rejected)

    def test_report_serialises(self):
        rep = TaskGenerator(_seeds(), seed=0).generate(8)
        assert len(rep.to_frame()) == len(rep.tasks)
        assert rep.summary()["generated"] == len(rep.tasks)
        assert json.dumps(rep.tasks[0].to_dict(), default=str)


class TestTargeting:
    def _profiles(self):
        # t0 and t1 have one unstable criterion each; t2 is uniformly stable.
        runs = {
            "t0": [{"a": 1.0, "b": float(i % 2)} for i in range(8)],
            "t1": [{"a": float(i % 2), "b": 1.0} for i in range(8)],
            "t2": [{"a": 1.0, "b": 1.0} for _ in range(8)],
        }
        return gap_profiles(runs)

    def test_targets_pick_the_unstable_criteria(self):
        targets = TaskGenerator.targets_from_gaps(self._profiles())
        by_task = {}
        for t in targets:
            by_task.setdefault(t.task, []).append(t.criterion)
        assert "b" in by_task.get("t0", [])
        assert "a" in by_task.get("t1", [])

    def test_targets_are_ranked_by_gap_share(self):
        targets = TaskGenerator.targets_from_gaps(self._profiles())
        shares = [t.gap_share for t in targets]
        assert shares == sorted(shares, reverse=True)

    def test_targeting_biases_generation_toward_targeted_seeds(self):
        seeds = [SeedTask(f"t{i}", _payload(i), verifier="v.sh") for i in range(3)]
        targets = TaskGenerator.targets_from_gaps(self._profiles())
        rep = TaskGenerator(seeds, seed=0).generate(
            20, targets=targets, target_bias=1.0, max_per_seed=20
        )
        picked = {t.seed_id for t in rep.tasks}
        assert picked <= {t.task for t in targets}

    def test_targeted_variants_record_what_they_probe(self):
        seeds = [SeedTask(f"t{i}", _payload(i), verifier="v.sh") for i in range(3)]
        targets = TaskGenerator.targets_from_gaps(self._profiles())
        rep = TaskGenerator(seeds, seed=0).generate(10, targets=targets, target_bias=1.0)
        assert any(t.targets for t in rep.tasks)

    def test_no_targets_still_generates(self):
        assert len(TaskGenerator(_seeds(), seed=0).generate(6, targets=[]).tasks) == 6


class TestValidation:
    def test_clean_batch_passes_every_gate(self):
        seeds = _seeds()
        rep = TaskGenerator(seeds, seed=0).generate(15)
        out = validate_batch(rep, {s.task_id: s for s in seeds})
        assert out["n_valid"] == out["n_tasks"]
        assert out["failures_by_gate"] == {}

    def test_missing_verifier_fails_the_gradeable_gate(self):
        from disteval.taskgen import GeneratedTask

        t = GeneratedTask("g", "s", "reorder_files", RelationKind.INVARIANT,
                          _payload(0), None, "note", {})
        v = validate_task(t)
        assert v["gradeable"] is False and v["valid"] is False
        assert any("cannot be graded" in r for r in v["reasons"])

    def test_no_op_variant_fails_the_distinctness_gate(self):
        from disteval.taskgen import GeneratedTask

        p = _payload(0)
        t = GeneratedTask("g", "s", "reorder_files", RelationKind.INVARIANT,
                          dict(p), "v.sh", "note", {})
        v = validate_task(t, seed_payload=p)
        assert v["distinct_from_seed"] is False
        assert any("no-op" in r for r in v["reasons"])

    def test_empty_instruction_fails_well_formedness(self):
        from disteval.taskgen import GeneratedTask

        t = GeneratedTask("g", "s", "r", RelationKind.INVARIANT,
                          {"instruction": "   ", "files": ["a"]}, "v.sh", "n", {})
        assert validate_task(t)["well_formed"] is False

    def test_model_generated_is_excluded_until_confirmed(self):
        from disteval.taskgen import GeneratedTask

        t = GeneratedTask("g1", "s", "llm", RelationKind.INVARIANT, _payload(0),
                          "v.sh", "n", {}, model_generated=True)
        assert validate_task(t)["valid"] is False
        ok = validate_task(t, config=ValidationConfig(confirmed_ids=frozenset({"g1"})))
        assert ok["valid"] is True

    def test_solver_probe_rejects_unsolvable_variants(self):
        seeds = _seeds(2)
        rep = TaskGenerator(seeds, seed=0).generate(8)
        cfg = ValidationConfig(solver=lambda t: False, n_probe=2)
        out = validate_batch(rep, {s.task_id: s for s in seeds}, cfg)
        assert out["n_valid"] < out["n_tasks"]
        assert "solvability_bias_warning" in out

    def test_solver_probe_skips_refuted_relations(self):
        """A refuted variant is unsolvable by design; probing it would reject it."""
        from disteval.taskgen import GeneratedTask

        t = GeneratedTask("g", "s", "remove_required_input", RelationKind.REFUTED,
                          _payload(0), "v.sh", "n", {})
        v = validate_task(t, config=ValidationConfig(solver=lambda x: False))
        assert v["solvable"] is True
        assert "by design" in v["solvable_note"]

    def test_solvability_bias_is_warned_about(self):
        seeds = _seeds(2)
        rep = TaskGenerator(seeds, seed=0).generate(6)
        out = validate_batch(rep, {s.task_id: s for s in seeds},
                             ValidationConfig(solver=lambda t: False))
        assert "biasing the surviving set toward easier tasks" in out["solvability_bias_warning"]

    def test_all_gates_are_reported(self):
        seeds = _seeds(2)
        rep = TaskGenerator(seeds, seed=0).generate(4)
        validate_batch(rep, {s.task_id: s for s in seeds})
        for t in rep.tasks:
            assert set(GATES) <= set(t.validation) | {"solvable"}


class TestDiversity:
    def test_detects_a_collapsed_batch(self):
        seeds = _seeds()
        rep = TaskGenerator(seeds, seed=0).generate(
            10, seed_ids=["task_0"], max_per_seed=99
        )
        d = batch_diversity(rep.tasks)
        assert d["n_distinct_seeds"] == 1
        assert d["seed_concentration"] == pytest.approx(1.0)
        assert d["largest_seed_share"] == pytest.approx(1.0)

    def test_spread_batch_scores_lower_concentration(self):
        rep = TaskGenerator(_seeds(8), seed=0).generate(24, max_per_seed=3)
        d = batch_diversity(rep.tasks)
        assert d["seed_concentration"] < 0.5
        assert d["n_distinct_relations"] > 1

    def test_single_relation_batch_is_flagged(self):
        rep = TaskGenerator(_seeds(), relations=[get_relation("reorder_files")],
                            seed=0).generate(10)
        d = batch_diversity(rep.tasks)
        assert d["n_distinct_relations"] == 1
        assert d["relation_concentration"] == pytest.approx(1.0)

    def test_empty_batch(self):
        assert batch_diversity([])["n"] == 0


class TestFrontierBand:
    def test_band_validates_its_bounds(self):
        with pytest.raises(ValueError):
            FrontierBand(0.5, 0.2)
        with pytest.raises(ValueError):
            FrontierBand(-0.1, 0.5)

    def test_from_counts(self):
        b = FrontierBand.from_counts(1, 3, 8)
        assert b.lo == pytest.approx(0.125) and b.hi == pytest.approx(0.375)

    def test_band_probability_is_a_probability(self):
        for s in range(9):
            p = band_probability(binary_posterior(s, 8), MATH_CODE_BAND)
            assert 0.0 <= p <= 1.0

    def test_band_probability_peaks_inside_the_band(self):
        ps = [band_probability(binary_posterior(s, 16), MATH_CODE_BAND)
              for s in range(17)]
        peak = int(np.argmax(ps))
        assert MATH_CODE_BAND.lo <= peak / 16 <= MATH_CODE_BAND.hi

    def test_the_three_regions_sum_to_one(self):
        d = band_decision(binary_posterior(3, 8), MATH_CODE_BAND)
        assert d.p_in + d.p_below + d.p_above == pytest.approx(1.0, abs=1e-6)

    def test_narrow_band_is_unresolvable_at_small_k(self):
        """The finding that motivates this module."""
        assert not any(
            band_decision(binary_posterior(s, 8), MATH_CODE_BAND).label == "in_band"
            for s in range(9)
        ), "at K=8 a [1/8, 3/8] band should not be resolvable at 80% confidence"

    def test_becomes_resolvable_with_enough_runs(self):
        d = band_decision(binary_posterior(8, 32), MATH_CODE_BAND)
        assert d.label == "in_band" and d.p_in > 0.8

    def test_posterior_disagrees_with_the_plugin_rule(self):
        disagreements = sum(
            band_decision(binary_posterior(s, 8), MATH_CODE_BAND).disagrees_with_plugin
            for s in range(9)
        )
        assert disagreements > 0

    def test_plugin_error_is_large_near_the_band_edge(self):
        df = plugin_band_error(MATH_CODE_BAND, 8, true_ps=[0.45])
        assert float(df["p_mislabelled"].iloc[0]) > 0.3

    def test_plugin_error_is_small_far_from_the_band(self):
        df = plugin_band_error(MATH_CODE_BAND, 8, true_ps=[0.95])
        assert float(df["p_mislabelled"].iloc[0]) < 0.05

    def test_band_information_is_non_negative_and_vanishes_when_settled(self):
        for s, n in [(0, 4), (2, 8), (40, 80), (0, 80)]:
            assert band_information(binary_posterior(s, n), MATH_CODE_BAND) >= 0
        settled = band_information(binary_posterior(76, 80), MATH_CODE_BAND)
        unsettled = band_information(binary_posterior(2, 8), MATH_CODE_BAND)
        assert settled < unsettled

    def test_band_information_rejects_a_zero_horizon(self):
        with pytest.raises(ValueError):
            band_information(binary_posterior(2, 8), MATH_CODE_BAND, horizon=0)

    def test_expected_runs_is_zero_when_already_resolved(self):
        assert expected_runs_to_resolve_band(
            binary_posterior(8, 32), MATH_CODE_BAND
        ) == 0.0

    def test_expected_runs_is_positive_when_unresolved(self):
        r = expected_runs_to_resolve_band(binary_posterior(2, 8), MATH_CODE_BAND)
        assert r > 0 and np.isfinite(r)

    def test_swe_band_at_k3_is_a_coin_flip(self):
        """Their K=3 setting: 1/3 and 2/3 are both unresolved."""
        for s in (1, 2):
            d = band_decision(binary_posterior(s, 3), SWE_BAND)
            assert d.label == "unresolved"
            assert d.plugin_label == "in_band"

    def test_study_runs_and_reports_resolvability(self):
        from disteval.sim.studies import band_label_noise_study

        df = band_label_noise_study(ks=(8, 32))
        assert len(df) == 4
        assert not df[df["K"] == 8]["in_band_resolvable_at_80pct"].any()
        assert df[df["K"] == 32]["in_band_resolvable_at_80pct"].all()


class TestGenerateCLI:
    def _run(self, *args):
        return subprocess.run([sys.executable, "-m", "disteval", "generate", *args],
                              capture_output=True, text=True, cwd=str(ROOT), timeout=300)

    def test_generates_from_the_demo_dataset(self, tmp_path):
        out_path = tmp_path / "gen.jsonl"
        out = self._run("examples/tasks.json", "-n", "12", "-o", str(out_path))
        assert out.returncode == 0, out.stderr
        rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        assert 0 < len(rows) <= 12
        assert all(r["verifier"] and r["verifier_note"] for r in rows)
        assert all(not r["model_generated"] for r in rows)

    def test_targets_from_runs_when_supplied(self, tmp_path):
        out = self._run("examples/tasks.json", "--runs", "examples/runs.json",
                        "-n", "10", "-o", str(tmp_path / "g.jsonl"))
        assert out.returncode == 0, out.stderr
        assert "Targeting" in out.stdout

    def test_reports_diversity(self, tmp_path):
        out = self._run("examples/tasks.json", "-n", "15", "-o", str(tmp_path / "g.jsonl"))
        assert "diversity:" in out.stdout

    def test_relation_filter(self, tmp_path):
        p = tmp_path / "g.jsonl"
        out = self._run("examples/tasks.json", "-n", "8", "--relations",
                        "reorder_files", "-o", str(p))
        assert out.returncode == 0, out.stderr
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        assert {r["relation"] for r in rows} == {"reorder_files"}

    def test_unknown_relation_fails(self, tmp_path):
        out = self._run("examples/tasks.json", "--relations", "telepathy",
                        "-o", str(tmp_path / "g.jsonl"))
        assert out.returncode != 0
