"""Metamorphic relations: transformations whose effect on the answer is known.

The problem with generated eval tasks
-------------------------------------
A generated task is only an eval task if you can grade it. Free-form task
invention usually cannot be graded: the generator does not know the answer, and
using the generator's own answer as ground truth measures agreement with the
generator rather than competence. Worse, if the same model family generates and
is evaluated, the task distribution is the intersection of "what the model can
imagine" with "what it can do" -- a biased sample that will systematically
flatter the model.

Metamorphic testing sidesteps this. Instead of inventing a task and needing its
answer, you take a task whose grading you already trust and apply a
transformation whose effect on the correct answer is **known by construction**:

* reorder the files in a directory -> the answer does not change;
* insert a document irrelevant to the question -> the answer does not change;
* rename an entity consistently throughout -> the answer changes by the same
  renaming;
* scale every quantity by k -> a quantity answer scales by k.

The verifier transfers. You never needed to know the answer, only the *relation*
between the seed's answer and the variant's.

Why this is the right instrument for reliability specifically
-------------------------------------------------------------
An agent that solves a task but fails its paraphrase has an inconsistency the
seed task cannot reveal: it got the right answer for a reason that did not
survive a change that should not have mattered. That is precisely what this
framework calls unreliability, and repeated runs of the *same* task cannot
distinguish it from sampling noise. Metamorphic variants can.

What a relation must declare
----------------------------
``kind`` -- how the expected outcome relates to the seed's:

``invariant``   the correct answer is unchanged. Grading: apply the seed's
                verifier unmodified.
``equivariant`` the answer transforms in a stated, invertible way. Grading:
                invert the transform, then apply the seed's verifier.
``refuted``     the task becomes unsatisfiable and the correct behaviour is to
                say so. Grading: the agent must decline, not answer.

Anything that does not fit one of these is **not a metamorphic relation** and
does not belong here, because its verifier would not transfer -- which is the
whole point.

``stresses`` names the failure modes and criteria a relation is expected to
probe, which is what lets :mod:`disteval.taskgen.generator` aim generation at a
measured weakness rather than scattering it.

No model API is required. Every relation below is structural. An optional
model-backed paraphraser is a plug point (:class:`CallableRelation`), never a
dependency.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

__all__ = [
    "RelationKind",
    "MetamorphicRelation",
    "CallableRelation",
    "RELATIONS",
    "register_relation",
    "get_relation",
    "relations_stressing",
]


class RelationKind:
    INVARIANT = "invariant"
    EQUIVARIANT = "equivariant"
    REFUTED = "refuted"


@dataclass(frozen=True)
class MetamorphicRelation:
    """One transformation, plus how it changes the expected answer.

    Attributes
    ----------
    name:
        Stable identifier, recorded in every generated task's provenance.
    kind:
        One of :class:`RelationKind`. Determines how the seed's verifier is
        reused, which is the entire justification for the generated task being
        gradeable.
    transform:
        ``(task_payload, rng) -> (new_payload, evidence)``. ``evidence`` records
        what actually changed, so a failure can be traced to the specific edit.
    stresses:
        Failure modes (from :mod:`disteval.diagnosis.taxonomy`) and rubric
        criterion name-fragments this relation is expected to probe.
    answer_map / answer_unmap:
        Required for ``equivariant`` relations: how to map an answer from seed
        space to variant space and back. ``None`` for the other kinds.
    preserves_difficulty:
        Whether the variant is intended to be about as hard as the seed. When
        False, a difficulty shift is an expected confound and is flagged on the
        generated task rather than silently absorbed.
    """

    name: str
    kind: str
    description: str
    transform: Callable[[Mapping, object], tuple[dict, dict]]
    stresses: tuple[str, ...] = ()
    answer_map: Optional[Callable[[object], object]] = None
    answer_unmap: Optional[Callable[[object], object]] = None
    preserves_difficulty: bool = True

    def __post_init__(self) -> None:
        if self.kind not in (RelationKind.INVARIANT, RelationKind.EQUIVARIANT,
                             RelationKind.REFUTED):
            raise ValueError(f"unknown relation kind {self.kind!r}")
        if self.kind == RelationKind.EQUIVARIANT and (
            self.answer_map is None or self.answer_unmap is None
        ):
            raise ValueError(
                f"equivariant relation {self.name!r} must supply answer_map and "
                "answer_unmap, or its verifier cannot transfer"
            )

    def apply(self, payload: Mapping, rng) -> tuple[dict, dict]:
        """Run the transformation. Returns ``(new_payload, evidence)``."""
        return self.transform(payload, rng)


RELATIONS: dict[str, MetamorphicRelation] = {}


def register_relation(rel: MetamorphicRelation) -> MetamorphicRelation:
    """Add a relation. Benchmarks extend the set this way rather than editing here."""
    RELATIONS[rel.name] = rel
    return rel


def get_relation(name: str) -> MetamorphicRelation:
    if name not in RELATIONS:
        raise ValueError(f"unknown relation {name!r}; have {sorted(RELATIONS)}")
    return RELATIONS[name]


def relations_stressing(*targets: str) -> list[MetamorphicRelation]:
    """Relations expected to probe any of ``targets`` (failure modes or criteria).

    Matching is by substring on both sides, so ``"retrieve"`` finds a relation
    declaring ``"r_retrieve"`` and vice versa. Aiming generation at a measured
    weakness is the whole reason relations declare what they stress.
    """
    want = [t.lower() for t in targets]
    out = []
    for rel in RELATIONS.values():
        for s in rel.stresses:
            sl = s.lower()
            if any(w in sl or sl in w for w in want):
                out.append(rel)
                break
    return out


# --------------------------------------------------------------------------- #
# Built-in structural relations. None requires a model.                       #
# --------------------------------------------------------------------------- #
def _files(payload: Mapping) -> list[str]:
    return list(payload.get("files") or [])


def _t_reorder_files(payload: Mapping, rng) -> tuple[dict, dict]:
    files = _files(payload)
    if len(files) < 2:
        raise ValueError("reorder_files needs at least two files")
    new = list(files)
    for _ in range(8):
        rng.shuffle(new)
        if new != files:
            break
    else:
        raise ValueError("could not produce a distinct ordering")
    out = dict(payload)
    out["files"] = new
    return out, {"from": files, "to": new}


def _t_add_distractor(payload: Mapping, rng) -> tuple[dict, dict]:
    files = _files(payload)
    n = int(rng.integers(1, 4))
    added = [f"distractor_{int(rng.integers(1000, 9999))}.md" for _ in range(n)]
    out = dict(payload)
    pos = int(rng.integers(0, len(files) + 1)) if files else 0
    out["files"] = files[:pos] + added + files[pos:]
    out["distractors"] = list(out.get("distractors") or []) + added
    return out, {"added": added, "inserted_at": pos}


def _t_rename_entity(payload: Mapping, rng) -> tuple[dict, dict]:
    """Consistent renaming: equivariant, and the map is the renaming itself."""
    ents = list(payload.get("entities") or [])
    if not ents:
        raise ValueError("rename_entity needs declared entities")
    old = str(ents[int(rng.integers(0, len(ents)))])
    new = f"{old}_{int(rng.integers(100, 999))}"
    out = dict(payload)
    out["entities"] = [new if e == old else e for e in ents]
    if payload.get("instruction"):
        out["instruction"] = re.sub(
            rf"\b{re.escape(old)}\b", new, str(payload["instruction"])
        )
    out["files"] = [
        re.sub(rf"\b{re.escape(old)}\b", new, f) for f in _files(payload)
    ]
    out["_rename"] = {"old": old, "new": new}
    return out, {"old": old, "new": new}


def _t_scale_quantities(payload: Mapping, rng) -> tuple[dict, dict]:
    """Scale every declared quantity by k: a quantity answer scales by k."""
    q = dict(payload.get("quantities") or {})
    if not q:
        raise ValueError("scale_quantities needs declared quantities")
    k = float(rng.choice([2.0, 10.0, 0.5]))
    out = dict(payload)
    out["quantities"] = {name: float(v) * k for name, v in q.items()}
    out["_scale"] = k
    return out, {"factor": k, "n_quantities": len(q)}


def _t_paraphrase_template(payload: Mapping, rng) -> tuple[dict, dict]:
    """Template-based instruction rewording. Structural, no model involved.

    Deliberately conservative: it rewraps the instruction rather than rewriting
    it, because a transformation that changed the task's meaning would silently
    break the verifier -- the failure mode this whole module exists to avoid.
    """
    instr = str(payload.get("instruction") or "").strip()
    if not instr:
        raise ValueError("paraphrase_template needs an instruction")
    frames = [
        "Your task: {}",
        "Please complete the following. {}",
        "Objective — {}",
        "Carry out this request: {}",
        "{} Report the result when finished.",
    ]
    frame = frames[int(rng.integers(0, len(frames)))]
    new = frame.format(instr)
    if new == instr:
        raise ValueError("paraphrase produced an identical instruction")
    out = dict(payload)
    out["instruction"] = new
    return out, {"frame": frame}


def _t_remove_required_input(payload: Mapping, rng) -> tuple[dict, dict]:
    """Delete an input the task needs: the correct behaviour becomes refusal."""
    req = list(payload.get("required_inputs") or [])
    if not req:
        raise ValueError("remove_required_input needs declared required_inputs")
    victim = str(req[int(rng.integers(0, len(req)))])
    out = dict(payload)
    out["required_inputs"] = [r for r in req if r != victim]
    out["files"] = [f for f in _files(payload) if victim not in f]
    out["_removed_input"] = victim
    return out, {"removed": victim}


for _rel in (
    MetamorphicRelation(
        name="reorder_files",
        kind=RelationKind.INVARIANT,
        description="Present the same files in a different order.",
        transform=_t_reorder_files,
        stresses=("retrieval", "memory", "state_tracking", "r_retrieve"),
    ),
    MetamorphicRelation(
        name="add_distractor",
        kind=RelationKind.INVARIANT,
        description="Insert documents irrelevant to the question.",
        transform=_t_add_distractor,
        stresses=("retrieval", "tool_selection", "reasoning", "r_retrieve"),
        # More material to sift is genuinely harder; say so rather than pretend.
        preserves_difficulty=False,
    ),
    MetamorphicRelation(
        name="paraphrase_instruction",
        kind=RelationKind.INVARIANT,
        description="Reword the instruction without changing what it asks.",
        transform=_t_paraphrase_template,
        stresses=("planning", "reasoning", "r_report"),
    ),
    MetamorphicRelation(
        name="rename_entity",
        kind=RelationKind.EQUIVARIANT,
        description="Rename an entity consistently; the answer renames with it.",
        transform=_t_rename_entity,
        stresses=("memory", "state_tracking", "reasoning", "r_compute"),
        answer_map=lambda a: a,
        answer_unmap=lambda a: a,
    ),
    MetamorphicRelation(
        name="scale_quantities",
        kind=RelationKind.EQUIVARIANT,
        description="Multiply every quantity by k; a quantity answer scales by k.",
        transform=_t_scale_quantities,
        stresses=("reasoning", "tool_execution", "r_compute"),
        answer_map=lambda a: a,
        answer_unmap=lambda a: a,
    ),
    MetamorphicRelation(
        name="remove_required_input",
        kind=RelationKind.REFUTED,
        description="Remove an input the task needs; correct behaviour is to decline.",
        transform=_t_remove_required_input,
        stresses=("verification", "recovery", "synthesis", "r_verify"),
        preserves_difficulty=False,
    ),
):
    register_relation(_rel)


class CallableRelation(MetamorphicRelation):
    """Adapter for a model-backed transformation, e.g. an LLM paraphraser.

    Provided as a seam, never a dependency: nothing in this package imports a
    provider. A model-generated variant carries a strictly weaker guarantee than
    a structural one -- the model may change the task's meaning and thereby break
    the inherited verifier -- so :mod:`disteval.taskgen.validate` treats
    model-transformed variants as requiring explicit human or programmatic
    confirmation before they count toward any reported metric.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[str], str],
        *,
        kind: str = RelationKind.INVARIANT,
        description: str = "model-backed transformation",
        stresses: Sequence[str] = (),
        field_name: str = "instruction",
    ):
        def _transform(payload: Mapping, rng) -> tuple[dict, dict]:
            before = str(payload.get(field_name) or "")
            if not before:
                raise ValueError(f"{name} needs a non-empty {field_name!r}")
            after = fn(before)
            if not after or after == before:
                raise ValueError(f"{name} produced no change")
            out = dict(payload)
            out[field_name] = after
            out["_model_generated"] = True
            return out, {"field": field_name, "before": before[:200],
                         "after": after[:200]}

        super().__init__(
            name=name, kind=kind, description=description, transform=_transform,
            stresses=tuple(stresses),
            answer_map=(lambda a: a) if kind == RelationKind.EQUIVARIANT else None,
            answer_unmap=(lambda a: a) if kind == RelationKind.EQUIVARIANT else None,
        )
