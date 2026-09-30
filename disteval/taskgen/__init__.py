"""Metamorphic eval-task generation: variants whose verifier transfers by construction.

See :mod:`disteval.taskgen.relations` for why free-form task synthesis is avoided
and what guarantee this approach offers instead.
"""
from .generator import (
    GeneratedTask,
    GenerationReport,
    GenerationTarget,
    SeedTask,
    TaskGenerator,
)
from .relations import (
    RELATIONS,
    CallableRelation,
    MetamorphicRelation,
    RelationKind,
    get_relation,
    register_relation,
    relations_stressing,
)
from .validate import (
    GATES,
    ValidationConfig,
    batch_diversity,
    validate_batch,
    validate_task,
)

__all__ = [
    "RelationKind", "MetamorphicRelation", "CallableRelation", "RELATIONS",
    "register_relation", "get_relation", "relations_stressing",
    "SeedTask", "GeneratedTask", "GenerationTarget", "GenerationReport",
    "TaskGenerator",
    "ValidationConfig", "validate_task", "validate_batch", "batch_diversity",
    "GATES",
]
