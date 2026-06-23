"""
disteval.test_suite_parser — Parse task test.sh files into checkpoint reward specs.

OVERVIEW
────────
Many disteval tasks award partial credit through a sequence of test-script
checkpoints. This module parses `tasks/<task>/tests/test.sh` and returns an
ordered list of `CheckpointSpec` objects describing each checkpoint's reward
weight, its condition source, and a human-readable label.

The parser is deliberately tolerant: it recognises the common patterns used in
this repository's bash test scripts and falls back to reasonable defaults when a
checkpoint cannot be fully annotated.

Example output for `tasks/medium-2/tests/test.sh`:

    [
        CheckpointSpec(index=0, task_name="medium-2", description="Validate JSON",
                       reward_weight=0.10, score_increment=10, total_score=100),
        CheckpointSpec(index=1, task_name="medium-2", description="Check total_eligible_users",
                       reward_weight=0.25, score_increment=25, total_score=100),
        ...
    ]
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckpointSpec:
    """One scoreable checkpoint parsed from a test.sh file."""

    index: int
    task_name: str
    description: str
    reward_weight: float
    score_increment: int
    total_score: int
    condition_source: str
    checkpoint_id: str


# Regex patterns used by the parser.
_SCORE_INCREMENT_RE = re.compile(
    r"SCORE=\$\(\(SCORE\s*\+\s*(\d+)\)\)"  # SCORE=$((SCORE + 10))
    r"|SCORE=\$\[SCORE\s*\+\s*(\d+)\]"      # SCORE=$[SCORE + 10]
    r"|SCORE\s*=\s*\$\(\s*expr\s+\$SCORE\s*\+\s*(\d+)\s*\)",  # SCORE=$(expr $SCORE + 10)
    re.IGNORECASE,
)
_FINAL_SCORE_RE = re.compile(
    r'python3\s+-c\s+"print\(\$SCORE\s*/\s*(\d+)\.?\d*\)"'
    r"|python3\s+-c\s+'print\(\$SCORE\s*/\s*(\d+)\.?\d*\)'"
    r"|echo\s+\"\$SCORE\s*/\s*(\d+)\"",
    re.IGNORECASE,
)


def _extract_description(lines_before: list[str]) -> str:
    """Build a checkpoint description from preceding comments and code."""
    # Collect trailing comment lines immediately before the SCORE increment.
    comments: list[str] = []
    for line in reversed(lines_before):
        stripped = line.strip()
        if stripped.startswith("#"):
            comments.insert(0, stripped.lstrip("#").strip())
        elif stripped == "":
            # Continue scanning past blank lines to grab the nearest comment block.
            continue
        else:
            break

    if comments:
        # Drop leading words like "Check", "Validate", "Test" for consistency.
        text = " ".join(comments)
        text = re.sub(r"^(Check|Validate|Test)\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
        text = text.strip()
        if text:
            # Truncate to a reasonable label.
            if len(text) > 80:
                text = text[:77] + "..."
            return text

    # Fallback: try to find a Python assertion key or a descriptive variable.
    source = "\n".join(lines_before[-3:]) if lines_before else ""
    key_match = re.search(r"assert\s+['\"]([^'\"]+)['\"]\s+in\s+d", source)
    if key_match:
        return f"Key {key_match.group(1)} present"
    key_match = re.search(r"assert\s+d\[[\'\"]([^\'\"]+)[\'\"]\]", source)
    if key_match:
        return f"Field {key_match.group(1)} correct"
    key_match = re.search(r"assert\s+d\['([^']+)'\]\['([^']+)'\]", source)
    if key_match:
        return f"{key_match.group(1)}.{key_match.group(2)} correct"

    return "Checkpoint"


def _extract_condition_source(lines_before: list[str]) -> str:
    """Return the raw shell/python block that gates the checkpoint."""
    # We collect lines back until we hit a blank line, a previous SCORE increment,
    # or an early-exit block.
    source_lines: list[str] = []
    for line in reversed(lines_before):
        stripped = line.strip()
        if not stripped:
            break
        if _SCORE_INCREMENT_RE.search(stripped):
            break
        if stripped.startswith("if [ ! -f") or stripped.startswith("if [ ! -d"):
            break
        source_lines.insert(0, line)
    return "\n".join(source_lines).strip()


def _extract_total_score(text: str) -> int:
    """Find the denominator used in the final reward.txt write."""
    # The final line is typically: python3 -c "print($SCORE / 100.0)" > /logs/verifier/reward.txt
    for line in text.splitlines():
        match = _FINAL_SCORE_RE.search(line)
        if match:
            # Group 1 from the first alternative, group 4 from the second, group 7 from the third.
            for g in match.groups():
                if g is not None:
                    return int(g)
    # Fallback: sum of all increments in the file.
    return sum(int(m.group(1) or m.group(2) or m.group(3))
               for m in _SCORE_INCREMENT_RE.finditer(text))


def parse_test_suite(
    test_sh_path: str | Path,
    task_name: str | None = None,
) -> list[CheckpointSpec]:
    """
    Parse a disteval test.sh file and return an ordered list of CheckpointSpec.

    Parameters
    ----------
    test_sh_path : str | Path
        Path to the test.sh file.
    task_name : str | None
        Optional task name override. If not provided, the task name is derived
        from the parent directory structure (e.g., tasks/medium-2/tests/test.sh
        -> medium-2).

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If no SCORE variable is found (no checkpoints or early-exit structure).
    """
    path = Path(test_sh_path)
    if not path.exists():
        raise FileNotFoundError(f"Test script not found: {path}")

    text = path.read_text()
    lines = text.splitlines()

    if task_name is None:
        # Derive from tasks/<task_name>/tests/test.sh
        parts = path.parts
        if "tasks" in parts:
            idx = parts.index("tasks")
            if idx + 1 < len(parts):
                task_name = parts[idx + 1]
        if task_name is None:
            task_name = path.parent.parent.name or "unknown"

    total_score = _extract_total_score(text)
    if total_score == 0:
        raise ValueError(f"No SCORE denominator or increments found in {path}")

    checkpoints: list[CheckpointSpec] = []
    for idx, line in enumerate(lines):
        match = _SCORE_INCREMENT_RE.search(line)
        if not match:
            continue
        increment = int(next(g for g in match.groups() if g is not None))
        lines_before = lines[:idx]
        description = _extract_description(lines_before)
        condition_source = _extract_condition_source(lines_before)
        reward_weight = increment / total_score
        checkpoint_id = f"{task_name}::phase-{len(checkpoints)}"
        checkpoints.append(
            CheckpointSpec(
                index=len(checkpoints),
                task_name=task_name,
                description=description,
                reward_weight=reward_weight,
                score_increment=increment,
                total_score=total_score,
                condition_source=condition_source,
                checkpoint_id=checkpoint_id,
            )
        )

    if not checkpoints:
        # All-or-nothing scripts have no increments but a denominator. Represent as one checkpoint.
        checkpoints.append(
            CheckpointSpec(
                index=0,
                task_name=task_name,
                description="All-or-nothing task completion",
                reward_weight=1.0,
                score_increment=total_score,
                total_score=total_score,
                condition_source="",
                checkpoint_id=f"{task_name}::phase-0",
            )
        )

    return checkpoints


def parse_all_tasks(
    tasks_dir: str | Path = "tasks",
) -> dict[str, list[CheckpointSpec]]:
    """
    Walk tasks_dir and parse every test.sh found.

    Returns {task_name: [CheckpointSpec, ...]} for all parseable tasks. Tasks whose
    test.sh cannot be parsed are silently skipped and logged at WARNING level.
    """
    import logging

    logger = logging.getLogger(__name__)
    result: dict[str, list[CheckpointSpec]] = {}
    tasks_path = Path(tasks_dir)
    if not tasks_path.is_dir():
        return result

    for test_sh in tasks_path.rglob("tests/test.sh"):
        try:
            specs = parse_test_suite(test_sh)
            if specs:
                result[specs[0].task_name] = specs
        except (FileNotFoundError, ValueError, Exception) as exc:
            logger.warning("Skipping %s: %s", test_sh, exc)
    return result


def checkpoint_weights(specs: list[CheckpointSpec]) -> list[float]:
    """Return just the reward weights in checkpoint order."""
    return [s.reward_weight for s in specs]
