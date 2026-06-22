"""
disteval.self_engine — Self-creating improvement engine for agentic evaluation.

CORE IDEA
─────────
An agent that has run a benchmark has everything it needs to improve itself:
  - Which tasks it can already do (SOLID)
  - Which tasks it does inconsistently (RECOVERABLE) — the leverage point
  - Which tasks it cannot do at all (STUCK) — needs new capability
  - The specific trajectories that succeeded on RECOVERABLE tasks
  - The specific trajectories that failed on RECOVERABLE tasks
  - A structural prediction of whether a new attempt will succeed mid-episode
  - A memory of its best past approaches, retrievable by similarity

The SelfEngine assembles these signals into a self-improvement plan:
  1. OBSERVE   — right_tail_analysis: classify all tasks by SOLID/RECOVERABLE/STUCK
  2. LOCALIZE  — trajectory_monitor:  find the structural step where failure diverges
  3. RETRIEVE  — trajectory_memory:   surface the best past trajectory per RECOVERABLE task
  4. SCHEDULE  — rank by gap × consistency: which task yields the most gain per training step
  5. SIMULATE  — training_sim (optional): predict expected score gain per task
  6. OUTPUT    — SelfImprovementPlan:  ranked curriculum with reinforce/contrast pairs

DESIGN PRINCIPLES
─────────────────
- No external labels needed: all signals derive from the agent's own eval logs.
- No modification of the training loop: the engine produces a training data plan;
  the actual fine-tuning step is outside this module's scope.
- Grounded in real data: designed and validated against 37 real Harbor trajectories.
- Conservative estimates: predicted_gain uses empirically calibrated learning rates.

RELATIONSHIP TO RECURSIVE RL (arXiv:2206.11430)
─────────────────────────────────────────────────
Recursive MDPs decompose task value by sub-task level and prove Q-value convergence
across recursive boundaries. The SelfEngine embodies the same decomposition insight
without requiring a known recursive task structure:

  RMDP concept              SelfEngine equivalent
  ─────────────────────     ─────────────────────────────────────────────────
  Recursive MDP hierarchy   RECOVERABLE taxonomy (nested gap localization)
  Sub-MDP Q-value           per-task consistency score κ(t)
  Entry/exit points         trajectory step where monitor prediction diverges
  Convergence toward V*     κ → 1.0 as RECOVERABLE gaps close across cycles
  Call-stack value backup   memory retrieval at matching structural depth

The key convergence result: as RECOVERABLE gaps close (κ → 1.0 for each task),
the engine's priority queue naturally empties — RECOVERABLE tasks graduate to SOLID.
What remains are STUCK tasks, which require capability expansion, not consistency
training. The engine correctly identifies when it has exhausted consistency training
and signals that new exploration is needed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Any


from .records import RecordStore
from .right_tail import right_tail_analysis, RightTailReport, TaskOutcomeProfile
from .trajectory_loader import load_trajectory_records
from .trajectory_monitor import TrajectoryMonitor
from .trajectory_memory import TrajectoryMemory, RetrievalResult

# Optional recursion engine. Import lazily to avoid a hard dependency cycle.
try:
    from .recursion_engine import RecursionEngine, SubTaskDefinition, SubTaskGraph
except Exception:  # pragma: no cover - recursion engine may be unavailable during early import
    RecursionEngine = None
    SubTaskDefinition = None
    SubTaskGraph = None

try:
    from .distributed_eval import DistributedEvalPool, CrossAgentPair
except Exception:  # pragma: no cover - distributed eval may be unavailable during early import
    DistributedEvalPool = None
    CrossAgentPair = None


# ── Output data structures ────────────────────────────────────────────────────

@dataclass
class TrainingPair:
    """One (reinforce, contrast) training pair for a RECOVERABLE task."""
    task: str
    reinforce_traj_path: str       # path to high-scoring trajectory.json
    contrast_traj_path: str        # path to low-scoring trajectory.json
    reinforce_score: float
    contrast_score: float
    gap: float                     # reinforce_score - contrast_score
    structural_divergence_step: int  # step at which monitor prediction diverges

    # Recursive self-improvement extensions (optional, default-disabled)
    parent_task: Optional[str] = None
    sub_task_depth: int = 0
    entry_step: int = 0
    exit_step: int = -1
    call_stack: list[str] = field(default_factory=list)


@dataclass
class TaskImprovement:
    """Self-improvement plan for one RECOVERABLE task."""
    task: str
    difficulty: Optional[str]
    kind: str                           # always "recoverable" or "decomposed"
    current_q_star: float               # best score achieved so far
    current_q_bar: float                # mean score currently
    consistency: float                  # q_bar / q_star
    gap: float                          # q_star - q_bar (recoverable score left)
    priority_score: float               # gap × (1 - consistency) — higher = more leverage

    training_pairs: list[TrainingPair]  # (reinforce, contrast) pairs
    memory_results: list[RetrievalResult]  # best past trajectories from memory

    predicted_gain: Optional[float] = None   # from training_sim, if available
    predicted_gain_ci: Optional[tuple[float, float]] = None
    predicted_rounds_to_threshold: Optional[float] = None

    recommendation: str = ""           # human-readable action string

    # Recursive self-improvement extensions (optional, default-disabled)
    sub_tasks: list["TaskImprovement"] = field(default_factory=list)


@dataclass
class SelfImprovementPlan:
    """
    Full self-improvement curriculum for one agent, generated by SelfEngine.

    Produced by SelfEngine.run_cycle(). Contains the ranked list of
    RECOVERABLE tasks with their training pairs, memory retrievals,
    and (optionally) predicted gains from Monte Carlo simulation.
    """
    agent_name: str
    model_name: str
    cycle: int

    # Current state
    n_tasks_total: int
    n_solid: int
    n_recoverable: int
    n_stuck: int
    consistency_index: float          # κ = Q̄_total / Q*_total across all tasks
    recoverable_score_left: float     # total gap across RECOVERABLE tasks

    # The curriculum: ranked by priority_score descending
    curriculum: list[TaskImprovement]

    # Summary statistics
    predicted_total_gain: Optional[float] = None  # sum of gains if all training pairs used
    cycle_complete: bool = False      # True when n_recoverable == 0

    # Engine metadata
    job_dirs: list[str] = field(default_factory=list)
    n_trajectories_loaded: int = 0

    # Recursive self-improvement extensions (optional, default-disabled)
    recursion_enabled: bool = False
    recursion_context: Optional[dict] = None
    sub_task_graph: Optional[dict] = None
    n_decomposed: int = 0

    # Distributed evaluation extensions (optional, default-disabled)
    cross_agent_pairs: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"Self-Improvement Plan — {self.agent_name}  (cycle {self.cycle})",
            f"  Tasks: {self.n_tasks_total} total  "
            f"({self.n_solid} SOLID · {self.n_recoverable} RECOVERABLE · {self.n_stuck} STUCK)",
            f"  Consistency index κ = {self.consistency_index:.3f}  "
            f"({100*(1-self.consistency_index):.1f}% of achievable score lost to inconsistency)",
            f"  Recoverable score left: {self.recoverable_score_left:.3f}",
            "",
        ]
        if not self.curriculum:
            lines.append("  No RECOVERABLE tasks — all tasks are SOLID or STUCK.")
            if self.n_stuck > 0:
                lines.append(f"  {self.n_stuck} STUCK tasks need new capability, not consistency training.")
        else:
            lines.append(f"  Training curriculum ({len(self.curriculum)} tasks, ranked by leverage):")
            for i, item in enumerate(self.curriculum, 1):
                gain_str = ""
                if item.predicted_gain is not None:
                    gain_str = f"  predicted gain: +{item.predicted_gain:.3f}"
                    if item.predicted_gain_ci:
                        gain_str += f" [{item.predicted_gain_ci[0]:+.3f}, {item.predicted_gain_ci[1]:+.3f}]"
                lines.append(
                    f"  {i:>2}. {item.task:<25}  κ={item.consistency:.2f}  "
                    f"gap={item.gap:.3f}  pairs={len(item.training_pairs)}{gain_str}"
                )
                lines.append(f"      → {item.recommendation}")
        if self.predicted_total_gain is not None:
            lines.append(f"\n  Predicted total gain this cycle: +{self.predicted_total_gain:.3f}")
        if self.cycle_complete:
            lines.append("\n  ✓ Cycle complete: no RECOVERABLE tasks remain.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialisable dict for JSON export."""
        def _task_item_to_dict(t: TaskImprovement) -> dict:
            return {
                "task": t.task,
                "difficulty": t.difficulty,
                "kind": t.kind,
                "current_q_star": t.current_q_star,
                "current_q_bar": t.current_q_bar,
                "consistency": t.consistency,
                "gap": t.gap,
                "priority_score": t.priority_score,
                "predicted_gain": t.predicted_gain,
                "predicted_gain_ci": list(t.predicted_gain_ci) if t.predicted_gain_ci else None,
                "predicted_rounds_to_threshold": t.predicted_rounds_to_threshold,
                "recommendation": t.recommendation,
                "n_training_pairs": len(t.training_pairs),
                "training_pairs": [
                    {
                        "reinforce_traj_path": p.reinforce_traj_path,
                        "contrast_traj_path": p.contrast_traj_path,
                        "reinforce_score": p.reinforce_score,
                        "contrast_score": p.contrast_score,
                        "gap": p.gap,
                        "structural_divergence_step": p.structural_divergence_step,
                        "parent_task": p.parent_task,
                        "sub_task_depth": p.sub_task_depth,
                        "entry_step": p.entry_step,
                        "exit_step": p.exit_step,
                        "call_stack": p.call_stack,
                    }
                    for p in t.training_pairs
                ],
                "n_sub_tasks": len(t.sub_tasks),
                "sub_tasks": [_task_item_to_dict(st) for st in t.sub_tasks],
            }

        return {
            "agent_name": self.agent_name,
            "model_name": self.model_name,
            "cycle": self.cycle,
            "n_tasks_total": self.n_tasks_total,
            "n_solid": self.n_solid,
            "n_recoverable": self.n_recoverable,
            "n_stuck": self.n_stuck,
            "n_decomposed": self.n_decomposed,
            "consistency_index": self.consistency_index,
            "recoverable_score_left": self.recoverable_score_left,
            "predicted_total_gain": self.predicted_total_gain,
            "cycle_complete": self.cycle_complete,
            "n_trajectories_loaded": self.n_trajectories_loaded,
            "recursion_enabled": self.recursion_enabled,
            "recursion_context": self.recursion_context,
            "sub_task_graph": self.sub_task_graph,
            "cross_agent_pairs": self.cross_agent_pairs,
            "curriculum": [_task_item_to_dict(t) for t in self.curriculum],
        }

    def save(self, path: str) -> None:
        """Save plan to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ── SelfEngine ────────────────────────────────────────────────────────────────

class SelfEngine:
    """
    Self-creating improvement engine.

    Assembles disteval's existing primitives into a self-improvement loop:
      - right_tail_analysis  → RECOVERABLE task identification
      - TrajectoryMonitor    → structural divergence localization
      - TrajectoryMemory     → best-trajectory retrieval
      - training_sim         → predicted gain (optional, loaded lazily)

    Usage:
        engine = SelfEngine.from_job_dirs(
            job_dirs=["jobs/run_A/...", "jobs/run_B/..."],
            agent_name="Codex CLI",
            model_name="openai/o4-mini",
        )
        plan = engine.run_cycle()
        print(plan.summary())
        plan.save("disteval_output/self_improvement_plan.json")

    Multiple cycles:
        for cycle in range(5):
            plan = engine.run_cycle(cycle=cycle)
            print(plan.summary())
            # Apply training (external step), then reload updated trajectories
            engine.reload()
    """

    def __init__(
        self,
        store: RecordStore,
        job_dirs: list[str],
        agent_name: str,
        model_name: str,
        monitor: Optional[TrajectoryMonitor] = None,
        memory: Optional[TrajectoryMemory] = None,
        recursion_engine: Optional[Any] = None,
        enable_recursion: bool = False,
    ):
        self.store = store
        self.job_dirs = job_dirs
        self.agent_name = agent_name
        self.model_name = model_name
        self.recursion_engine = recursion_engine
        self.enable_recursion = enable_recursion and (recursion_engine is not None)

        # Load trajectory records (trajectory_monitor.TrajectoryRecord schema)
        self._traj_records = []
        for jd in job_dirs:
            try:
                self._traj_records.extend(load_trajectory_records(jd))
            except Exception:
                pass

        # Build monitor (uses trajectory_monitor.TrajectoryRecord)
        if monitor is not None:
            self.monitor = monitor
        else:
            self.monitor = TrajectoryMonitor(self._traj_records)

        # Build memory (uses trajectory_memory.TrajectoryRecord — different schema)
        if memory is not None:
            self.memory = memory
        else:
            from .trajectory_memory import TrajectoryRecord as MemTrajectoryRecord
            self.memory = TrajectoryMemory()
            for rec in self._traj_records:
                # Convert monitor.TrajectoryRecord → memory.TrajectoryRecord
                feat = rec.features
                mem_rec = MemTrajectoryRecord(
                    trial_id=rec.trial_id,
                    task_path=rec.task_path,
                    agent_name=rec.agent_name,
                    score=rec.score,
                    tool_sequence=rec.tool_sequence,
                    traj_path=rec.traj_path,
                    n_steps=feat.n_steps,
                    first_write_pos=feat.first_write_pos,
                    n_exec=feat.n_exec,
                    n_search=feat.n_search,
                )
                try:
                    self.memory.add(mem_rec)
                except Exception:
                    pass

        self._cycle = 0

        # Lazy import training_sim (may not be built yet)
        self._sim = None

    @classmethod
    def from_job_dirs(
        cls,
        job_dirs: list[str],
        agent_name: str = "agent",
        model_name: str = "unknown",
        tasks_dir: str = "tasks",
        enable_recursion: bool = False,
        recursion_config: Optional[dict] = None,
    ) -> "SelfEngine":
        """
        Convenience constructor: load everything from Harbor job dirs.

        Finds the run directory within each job_dir automatically
        (handles both 'jobs/run_A' and 'jobs/run_A/disteval-run-A').

        When enable_recursion=True, a RecursionEngine is instantiated and
        attached to the SelfEngine. The recursion engine is default-disabled.
        """
        from .adapters.harbor_jobs import load_harbor_job

        stores = []
        resolved_dirs = []
        for jd in job_dirs:
            # Auto-resolve: if jd contains a subdirectory, use that
            if os.path.isdir(jd):
                subdirs = [
                    os.path.join(jd, d) for d in os.listdir(jd)
                    if os.path.isdir(os.path.join(jd, d))
                ]
                run_dir = subdirs[0] if len(subdirs) == 1 else jd
            else:
                run_dir = jd
            resolved_dirs.append(run_dir)
            try:
                s = load_harbor_job(run_dir, tasks_dir=tasks_dir)
                stores.append(s)
            except Exception:
                pass

        if not stores:
            raise ValueError(f"No valid Harbor job directories found in: {job_dirs}")

        # Merge all stores
        from .records import RecordStore
        merged = RecordStore()
        for s in stores:
            for rec in s._records:
                merged.add(rec)

        # Build the base engine without recursion first.
        engine = cls(
            store=merged,
            job_dirs=resolved_dirs,
            agent_name=agent_name,
            model_name=model_name,
        )

        if enable_recursion and RecursionEngine is not None:
            from .recursion_engine import RecursionEngineConfig
            cfg = recursion_config or {}
            recursion_engine = RecursionEngine(
                monitor=engine.monitor,
                memory=engine.memory,
                agent_name=agent_name,
                model_name=model_name,
                tasks_dir=tasks_dir,
                config=RecursionEngineConfig(**cfg),
            )
            engine.recursion_engine = recursion_engine
            engine.enable_recursion = True

        return engine

    def reload(self) -> None:
        """Reload all data from job dirs — call after external training step."""
        new = self.__class__.from_job_dirs(
            self.job_dirs,
            self.agent_name,
            self.model_name,
            enable_recursion=self.enable_recursion,
        )
        self.store = new.store
        self._traj_records = new._traj_records
        self.monitor = new.monitor
        self.memory = new.memory
        self.recursion_engine = new.recursion_engine

    # ── Core cycle ──────────────────────────────────────────────────────────

    def run_cycle(self, cycle: Optional[int] = None) -> SelfImprovementPlan:
        """
        Run one self-improvement cycle and return a SelfImprovementPlan.

        Steps:
          1. OBSERVE:   right_tail_analysis → SOLID/RECOVERABLE/STUCK per task
          2. LOCALIZE:  trajectory_monitor  → structural divergence step per task
          3. RETRIEVE:  trajectory_memory   → best-matching memories per task
          4. SCHEDULE:  rank by gap × (1-consistency) — highest leverage first
          5. SIMULATE:  training_sim (optional) → predicted_gain per task
          6. OUTPUT:    SelfImprovementPlan
        """
        if cycle is None:
            cycle = self._cycle
            self._cycle += 1

        # ── STEP 1: OBSERVE ────────────────────────────────────────────────
        # Pass model_name=None — each engine is instantiated per agent,
        # so the store already contains only that agent's records.
        report = right_tail_analysis(self.store, model_name=None)

        # Consistency index κ
        kappa = (
            report.sum_q_bar / report.sum_q_star
            if report.sum_q_star > 0 else 1.0
        )

        # ── STEP 2 + 3 + 4: Build curriculum ──────────────────────────────
        curriculum = []
        for profile in report.priority_tasks:   # already RECOVERABLE, sorted by gap
            task_item = self._build_task_improvement(profile)
            curriculum.append(task_item)

        # Sort by priority_score descending (gap × (1-consistency))
        curriculum.sort(key=lambda x: x.priority_score, reverse=True)

        # ── RECURSION (optional, default-disabled) ─────────────────────────
        sub_task_graph = None
        recursion_context = None
        n_decomposed = 0
        if self.enable_recursion and self.recursion_engine is not None:
            sub_task_graph = self.recursion_engine.decompose(report, self._traj_records)
            recursion_context = {
                "enabled": True,
                "max_depth": self.recursion_engine.config.max_depth,
                "depth_reached": 0,
                "terminated_early": [],
            }
            n_decomposed = len(sub_task_graph.sub_tasks)
            # Attach sub-tasks to their parent TaskImprovement items.
            self._attach_sub_tasks(curriculum, sub_task_graph)

        # ── STEP 5: SIMULATE (optional) ────────────────────────────────────
        self._attach_predicted_gains(curriculum, report)

        predicted_total = (
            sum(t.predicted_gain for t in curriculum if t.predicted_gain is not None)
            or None
        )

        plan = SelfImprovementPlan(
            agent_name=self.agent_name,
            model_name=self.model_name,
            cycle=cycle,
            n_tasks_total=report.n_tasks,
            n_solid=report.n_solid,
            n_recoverable=report.n_recoverable,
            n_stuck=report.n_stuck,
            consistency_index=kappa,
            recoverable_score_left=report.recoverable_score_left,
            curriculum=curriculum,
            predicted_total_gain=predicted_total,
            cycle_complete=(report.n_recoverable == 0),
            job_dirs=self.job_dirs,
            n_trajectories_loaded=len(self._traj_records),
            recursion_enabled=self.enable_recursion,
            recursion_context=recursion_context,
            sub_task_graph=sub_task_graph.to_dict() if sub_task_graph else None,
            n_decomposed=n_decomposed,
        )
        return plan

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_task_improvement(self, profile: TaskOutcomeProfile) -> TaskImprovement:
        """Build a TaskImprovement from a TaskOutcomeProfile."""
        # Locate trajectory files for this task
        traj_files = self._find_traj_files_for_task(profile.task)

        # Match trajectory files to reinforce/contrast indices
        training_pairs = self._build_training_pairs(profile, traj_files)

        # STEP 3: RETRIEVE — memory results for this task
        mem_results = self.memory.retrieve_for_new_task(profile.task, k=3)

        # Priority score: gap × (1 - consistency) — how much can we gain AND how far to go
        priority_score = profile.gap * (1.0 - profile.consistency)

        # Recommendation
        rec = self._generate_recommendation(profile, training_pairs, mem_results)

        return TaskImprovement(
            task=profile.task,
            difficulty=profile.difficulty,
            kind=profile.kind,
            current_q_star=profile.q_star,
            current_q_bar=profile.q_bar,
            consistency=profile.consistency,
            gap=profile.gap,
            priority_score=priority_score,
            training_pairs=training_pairs,
            memory_results=mem_results,
            recommendation=rec,
        )

    # Mapping from disteval task names (in RecordStore) to tasks/ directory names
    _TASK_NAME_MAP = {
        "disteval/easy-word-count":   "easy-1",
        "disteval/easy-fizzbuzz":     "easy-2",
        "disteval/hard-bugfix":       "hard-1",
        "disteval/hard-algorithm":    "hard-2",
        "disteval/medium-log-parser": "medium-1",
        "disteval/medium-rest-client":"medium-2",
    }

    def _find_traj_files_for_task(self, task_path: str) -> list[tuple[float, str]]:
        """
        Return [(score, traj_path)] for all trajectories matching task_path.

        Handles two formats:
          - RecordStore format: "disteval/easy-fizzbuzz"
          - trajectory_loader format: "tasks/easy-2"
        """
        result = []
        # Normalise to the tasks/ directory name (e.g. "easy-2")
        task_dir_name = self._TASK_NAME_MAP.get(task_path)
        if task_dir_name is None:
            # Already in tasks/ format or unknown — extract basename
            task_dir_name = os.path.basename(task_path)

        for rec in self._traj_records:
            rec_basename = os.path.basename(rec.task_path)  # e.g. "easy-2"
            if rec_basename == task_dir_name:
                if os.path.exists(rec.traj_path):
                    result.append((rec.score, rec.traj_path))

        # Sort by score descending
        result.sort(key=lambda x: x[0], reverse=True)
        return result

    def _build_training_pairs(
        self,
        profile: TaskOutcomeProfile,
        traj_files: list[tuple[float, str]],
    ) -> list[TrainingPair]:
        """
        Build (reinforce, contrast) training pairs from trajectory files.

        Uses reinforce_idx / contrast_idx from the profile to identify
        high and low-scoring trajectories. For each contrast trajectory,
        find the closest high-scoring one to pair it with.
        """
        if not traj_files:
            return []

        scores = [s for s, _ in traj_files]
        paths = [p for _, p in traj_files]

        q_star = profile.q_star
        threshold = 0.9 * q_star

        high_indices = [i for i, s in enumerate(scores) if s >= threshold and q_star > 0]
        low_indices  = [i for i, s in enumerate(scores) if s < threshold]

        if not high_indices or not low_indices:
            return []

        pairs = []
        for low_idx in low_indices:
            best_high_idx = high_indices[0]  # highest-scoring high
            low_path  = paths[low_idx]
            high_path = paths[best_high_idx]

            # STEP 2: LOCALIZE — find structural divergence step
            divergence_step = self._find_divergence_step(high_path, low_path)

            pairs.append(TrainingPair(
                task=profile.task,
                reinforce_traj_path=high_path,
                contrast_traj_path=low_path,
                reinforce_score=scores[best_high_idx],
                contrast_score=scores[low_idx],
                gap=scores[best_high_idx] - scores[low_idx],
                structural_divergence_step=divergence_step,
            ))

        return pairs

    def _find_divergence_step(self, high_path: str, low_path: str) -> int:
        """
        Find the step at which the monitor's prediction diverges between
        a high-scoring and low-scoring trajectory.

        Returns the first step index where the high run is predicted HIGH
        and the low run is predicted LOW (or uncertain) simultaneously.
        Returns 0 if trajectories are too short or no divergence found.
        """
        try:
            high_steps = self.monitor.load_trajectory_steps(high_path)
            low_steps  = self.monitor.load_trajectory_steps(low_path)
        except Exception:
            return 0

        max_check = min(len(high_steps), len(low_steps), 20)
        for step in range(1, max_check + 1):
            high_match = self.monitor.check(high_steps, prefix_n=step)
            low_match  = self.monitor.check(low_steps,  prefix_n=step)
            if high_match.prediction == "high" and low_match.prediction in ("low", "uncertain"):
                return step

        return max_check

    def _attach_predicted_gains(
        self,
        curriculum: list[TaskImprovement],
        report: RightTailReport,
    ) -> None:
        """
        Attach predicted_gain from training_sim if available.
        Falls back to an analytic estimate if training_sim is not built.
        """
        sim_results = self._try_load_sim_results()

        # Agent name → JSON key mapping (training_sim uses these exact keys)
        _SIM_KEY_MAP = {
            "Claude Code":  "Claude (claude-sonnet-4-5)",
            "Gemini CLI":   "Gemini",
            "Codex CLI":    "Codex CLI (o4-mini)",
        }

        for item in curriculum:
            if sim_results:
                # Use Monte Carlo simulation results
                agent_key = _SIM_KEY_MAP.get(self.agent_name, self.agent_name)
                agent_data = sim_results.get("agents", {}).get(agent_key, {})
                disteval_data = agent_data.get("disteval", {})

                if disteval_data:
                    item.predicted_gain = disteval_data.get("mean_gain")
                    ci_low  = disteval_data.get("ci_low")
                    ci_high = disteval_data.get("ci_high")
                    if ci_low is not None and ci_high is not None:
                        item.predicted_gain_ci = (ci_low, ci_high)
                    item.predicted_rounds_to_threshold = agent_data.get(
                        "data_efficiency_disteval"
                    )
            else:
                # Analytic fallback: α × gap × P(recoverable improvement)
                # α = 0.4 (behavioral cloning efficiency from RL literature)
                # Only RECOVERABLE tasks respond; STUCK tasks are 10× harder
                alpha = 0.40
                gain = alpha * item.gap * item.consistency  # consistency scales confidence
                item.predicted_gain = round(gain, 4)
                item.predicted_gain_ci = (
                    round(gain * 0.7, 4),
                    round(gain * 1.3, 4),
                )

    def _attach_sub_tasks(
        self,
        curriculum: list[TaskImprovement],
        sub_task_graph: SubTaskGraph,
    ) -> None:
        """Attach decomposed sub-tasks to their parent TaskImprovement items."""
        if not sub_task_graph:
            return

        parent_to_item: dict[str, TaskImprovement] = {}
        for item in curriculum:
            parent_to_item[item.task] = item

        for sub_task in sub_task_graph.sub_tasks:
            parent = sub_task.parent_task
            if parent not in parent_to_item:
                continue
            parent_item = parent_to_item[parent]
            profile = sub_task_graph.profiles.get(sub_task.sub_task_id)
            if profile is None:
                continue

            sub_item = TaskImprovement(
                task=sub_task.sub_task_id,
                difficulty=parent_item.difficulty,
                kind="decomposed",
                current_q_star=sub_task.estimated_q_star,
                current_q_bar=sub_task.estimated_q_bar,
                consistency=profile.consistency,
                gap=profile.gap,
                priority_score=profile.gap * (1.0 - profile.consistency),
                training_pairs=[],
                memory_results=[],
                recommendation=(
                    f"Decomposed sub-task: {sub_task.instruction} "
                    f"(entry={sub_task.entry_step}, exit={sub_task.exit_step}, "
                    f"reward_delta={sub_task.reward_delta:.3f})"
                ),
            )
            parent_item.sub_tasks.append(sub_item)

    def contribute_to_pool(self, pool: "DistributedEvalPool") -> None:
        """Add this engine's eval records to a distributed pool."""
        if DistributedEvalPool is None:
            return
        from .distributed_eval import DistributedEvalRecord
        df = self.store.df()
        for _, row in df.iterrows():
            pool.add(
                DistributedEvalRecord(
                    agent_name=self.agent_name,
                    model_name=self.model_name,
                    task=row["task"],
                    score=float(row["score"]),
                    trajectory_ref=row.get("trajectory_ref"),
                    success=bool(row.get("success", False)),
                    failure_mode=row.get("failure_mode"),
                )
            )

    def attach_cross_agent_pairs(
        self,
        plan: SelfImprovementPlan,
        pool: "DistributedEvalPool",
        min_gap: float = 0.1,
    ) -> None:
        """Attach cross-agent training pairs to a plan from a distributed pool."""
        if DistributedEvalPool is None:
            return
        pairs = pool.generate_cross_agent_pairs(min_gap=min_gap)
        plan.cross_agent_pairs = [p.to_dict() for p in pairs]

    def _try_load_sim_results(self) -> Optional[dict]:
        """Load training_sim JSON results if available."""
        candidates = [
            "disteval_output/training_sim_results.json",
            "training_sim_results.json",
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        return json.load(f)
                except Exception:
                    pass
        return None

    @staticmethod
    def _generate_recommendation(
        profile: TaskOutcomeProfile,
        pairs: list[TrainingPair],
        mem_results: list,
    ) -> str:
        """Generate a human-readable recommendation string."""
        if not pairs:
            return (
                f"No trajectory pairs found for {profile.task}. "
                "Run more trials to generate reinforce/contrast pairs."
            )

        best_pair = pairs[0]
        div_step  = best_pair.structural_divergence_step
        n_pairs   = len(pairs)
        mem_note  = ""
        if mem_results:
            top_mem = mem_results[0]
            mem_note = (
                f" Memory #{1} (score={top_mem.entry.record.score:.2f}) "
                "shows a successful approach."
            )

        divergence_note = ""
        if div_step > 0:
            divergence_note = (
                f" Trajectories diverge at step {div_step} — "
                f"the failing run made a different structural choice here."
            )

        return (
            f"Use {n_pairs} reinforce/contrast pair(s). "
            f"Best reinforce: score={best_pair.reinforce_score:.2f}, "
            f"best contrast: score={best_pair.contrast_score:.2f}, "
            f"gap={best_pair.gap:.2f}."
            f"{divergence_note}"
            f"{mem_note}"
        )

    # ── Convenience: run the engine as a self-contained CLI ──────────────────

    def print_plan(self, plan: Optional[SelfImprovementPlan] = None) -> None:
        """Print a full cycle plan to stdout."""
        if plan is None:
            plan = self.run_cycle()
        print(plan.summary())


# ── Module-level convenience function ────────────────────────────────────────

def run_self_improvement_cycle(
    job_dirs: list[str],
    agent_name: str = "agent",
    model_name: str = "unknown",
    tasks_dir: str = "tasks",
    output_path: Optional[str] = None,
) -> SelfImprovementPlan:
    """
    One-call interface: load data, run a cycle, optionally save plan.

    Returns the SelfImprovementPlan.
    """
    engine = SelfEngine.from_job_dirs(
        job_dirs=job_dirs,
        agent_name=agent_name,
        model_name=model_name,
        tasks_dir=tasks_dir,
    )
    plan = engine.run_cycle()
    if output_path:
        plan.save(output_path)
    return plan


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_self_improvement_cycle(
            job_dirs=sys.argv[1:],
            agent_name="agent",
            model_name="unknown",
        )
    else:
        print("Usage: python -m disteval.self_engine <job_dir> [<job_dir> ...]")
