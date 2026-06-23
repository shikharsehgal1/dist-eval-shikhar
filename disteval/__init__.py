"""disteval -- a distribution-first evaluation layer for RL / agentic benchmarks.

Composes on top of an Inspect-style per-sample record store (adapters/inspect_log.py)
and rliable-style distribution math (adapters/rliable_bridge.py), adding the primitives
the landscape review found missing everywhere:

  1. first-class stratification of a persisted per-episode distribution  (records.py)
  2. failure-mode tagging + distributional failure reporting             (failure.py)
  3. the repeated-evaluation META-distribution -- eval reliability /     (repeat.py)
     test-retest noise of the headline score itself.
  4. right-tail training signal -- separates inconsistency from missing  (right_tail.py)
     skill; identifies which trajectories to reinforce vs contrast.
  5. real-time trajectory monitoring -- mid-episode outcome prediction   (trajectory_monitor.py)
     from structural trajectory features; 89% LOO accuracy on real data.
  6. cross-session trajectory memory -- outcome-indexed retrieval store  (trajectory_memory.py)
     surfacing recoverable-high trajectories before and during tasks.
"""
from . import bootstrap, compare, failure, metrics, repeat, right_tail
from . import trajectory_monitor, trajectory_memory
from . import self_engine
from . import recursion_engine, environment_generator, environment_registry, distributed_eval
from .records import EpisodeRecord, RecordStore

__all__ = [
    "EpisodeRecord", "RecordStore",
    "metrics", "bootstrap", "compare", "failure", "repeat", "right_tail",
    "trajectory_monitor", "trajectory_memory", "self_engine",
    "recursion_engine", "environment_generator", "environment_registry", "distributed_eval",
]
