"""disteval.curriculum_optimizer — optimal control for curriculum scheduling.

This module formulates curriculum scheduling as a finite-horizon Markov Decision
Process (MDP) and solves it via value iteration or rolling-horizon model predictive
control (MPC).

MDP formulation
───────────────
* State  s = (κ_1, ..., κ_n, t)  where κ_i is the consistency index of task i
                                  and t is the elapsed training round.
* Action a ∈ {1, ..., n, STOP}  — choose which RECOVERABLE task to train on.
* Transition:  κ_i' = min(1, κ_i + α · Δ_i · (1 - κ_i))
               where Δ_i = q_star_i - q_bar_i is the recoverable gap.
* Reward:      R(s, a=i) = α · Δ_i · (1 - κ_i)
* Objective:   maximize Σ γ^t R(s_t, a_t) over the planning horizon.

Bellman optimality equation:

    V*(s) = max_a [ R(s, a) + γ · V*(s') ]

where s' is the deterministic next state and γ ∈ [0, 1) is a discount factor.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class CurriculumTask:
    """Lightweight task description for the curriculum optimizer."""

    task_id: str
    gap: float
    consistency: float

    @property
    def kappa(self) -> float:
        return self.consistency


class CurriculumValueIterator:
    """Solve a small curriculum MDP with value iteration.

    Parameters
    ----------
    tasks
        List of CurriculumTask objects (or any object with `gap` and `consistency`).
    alpha
        Learning-rate scaling in the transition dynamics.
    gamma
        Discount factor (must be < 1 for contraction).
    epsilon
        Convergence tolerance for the Bellman residual.
    n_bins
        Number of discrete bins per task consistency (κ is clipped to [0,1]).
    """

    def __init__(
        self,
        tasks: list[Any],
        alpha: float = 0.4,
        gamma: float = 0.99,
        epsilon: float = 1e-4,
        n_bins: int = 10,
    ):
        self.tasks = tasks
        self.n_tasks = len(tasks)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_bins = n_bins
        self._V: dict[tuple, float] = {}
        self._policy: dict[tuple, int] = {}

    def _discretize(self, kappa: float) -> int:
        return int(np.clip(int(kappa * self.n_bins), 0, self.n_bins))

    def _reward(self, kappas: tuple, action: int) -> float:
        task = self.tasks[action]
        return self.alpha * float(task.gap) * (1.0 - kappas[action] / self.n_bins)

    def _transition(self, kappas: tuple, action: int) -> tuple:
        kappas = list(kappas)
        reward = self._reward(kappas, action)
        # Δκ = reward / gap, so κ' = κ + reward / gap (clipped)
        gap = float(self.tasks[action].gap)
        if gap > 0:
            delta_kappa = reward / gap
        else:
            delta_kappa = 0.0
        new_kappa = min(self.n_bins, kappas[action] + delta_kappa * self.n_bins)
        kappas[action] = int(round(new_kappa))
        return tuple(kappas)

    def solve(self, horizon: int) -> dict:
        """Run value iteration and return the optimal value/policy tables.

        State keys are (κ_1, ..., κ_n, t) with discretized κ values.
        """
        if self.n_tasks == 0:
            return {"V": {}, "policy": {}}

        state_iter = itertools.product(range(self.n_bins + 1), repeat=self.n_tasks)
        states = [(kappas, t) for kappas in state_iter for t in range(horizon + 1)]

        for _ in range(1000):
            V_old = self._V.copy()
            max_delta = 0.0
            for kappas, t in states:
                if t == horizon:
                    self._V[(kappas, t)] = 0.0
                    continue

                best_value = 0.0  # STOP action
                best_action = -1
                for a in range(self.n_tasks):
                    reward = self._reward(kappas, a)
                    next_kappas = self._transition(kappas, a)
                    future = self.gamma * V_old.get((next_kappas, t + 1), 0.0)
                    value = reward + future
                    if value > best_value:
                        best_value = value
                        best_action = a

                key = (kappas, t)
                self._V[key] = best_value
                self._policy[key] = best_action
                max_delta = max(max_delta, abs(self._V[key] - V_old.get(key, 0.0)))

            if max_delta < self.epsilon:
                break

        return {"V": self._V, "policy": self._policy, "iterations": _}

    def plan(self, initial_kappas: list[float], horizon: int) -> list[str]:
        """Extract a full plan from the initial state."""
        self.solve(horizon)
        plan = []
        kappas = tuple(self._discretize(k) for k in initial_kappas)
        for t in range(horizon):
            action = self._policy.get((kappas, t), -1)
            if action < 0:
                break
            plan.append(self.tasks[action].task_id)
            kappas = self._transition(kappas, action)
        return plan


class MPCCurriculumPlanner:
    """Rolling-horizon model predictive control for curriculum scheduling.

    Brute-force enumerates action sequences over a short horizon and returns the
    first action (or full plan) with the highest cumulative discounted reward.
    """

    def __init__(
        self,
        tasks: list[Any],
        alpha: float = 0.4,
        gamma: float = 0.99,
        horizon: int = 5,
    ):
        self.tasks = tasks
        self.n_tasks = len(tasks)
        self.alpha = alpha
        self.gamma = gamma
        self.horizon = horizon

    def _reward(self, kappas: list[float], action: int) -> float:
        task = self.tasks[action]
        return self.alpha * float(task.gap) * (1.0 - kappas[action])

    def _step(self, kappas: list[float], action: int) -> list[float]:
        kappas = list(kappas)
        reward = self._reward(kappas, action)
        gap = float(self.tasks[action].gap)
        if gap > 0:
            kappas[action] = min(1.0, kappas[action] + reward / gap)
        return kappas

    def evaluate_sequence(self, initial_kappas: list[float], sequence: tuple) -> float:
        """Return cumulative discounted reward for an action sequence."""
        total = 0.0
        kappas = list(initial_kappas)
        for t, action in enumerate(sequence):
            reward = self._reward(kappas, action)
            total += (self.gamma ** t) * reward
            kappas = self._step(kappas, action)
        return total

    def solve(self, initial_kappas: list[float]) -> tuple[tuple, float]:
        """Find the best action sequence over the planning horizon."""
        best_seq = None
        best_value = -float("inf")
        for seq in itertools.product(range(self.n_tasks), repeat=self.horizon):
            value = self.evaluate_sequence(initial_kappas, seq)
            if value > best_value:
                best_value = value
                best_seq = seq
        return best_seq, best_value

    def next_action(self, initial_kappas: list[float]) -> Optional[int]:
        """Return only the first action of the optimal plan."""
        seq, _ = self.solve(initial_kappas)
        if seq is None:
            return None
        return int(seq[0])
