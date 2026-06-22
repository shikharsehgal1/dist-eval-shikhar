# Phase 1A — RMDP Formalism for Recursive Self-Improvement

**Paper:** *Recursive Reinforcement Learning* (Hahn et al., 2022), arXiv:2206.11430.  
**Sources:** arXiv abstract/PDF at <https://arxiv.org/abs/2206.11430> and <https://arxiv.org/pdf/2206.11430>; the original TeX source was extracted from the arXiv source tarball for line-level verification.  
**Scope:** This document summarizes the RMDP formalism and Recursive Q-learning, and identifies the concrete objects/notation that can be mapped to an agentic evaluation framework such as disteval.

---

## 1. Summary of the paper and its main contribution

Recursion is a natural way to finitely describe infinitely large or deeply nested decision problems, but standard reinforcement learning assumes a flat Markov Decision Process (MDP) interface: at every step the learner sees a single state, chooses an action, and receives a reward. When the environment has procedural structure (sub-routines that can call themselves, nested sub-tasks, or context-free constraints), the practitioner must manually "flatten" that structure into states and features, which is error-prone and opaque.

Hahn, Perez, Schewe, Somenzi, Trivedi, and Wojtczak address this by introducing **Recursive Markov Decision Processes (RMDPs)** as reinforcement-learning environments. An RMDP is a finite collection of component MDPs that can recursively invoke one another. Each component has **entry** and **exit** nodes that act like input parameters and return values, and **boxes** that represent calls to other components. The semantics of an RMDP is a countably infinite MDP whose state is a pair: the current vertex and a **call-stack** of pending boxes. This model is expressively equivalent to probabilistic pushdown systems and can capture probabilistic programs with unrestricted recursion, stochastic context-free grammars, and context-free reward machines. The paper's main algorithmic contribution is **Recursive Q-learning (RQL)**, a model-free RL algorithm that generalizes Watkins's Q-learning to RMDPs. The authors prove convergence for finite **1-exit** RMDPs and for **deterministic proper multi-exit** RMDPs, while the general multi-exit problem is undecidable.

---

## 2. Formal definitions

All notation below is taken directly from the paper's technical presentation (Section 2, lines 176–266 of `main.tex`).

### 2.1 Flat MDP (baseline)

An MDP is a tuple

$$
\mathcal{M} = (A, S, T, r)
$$

where $A$ is a finite action set, $S$ a finite state set, $T : S \times A \to \Delta(S)$ is the transition function, and $r : S \times A \to \mathbb{R}$ is the reward function. $A(s)$ denotes the actions available in state $s$.

### 2.2 Recursive MDP

A **recursive MDP** is a tuple

$$
M = (M_1, \dots, M_k)
$$

where each **component** $M_i$ is

$$
M_i = (A_i, N_i, B_i, Y_i, \mathsf{En}_i, \mathsf{Ex}_i, \delta_i, r_i).
$$

The parts are:

- $A_i$ — finite set of actions of component $i$.
- $N_i$ — finite set of **nodes**.
- $\mathsf{En}_i \subseteq N_i$ — **entry nodes** (input points of the component).
- $\mathsf{Ex}_i \subseteq N_i$ — **exit nodes** (output points), disjoint from $\mathsf{En}_i$.
- $B_i$ — finite set of **boxes**. Each box $b \in B_i$ is mapped by $Y_i : B_i \to \{1,\dots,k\}$ to another component (possibly itself).
- For each box $b$, the **call ports** and **return ports** are:

  $$
  \mathsf{Call}_b = \{(b, en) \mid en \in \mathsf{En}_{Y_i(b)}\},
  \qquad
  \mathsf{Return}_b = \{(b, ex) \mid ex \in \mathsf{Ex}_{Y_i(b)}\}.
  $$

- The **vertices** of $M_i$ are:

  $$
  Q_i = N_i \cup \mathsf{Call}^i \cup \mathsf{Return}^i,
  \quad
  \mathsf{Call}^i = \bigcup_{b\in B_i}\mathsf{Call}_b,
  \quad
  \mathsf{Return}^i = \bigcup_{b\in B_i}\mathsf{Return}_b.
  $$

- $\delta_i : Q_i \times A_i \to \Delta(Q_i)$ is the transition function. Transitions go from sources in $(N_i \setminus \mathsf{Ex}_i) \cup \mathsf{Return}^i$ to destinations in $(N_i \setminus \mathsf{En}_i) \cup \mathsf{Call}^i$.
- $r_i : Q_i \times A_i \to \mathbb{R}$ is the reward function.

An RMDP is **finite** when $k$ and all $A_i$, $N_i$, $B_i$ are finite. It is **1-exit** when every component has exactly one exit ($|\mathsf{Ex}_i| = 1$ for all $i$); otherwise it is a **multi-exit** RMDP.

### 2.3 Entry / exit points

- An **entry point** is either an entry node $en \in \mathsf{En}_i$ or a call port $(b, en) \in \mathsf{Call}_b$. It is where control enters a component.
- An **exit point** is either an exit node $ex \in \mathsf{Ex}_i$ or a return port $(b, ex) \in \mathsf{Return}_b$. It is where control leaves a component.

In the procedural analogy, entry nodes are function parameters, exit nodes are return values, call ports are procedure calls, and return ports are the places where the caller resumes after the callee returns.

### 2.4 Recursive call

A **recursive call** happens when the execution reaches a call port $(b, en)$. The box $b$ is pushed onto the call stack, and execution continues from the entry node $en$ of component $Y_i(b)$. Because $Y_i(b)$ can be the same component $i$, unbounded recursion is possible. The call stack records the chain of pending boxes.

### 2.5 Call-stack and semantics

The **call-stack** is a finite sequence of boxes $\kappa \in B^*$. The semantics of an RMDP $M$ is the (infinite-state) MDP

$$
\llbracket M \rrbracket = (A_M, S_M, T_M, r_M)
$$

where:

- $A_M = \bigcup_{i=1}^k A_i$.
- $S_M \subseteq B^* \times Q$ is the set of states $(\kappa, q)$: the current stack $\kappa$ and current vertex $q$.
- Transitions $T_M$ are defined as follows (lines 237–249):
  1. If $q = (b, en)$ is a call port, push the box: $(\kappa, q) \to (\kappa b, en)$ with probability 1.
  2. If $q = ex$ is an exit node and the stack is empty, the run terminates. Otherwise pop the top box: if $\kappa = \kappa' b$, then $(\kappa, ex) \to (\kappa', (b, ex))$ with probability 1, where $(b, ex)$ is the matching return port.
  3. Otherwise, internal transitions follow $\delta_i(q, a)(q')$.
- Rewards are zero at call ports and at exit nodes; otherwise they equal $r_i(q, a)$. The **diameter** is $r_{\max} = \max_{s,a} |r(s,a)|$.

The stack height increases by 1 on a call and decreases by 1 on a return. The empty stack means the top-level task has finished.

### 2.6 Objective: expected total reward

For a state $s \in S_M$ and strategy $\sigma$, the total reward is

$$
\mathbb{E}^{\text{Total}}_\sigma(s)
= \lim_{N\to\infty} \mathbb{E}_\sigma\Bigl[\sum_{1 \le i \le N} r(X_{i-1}, Y_i)\Bigr],
$$

where $X_i$ and $Y_i$ are the $i$-th state and action. The optimal value is

$$
\mathbb{E}^{\text{Total}}(s) = \sup_\sigma \mathbb{E}^{\text{Total}}_\sigma(s).
$$

A strategy $\sigma$ is **proper** for $s$ if the expected number of steps before termination is finite. The paper assumes (Assumption 1, line 271):

> **Proper Policy Assumption:** All strategies are proper for all states.

An RMDP satisfying this is called a **proper RMDP**. Properness guarantees that the infinite-horizon total reward is finite and well-defined.

---

## 3. Recursive Q-learning at a high level

Recursive Q-learning is a model-free algorithm that learns action-values for the infinite-state semantics $\llbracket M \rrbracket$ without representing the stack explicitly. The key idea is to **abstract the stack by the vector of expected rewards at the current component's exits** (lines 340–365).

### 3.1 What is learned

For a multi-exit RMDP, the algorithm maintains a value function

$$
Q(s, \mathbf{v}, a)
$$

where:

- $s$ is the current vertex (a node, call port, or return port).
- $\mathbf{v} \in \mathbb{R}^{|\mathsf{Ex}_i|}$ is the **exit-value vector** for the current component: $\mathbf{v}(ex)$ is the expected future reward obtained once the component reaches exit $ex$.
- $a$ is an available action.

When the agent enters a box, it does not know the stack below; it only needs to know what each possible exit of that box is worth. Thus the stack is replaced by the vector of exit values, making the state representation finite-dimensional.

### 3.2 Algorithm outline (multi-exit case)

Initialize $Q(s, \mathbf{v}, a)$ arbitrarily. Repeat until convergence:

1. Reset $\mathbf{v} \gets \mathbf{0}$ and the stack to empty.
2. Sample a trajectory $\tau$ from the RMDP.
3. For each transition $(s, a, r, s')$ in $\tau$:
   - **Entered a box** ($s'$ is a call port into box $b$):
     - Look up the return ports $\{s_{\text{exit}_1}, \dots, s_{\text{exit}_n}\}$ of $b$.
     - Compute the exit-value vector
       $$
       \mathbf{v}' = \bigl[\max_{a'} Q(s_{\text{exit}_1}, \mathbf{v}, a'), \dots, \max_{a'} Q(s_{\text{exit}_n}, \mathbf{v}, a')\bigr].
       $$
     - Normalize by subtracting $\min(\mathbf{v}')$ (one exit is anchored to 0; this does not change optimal policies).
     - Push the old $\mathbf{v}$ onto the stack; set $\mathbf{v} \gets \mathbf{v}'$.
     - Update
       $$
       Q(s, \mathbf{v}, a) \gets (1-\alpha_i) Q(s, \mathbf{v}, a) + \alpha_i\bigl(r + \max_{a'} Q(s', \mathbf{v}', a') + \min(\mathbf{v}')\bigr).
       $$
   - **Exited a box** ($s'$ is the $k$-th exit of the box):
     - Update
       $$
       Q(s, \mathbf{v}, a) \gets (1-\alpha_i) Q(s, \mathbf{v}, a) + \alpha_i\bigl(r + \mathbf{v}(k)\bigr).
       $$
     - Pop the previous $\mathbf{v}$ from the stack.
   - **Internal transition**:
     - Standard Q-learning update:
       $$
       Q(s, \mathbf{v}, a) \gets (1-\alpha_i) Q(s, \mathbf{v}, a) + \alpha_i\bigl(r + \max_{a'} Q(s', \mathbf{v}, a')\bigr).
       $$

A 1-exit RMDP simplifies this dramatically: the exit-value vector $\mathbf{v}$ is always 0, so the algorithm reduces to a standard $Q(s, a)$ table with a special bonus for entering a box (the max Q-value of the box's exit) and a simple reward-only update for exiting a box (Algorithm 2, lines 469–497).

### 3.3 What converges and under what conditions

- **1-exit RMDPs:** The simplified optimality equations have a **unique fixed point** equal to the optimal total reward vector $(\mathbb{E}^{\text{Total}}(q))_{q\in Q}$. Under the standard Robbins–Monro conditions on the learning rates ($\sum_i \alpha_i = \infty$, $\sum_i \alpha_i^2 < \infty$) and with every state-action pair visited infinitely often, tabular Recursive Q-learning converges to the optimal values (Theorem 4, lines 448–459).
- **Deterministic proper multi-exit RMDPs:** Because properness turns the semantics into a directed acyclic graph, the correct values propagate from leaves to roots. Tabular RQL with learning rate 1 converges to the optimal values when all state-action pairs are visited infinitely often (Theorem after Algorithm 1, lines 423–425).
- **General multi-exit RMDPs:** Convergence is **not guaranteed**; the strategy-existence problem is undecidable (Theorem 1, lines 283–287).

---

## 4. Key convergence guarantees and assumptions

| Class of RMDP | Guarantee | Assumptions |
|---|---|---|
| **General multi-exit** | Strategy-existence problem is **undecidable** (Theorem 1, lines 283–287). | — |
| **Finite, single-exit (1-exit)** | Recursive Q-learning converges to optimal total-reward values (Theorem 4, lines 448–459). | RMDP is finite and **proper**; all state-action pairs visited infinitely often; learning rates satisfy $\sum \alpha_i = \infty$, $\sum \alpha_i^2 < \infty$. |
| **Deterministic multi-exit** | Tabular RQL converges to optimal values with learning rate 1 (lines 423–425). | RMDP is **proper** and **deterministic**; all state-action pairs visited infinitely often. |
| **PAC learnability (1-exit)** | $\mathbb{E}^{\text{Total}}(s)$ is efficiently PAC-learnable (Theorem 5, lines 464–467). | $\varepsilon$-properness, known diameter $r_{\max}$, and known upper bound $K$ on expected termination time. |

The core assumptions are:

1. **Finiteness:** The number of components, actions, nodes, and boxes is finite.
2. **Properness (Assumption 1, line 271):** Every strategy terminates in finite expected time from every state. This is the analogue of the stochastic-shortest-path proper-policy assumption and ensures the total-reward objective is finite.
3. **Sufficient exploration:** All relevant state-action pairs are visited infinitely often (or a PAC-sampling equivalent).
4. **Robbins–Monro learning rates** (for the stochastic 1-exit case).

Discounting can be used to enforce properness: terminating with probability $1-\lambda$ at each step corresponds to discount factor $\lambda$ and yields parameters $b = r_{\max}/(1-\lambda)$, $c_o = 1 + 1/(1-\lambda)$, $\mu = 1-\lambda$ for the PAC bound (lines 312–313).

---

## 5. Concrete RMDP objects and notation mappable to an agentic eval framework (e.g., disteval)

The table below uses the RMDP notation from the paper and maps it to the concepts already present in disteval, particularly the `SelfEngine` docstring at `disteval/self_engine.py` lines 31–49, the right-tail taxonomy in `disteval/right_tail.py` lines 21–41, and the trajectory monitor/memory primitives in `disteval/trajectory_monitor.py` and `disteval/trajectory_memory.py`.

| RMDP object / notation | Meaning in the paper | Agentic-eval / disteval analogue |
|---|---|---|
| $M = (M_1, \dots, M_k)$ | The whole recursive environment | The full benchmark or suite of agentic tasks (e.g., `disteval` task suite) |
| Component $M_i$ | A sub-MDP with its own entry/exit and boxes | A single task or a reusable sub-task template (e.g., `disteval/easy-fizzbuzz`, `tasks/medium-2`) |
| Entry node $en \in \mathsf{En}_i$ | Initial state / input of a component | Task instruction + initial state/configuration (the prompt the agent receives) |
| Exit node $ex \in \mathsf{Ex}_i$ | Terminal state / return value of a component | Task solution / test success / final graded outcome |
| Box $b \in B_i$ with map $Y_i(b)$ | A sub-routine call to another component | A sub-task invocation (e.g., calling a code module, a tool, or a sub-agent) |
| Call port $(b, en)$ | The point where component $i$ calls component $Y_i(b)$ | The trajectory step at which a sub-task is initiated |
| Return port $(b, ex)$ | The point where control resumes after the call | The step where the sub-task outcome is returned and execution continues |
| Call stack $\kappa \in B^*$ | The chain of pending boxes | Trajectory step history / dependency chain of nested sub-tasks (`tool_sequence`, `TrajectoryRecord` in `trajectory_monitor.py` and `trajectory_memory.py`) |
| Exit-value vector $\mathbf{v}$ | Abstraction of the stack by the values of each exit | Per-task consistency score $\kappa(t)$ and gap $\Delta(t)$ from `disteval/right_tail.py` lines 21–41; also the `consistency` and `gap` fields in `TaskImprovement` (`self_engine.py` lines 81–100) |
| $Q(s, \mathbf{v}, a)$ or $Q(s, a)$ | Recursive action-value function | Estimated value of an action at a given task state, possibly conditioned on the expected outcome of nested sub-tasks |
| Strategy / policy $\sigma$ | Mapping from (stack, vertex) to actions | The agent's policy over tasks and tools |
| Properness | Every strategy terminates in finite expected time | Every eval trajectory terminates; no infinite loops or unbounded recursion in practice |
| Total reward objective | Expected sum of rewards until termination | Episode score or task outcome, aggregated across attempts (disteval's right-tail view) |
| 1-exit RMDP | Exactly one return value per component | Tasks with a single pass/fail success criterion |
| Multi-exit RMDP | Multiple return values per component | Tasks with multiple graded outcomes or branching success criteria (e.g., partial credit, multiple test cases) |
| Deterministic multi-exit | Transitions are deterministic | Sub-tasks with known, reproducible transitions; useful for safe curriculum generation |

### Notes on the mapping

- The `SelfEngine` docstring at `disteval/self_engine.py:31–49` explicitly frames its work as an RMDP-style decomposition: the taxonomy of SOLID / RECOVERABLE / STUCK tasks corresponds to nested gap localization, the per-task consistency $\kappa(t)$ corresponds to a sub-MDP Q-value, and the trajectory monitor's divergence step corresponds to an entry/exit boundary.
- `disteval/right_tail.py:21–41` defines $Q^*(t)$, $\bar{Q}(t)$, $\Delta(t)$, and $\kappa(t)$ — these are the empirical analogues of the RMDP exit-value vector and recursive Q-value.
- `disteval/trajectory_monitor.py` provides the structural signature and divergence-step detection that can serve as the call-port / return-port detector for an RMDP-style recursion engine.
- `disteval/trajectory_memory.py` stores and retrieves past trajectories by structural similarity, which is the natural implementation of the "stack-value backup" or memory retrieval at matching depth mentioned in the `SelfEngine` docstring.

---

## 6. Open threads for Phase 2

- The full RMDP model is undecidable in general. For disteval, the practical restriction is likely to **1-exit** or **deterministic multi-exit** sub-tasks generated from RECOVERABLE gaps.
- How to automatically infer boxes $B_i$ and the component map $Y_i$ from trajectory divergence steps (Phase 2 — recursion engine design).
- How to enforce properness in generated environments: e.g., episode-length limits, discounting, or termination tests.
- Whether to learn tabular $Q$ values or use neural function approximation (the paper reports both tabular and deep variants; the deep case requires the exit-value-vector abstraction).
