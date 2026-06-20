# disteval — Demo Script

**Total time:** 7–9 minutes  
**Setup:** `python3 disteval_gui.py` → open `http://localhost:9173`  
**Audience:** anyone who cares about AI agent reliability — researchers, PMs, investors, engineers

---

## Before you start

Have the browser open, full-screen, on **Step 1 of 7 — "The training signal problem"**.  
The nav bar at the bottom shows "Step 1 of 7". Don't click anything yet.

---

## Step 1 — The training signal problem  *(~60 seconds)*

**Say:**

> "This is the core problem. Two agents. Both score 0.5 mean on a task."

*(point at the two side-by-side columns in the panel)*

> "Agent A scored 1.0 on one run and 0.0 on another. Agent B scored 0.5 on both runs."

*(pause one beat)*

> "Same mean. Completely different situations. Agent A has a training signal — reinforce the 1.0 run, contrast it against the 0.0 run. Agent B doesn't — it's stuck at 0.5 and needs new capability entirely."

*(pause)*

> "Mean reward cannot tell these apart. disteval can. And it automatically extracts the training pair from Agent A's eval data — no labels, no synthetic data."

*(click Next →)*

---

## Step 2 — The full outcome distribution  *(~75 seconds)*

**Say:**

> "This is the same benchmark Harbor reported. Three real agents — Claude Code, Gemini CLI, Codex CLI — each ran six tasks inside isolated Docker containers."

*(click the "Gemini CLI" pill to select it)*

> "Harbor showed Gemini at 0.754. disteval shows you four numbers instead of one."

*(point at the four bars)*

> "Mean: 0.754 — that's Harbor's number, it's real. IQM — mean with the top and bottom 25% stripped — 0.955. So Gemini's robust center is actually excellent."

*(pause — let them see the CVaR bar)*

> "CVaR at 0.1: the expected score in the **worst 10%** of runs."

*(point at the 0.000)*

> "**Zero.**"

*(pause two beats — don't rush this)*

> "And look at pass@3 versus pass^3. Pass@3 — the probability of solving a task in at least one of three tries — is high. Pass^3 — the probability of solving it every time — drops sharply. That gap is the signature of inconsistency. That gap is where the DPO training pairs live."

*(click Next →)*

---

## Step 3 — Where inconsistency lives  *(~90 seconds)*

**Say:**

> "The obvious question is: where? On which tasks does the tail collapse?"

*(point at the difficulty bands — Easy, Medium, Hard)*

> "disteval stratifies by difficulty. Harbor can't do this — it only knows the aggregate."

*(click the "Hard" band)*

> "Hard tasks: Gemini CVaR is 0.85. It's actually **reliable** on the hard stuff."

*(click "Easy")*

> "Easy tasks: CVaR near zero."

*(let that sit for two seconds)*

> "These are beginner tasks. A junior developer handles them in five minutes. Gemini scores well on average on easy tasks — but its worst runs on them score zero. The pass@3 versus pass^3 gap opens widest here."

*(point at the chart)*

> "**This is where the training pairs are.** The easy tasks where the agent sometimes fails are exactly the RECOVERABLE tasks. They have a passing run and a failing run already sitting in the data."

*(click Next →)*

---

## Step 4 — The free training signal  *(~90 seconds)*

**Say:**

> "Here's the math behind the taxonomy."

*(point at the three equations)*

> "Q-star of t — the agent's best score on task t across all runs. Q-bar — the mean. Delta — the gap between them."

*(point at the kappa definition)*

> "And kappa: Q-bar divided by Q-star. The consistency index. Kappa equals 1 means the agent achieves its best every time. Kappa equals 0.63 — Codex's number — means 37% of its demonstrated capability is left unrealized on every attempt."

*(point at the three taxonomy cards)*

> "SOLID: Q-star equals Q-bar, delta is zero. The agent is already consistent. Skip it — no training signal here."

*(point at RECOVERABLE)*

> "**RECOVERABLE: Q-star is positive, delta is positive.** The agent solved this task — we have a passing run. It also failed it — we have a failing run. Those two trajectories are a DPO training pair. Right now, in the jobs/ directory. No labels. No synthetic data."

*(point at STUCK)*

> "STUCK: Q-star is zero. The agent never solved it. No positive run to reinforce. This needs new capability, not consistency training."

*(pause)*

> "Mean reward treats RECOVERABLE and STUCK identically — both look like 'low score, needs improvement.' They need completely different interventions. This is the information mean reward throws away."

*(click Next →)*

---

## Step 5 — Agent drill-down: the actual training pairs  *(~75 seconds)*

**Say:**

> "You can drill into any agent and see exactly what disteval found."

*(click the "Codex CLI" pill)*

> "Codex: consistency index kappa 0.63. The ring shows how much of its demonstrated capability it actually deploys consistently."

*(point at the task table)*

> "Every task is classified. SOLID tasks are greyed out — nothing to do. RECOVERABLE tasks are highlighted."

*(click a RECOVERABLE row to expand it)*

> "Here's the actual training pair. The reinforce trajectory — the run where it scored 1.0. The contrast trajectory — the run where it scored 0.0. Same task. And the structural divergence step: step 5 is where these two runs first made a different tool-call choice. That's the moment the DPO loss should be anchored to."

*(click Gemini pill)*

> "Gemini: kappa 0.885. Fewer RECOVERABLE tasks, but the ones that are there have real pairs with large gaps."

*(click Next →)*

---

## Step 6 — Self-engine: the live loop  *(~75 seconds)*

**Say:**

> "disteval doesn't just diagnose — it runs the full loop automatically."

*(click "Run SelfEngine")*

> "Watch what happens."

*(let the SSE stream run — point at each stage as it appears)*

> "Stage one: load all trajectory files from the job directories. Stage two: compute the score distribution. Stage three: classify every task. Stage four: extract the reinforce-contrast pairs. Stage five: rank by priority — delta times one-minus-kappa, so tasks with the largest gap and lowest consistency come first. Stage six: output the curriculum."

*(point at the JSON output when it appears)*

> "That JSON file — improvement_plan.json — contains the ranked list of tasks with the actual trajectory file paths for each training pair. You feed that directly to your DPO trainer. One command:"

*(show terminal or just say it)*

> "`disteval engine jobs/run_1/ --agent my-agent --output plan.json`"

*(click Next →)*

---

## Step 7 — Monte Carlo proof  *(~60 seconds)*

**Say:**

> "The natural question: does this actually improve training? Or is it just a nicer way to look at the same data?"

*(point at the chart)*

> "We ran a Monte Carlo simulation — 5,000 bootstrap iterations — on the real trajectory data. Three training data selection strategies: disteval's reinforce-contrast pairs, top-K mean reward, and random sampling."

*(point at the +249% annotation)*

> "**disteval produces 249% more score gain per training round than random sampling.** p-value 0.030."

*(point at the per-agent breakdown)*

> "Against top-K mean reward, disteval wins by 172% for Gemini, 89% for Codex. The reason: when the right tail is sparse — when there are only a few reliable successes — the contrastive signal matters. You're not just picking the best runs, you're identifying the runs where the same task went both ways."

*(pause)*

> "Claude is the exception — kappa 1.0, fully consistent, zero RECOVERABLE tasks. When the agent is already solid, the selection strategy matters less. That's expected."

---

## Closing line

*(leave on Step 7 or click ↺ Restart to go back to Step 1)*

> "Everything you've seen is from three real benchmark runs — 54 episodes, real Docker environments, real LLM calls. disteval is open source. `pip install disteval`. You drop it on any Harbor run, any Inspect log, or any JSONL file of scores and trajectories, and you get this in one command."

---

## Q&A prep

**"What if I don't use Harbor?"**
> "disteval reads a generic JSONL format — episode records with a score, task ID, model name, and optional difficulty. Any benchmark that produces 0–1 scores can feed it. There are adapters for Harbor, Inspect (UK AISI), and rliable, plus a generic adapter for anything else."

**"How do you compute CVaR from so few runs?"**
> "Honestly, 3 runs per task is thin for tight confidence intervals. The demo shows the pattern clearly, but in production you'd want 10+ runs. disteval includes stratified bootstrap CIs so you can see the uncertainty explicitly."

**"Is the +249% number real?"**
> "It's a simulation on real scores — the actual run data, bootstrapped. The training effect model uses standard behavioral cloning efficiency from the RL literature, with a 1.5× bonus for contrastive pairs. The simulation code and real run data are both in the repo."

**"What's kappa exactly?"**
> "Q-bar over Q-star — the agent's average score on a task divided by its best score. Kappa 1.0 means it always achieves its own best. Kappa 0.63 means on average it captures 63% of what it's demonstrated it can do. It's a per-agent, per-task measure of consistency, not raw capability."

**"What about the training pairs — how do you know which trajectory to reinforce?"**
> "The reinforce trajectory is the highest-scoring run on that task. The contrast is the lowest-scoring run. The structural divergence step is where their tool-call sequences first differed — write vs. search vs. exec. That's the step the DPO loss is most informative around."

**"What's next?"**
> "Closing the loop with an actual fine-tuning run — watching kappa rise cycle over cycle. Then a larger benchmark, 50+ tasks. And publishing kappa as a standard metric alongside IQM and CVaR in the eval reliability literature."

---

## Timing guide

| Step | Title | Target time |
|------|-------|-------------|
| 1 | The training signal problem | 60s |
| 2 | The full outcome distribution | 75s |
| 3 | Where inconsistency lives | 90s |
| 4 | The free training signal | 90s |
| 5 | Agent drill-down: training pairs | 75s |
| 6 | Self-engine: live loop | 75s |
| 7 | Monte Carlo proof | 60s |
| **Total** | | **~8 min** |

**3-minute version:** Steps 1, 2, 3 only. Stop after "this is where the training pairs are." That's the complete argument.

**5-minute version:** Steps 1–5. Skip the live engine run and the Monte Carlo. End on the agent drill-down showing the actual reinforce/contrast pair.
