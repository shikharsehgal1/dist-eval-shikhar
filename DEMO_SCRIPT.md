# disteval — Demo Script

**Total time:** 6–8 minutes  
**Setup:** `python3 disteval_gui.py` → open `http://localhost:8000`  
**Audience:** anyone who cares about AI agent reliability — researchers, PMs, investors, engineers

---

## Before you start

Have the browser open, full-screen, on **Tab ①  Harbor View**.  
The water animation is running. Don't say anything about the animation — let it land visually.

---

## Tab ① — Harbor View  *(~60 seconds)*

**Say:**

> "This is a real benchmark. Three AI coding agents — Claude Code, Gemini CLI, Codex CLI — each ran against a set of tasks inside isolated Docker containers. Harbor, the benchmark framework, ran each agent, scored the output, and reported this leaderboard."

*(gesture at the bar chart)*

> "Claude at 0.836, Gemini at 0.754, Codex at 0.300. Reasonable. Claude wins, Gemini is solid, Codex is struggling."

*(pause one beat)*

> "**If you were deploying Gemini based on this, you'd say it's fine. 0.754 is a decent score.**"

*(click Tab ②)*

---

## Tab ② — The Reveal  *(~75 seconds)*

**Say:**

> "disteval runs on the same data — the same runs you already did — and computes the full distribution."

*(point at the three bars: Mean, IQM, CVaR)*

> "The mean is still 0.754. That's Harbor's number, it's real. IQM — that's the mean with the top and bottom 25% stripped out — comes out at 0.955. So the robust center of Gemini's scores is actually very high."

*(pause — let them see the CVaR bar is flat)*

> "And then there's CVaR at 0.1. That's the expected score in the **worst 10%** of runs. The tail."

*(point at the red 0.000)*

> "**Zero.**"

*(pause two beats — don't rush this)*

> "In its worst runs, Gemini scores zero. The 0.754 mean is real — but it's averaging over a tail that completely collapses."

*(click Tab ③)*

---

## Tab ③ — Wow Moment  *(~90 seconds)*

**Say:**

> "The obvious question is: where? On which tasks does the tail collapse?"

*(point at the chart)*

> "disteval stratifies by difficulty. Harbor can't do this — it only knows the mean across everything."

*(point at the hard column first)*

> "Hard tasks: Gemini CVaR is 0.850. It's actually **reliable** on the hard stuff."

*(pause — then move to easy)*

> "Easy tasks: Gemini CVaR is **zero**."

*(let that sit for two seconds)*

> "These are beginner tasks. A junior developer handles them in five minutes. Gemini scores 0.910 mean on hard tasks, but it randomly scores zero on easy ones."

*(point at Codex easy column too)*

> "Codex does the same thing. Four of its six easy-task runs scored zero."

*(point at the annotation box)*

> "**Harbor's 0.754 showed nothing about this.** You would have deployed this agent not knowing it occasionally fails tasks that are trivially easy — and that's the failure mode that gets you in production, because those are the tasks you run constantly."

*(pause)*

> "This is the gap between a leaderboard number and a reliability claim."

*(click Tab ④)*

---

## Tab ④ — Training Signal  *(~75 seconds)*

**Say:**

> "So we know what's wrong. The distribution also tells you **how to fix it**."

*(point at the three bars: SOLID, RECOVERABLE, STUCK)*

> "disteval classifies every task into one of three states."

*(point at SOLID)*

> "SOLID: the agent always achieves its own best score on this task. No training needed — don't waste compute on it."

*(point at RECOVERABLE)*

> "RECOVERABLE: the agent solved it at least once, but not consistently. **This is the training priority.** These are the episodes where you have a successful trajectory and a failing trajectory on the same task — exactly the reinforce and contrast pair you need for DPO."

*(point at STUCK)*

> "STUCK: the agent never solved it. This is a capability gap. Training on these wastes compute — you need new skills, not more examples of failure."

*(point at Gemini's RECOVERABLE bar)*

> "Gemini has two RECOVERABLE tasks. Those are your DPO training targets. Codex has more STUCK tasks — different intervention entirely."

> "**The distribution doesn't just diagnose the problem. It tells you exactly which trajectories to put in your training batch.**"

*(click Tab ⑤)*

---

## Tab ⑤ — Agent Drill-down  *(~60 seconds)*

**Say:**

> "You can drill into any agent."

*(click Gemini pill)*

> "Gemini: mean 0.754, CVaR zero, κ — that's the consistency index — 0.885. κ measures how often the agent achieves its own best score. 0.885 is solid overall, but remember: that's pulled up by strong hard-task performance."

*(point at the table — highlight the RECOVERABLE rows)*

> "Word Count is RECOVERABLE: Q-star 1.0, gap 0.333, κ 0.67. The agent solved it — scored 1.0 — but not on every run. That's a reinforce run and a contrast run sitting right there in the data."

*(click Codex pill)*

> "Codex: κ 0.63, two RECOVERABLE, two STUCK. The STUCK tasks need capability work. The RECOVERABLE ones can be trained on immediately."

*(click Tab ⑥)*

---

## Tab ⑥ — Proof  *(~60 seconds)*

**Say:**

> "The natural question is: does this actually improve training? Or is it just a nicer way to look at the same data?"

*(point at the chart)*

> "We ran a Monte Carlo simulation — 5,000 bootstrap iterations — on the real trajectory data. Comparing three training data selection strategies: disteval's reinforce-contrast pairs, top-K mean reward, and random sampling."

*(point at the +249% annotation)*

> "**disteval produces 249% more score gain per training round than random sampling.** p-value 0.030."

*(point at the Gemini bar comparison)*

> "Against top-K mean reward, disteval wins by 172% for Gemini. The reason: when the right tail is sparse — when there are only a few reliable successes — the contrastive signal matters. You're not just picking the best runs, you're identifying the runs where the same task went both ways."

*(pause)*

> "Claude is the exception — it has so many high-score runs that top-K already approximates disteval. That's fine. When the agent is already solid, the selection strategy matters less."

---

## Closing line

*(back to the hero page, or just leave on the proof tab)*

> "Everything you've seen is from three real benchmark runs I did myself — 37 trajectories, real Docker environments, real LLM calls. disteval is a Python library. You drop it on any Harbor run and get this in two seconds."

---

## Q&A prep

**"What if I don't use Harbor?"**  
> "disteval reads a simple RecordStore — episode records with a score, task ID, and optional difficulty tag. Any benchmark that produces pass/fail or 0–1 scores can feed it."

**"How do you compute CVaR from so few runs?"**  
> "Honestly, 3 runs per task is thin for tight confidence intervals. The demo shows the pattern clearly, but in production you'd want 10+ runs per task. disteval includes a bootstrap CI so you can see the uncertainty explicitly."

**"Is the +249% number real?"**  
> "It's a simulation on real scores — the scores are from the actual runs, the '+249%' is from bootstrapping one round of DPO training. It models the training effect using the standard behavioral cloning efficiency from the RL literature. The simulation code and the real run data are both in the repo."

**"What's κ exactly?"**  
> "It's Q-bar over Q-star — the agent's average score on a task divided by its best score on that task. κ = 1.0 means it always achieves its own best. κ = 0.5 means on average it only achieves half of what it's capable of. It's a per-agent, per-task measure of consistency, not capability."

---

## Timing guide

| Tab | Script section | Target time |
|---|---|---|
| ① Harbor View | Setup + leaderboard | 60s |
| ② The Reveal | CVaR reveal | 75s |
| ③ Wow Moment | Easy task collapse | 90s |
| ④ Training Signal | RECOVERABLE taxonomy | 75s |
| ⑤ Agent Drill-down | Per-task table | 60s |
| ⑥ Proof | Monte Carlo numbers | 60s |
| **Total** | | **~7 min** |

For a **3-minute version**: tabs ①, ②, ③ only. Stop after "Harbor's 0.754 showed nothing about this." That's the complete argument.
