# 🧊 Dynamic Programming for Reinforcement Learning

> A clean, from-scratch implementation of the classical **Dynamic Programming (DP)** algorithms that form the mathematical backbone of Reinforcement Learning — demonstrated on OpenAI Gym's **FrozenLake** environment.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.6+-3776AB?logo=python&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-scientific-013243?logo=numpy&logoColor=white">
  <img alt="Gym" src="https://img.shields.io/badge/OpenAI-Gym-0081A5?logo=openai&logoColor=white">
  <img alt="Notebook" src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white">
  <img alt="Domain" src="https://img.shields.io/badge/Domain-Reinforcement%20Learning-6E56CF">
</p>

---

## 📖 Overview

This project implements — with no black-box RL libraries — the core algorithms of **planning by Dynamic Programming**, exactly as taught in Sutton & Barto's *Reinforcement Learning: An Introduction* and David Silver's UCL lectures.

Dynamic Programming solves a **Markov Decision Process (MDP)** when the agent has *perfect knowledge of the environment's dynamics* — that is, the transition probabilities and rewards are known in advance. This "model-based" setting is the ideal starting point for building deep intuition about **value functions**, **policies**, and the **Bellman equations** before moving on to model-free methods like Q-Learning and Deep RL.

Every algorithm here is written in plain NumPy so the math stays visible, one loop at a time.

---

## 🎯 The Environment: FrozenLake

You and your friends were tossing a frisbee around a park when a wild throw left it out in the middle of a frozen lake. The ice is mostly solid — but there are holes where it has melted. Step in one, and you fall into the freezing water. Your job: **cross the lake and retrieve the disc.**

The lake is a 4×4 grid (an 8×8 map is also included):

```
S F F F        S : Start        (safe)
F H F H        F : Frozen       (safe)
F F F H        H : Hole         (episode ends — you fall in)
H F F G        G : Goal         (episode ends — reward = +1)
```

- **State space:** 16 discrete states (one per tile) — `Discrete(16)`
- **Action space:** 4 actions — `Left (0)`, `Down (1)`, `Right (2)`, `Up (3)` — `Discrete(4)`
- **Reward:** `+1` for reaching the Goal, `0` otherwise
- **The twist — slippery ice:** by default the environment is **stochastic**. When you choose a direction, you only move that way with probability ⅓; with probability ⅔ you slide to one of the two perpendicular directions. This uncertainty is what makes the problem genuinely interesting to *plan* for.

The environment exposes its full **one-step dynamics** `env.P[s][a]` as a list of `(probability, next_state, reward, done)` tuples — the model that Dynamic Programming requires.

---

## 🧠 Algorithms Implemented

All five building blocks of DP-based control are implemented in [`Dynamic_Programming_Git.ipynb`](Dynamic_Programming_Git.ipynb):

| # | Concept | Function | What it does |
|---|---------|----------|--------------|
| 1 | **Policy Evaluation** | `policy_evaluation(env, policy)` | Given a policy π, iteratively applies the Bellman expectation equation until the state-value function **V<sup>π</sup>** converges. |
| 2 | **Action-Value Estimation** | `q_from_v(env, V, s)` | Converts a state-value function **V** into the action-value function **Q(s, a)** for a single state — "how good is each action from here?" |
| 3 | **Policy Improvement** | `policy_improvement(env, V)` | Acts greedily with respect to **V** to produce a new, at-least-as-good policy (ties split uniformly for a stochastic policy). |
| 4 | **Policy Iteration** | `policy_iteration(env)` | Alternates *evaluation* ⇄ *improvement* until the policy stops changing — provably converging to the **optimal policy π\***. |
| 5 | **Value Iteration** | `value_iteration(env)` | Collapses evaluation and improvement into a single Bellman-optimality sweep, converging to **V\*** and then extracting **π\*** in one final step. |

### The two paths to optimality

```
POLICY ITERATION                          VALUE ITERATION
─────────────────                         ────────────────
  random π                                  V ← 0
     │                                        │
     ▼                                        ▼
  evaluate π  ──►  V^π                     Bellman-OPTIMALITY sweep
     │                                     V(s) ← max_a Q(s,a)
     ▼                                        │   (repeat to convergence)
  improve  ──►  π'                            ▼
     │                                     one greedy improvement
  π changed? ──yes──► loop                     │
     │no                                       ▼
     ▼                                     optimal π*, V*
  optimal π*, V*
```

Both routes arrive at the same optimal solution — Policy Iteration takes fewer, more expensive iterations; Value Iteration takes more, cheaper ones. Running both on the same lake makes the trade-off concrete.

---

## 📁 Repository Structure

```
.
├── Dynamic_Programming_Git.ipynb   # ★ Main notebook — all five algorithms, end to end
├── frozenlake.py                   # FrozenLake MDP: maps, transition dynamics (env.P), rendering
├── plot_utils.py                   # Heat-map visualization of the state-value function
└── README.md                       # You are here
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy matplotlib gym six
```

> **Note on `gym`:** `frozenlake.py` imports `gym.envs.toy_text.discrete`, which was available in the classic OpenAI Gym releases (≈ `gym==0.21`). If you are on a newer `gymnasium`/`gym` version where `discrete` has moved, install a compatible legacy version — e.g. `pip install "gym==0.21.0"` — or adapt the import to `gym.envs.toy_text.utils`.

### Run it

```bash
git clone https://github.com/ZainRaza14/Dynamic_Programming-Deep_Reinforcement_Learning.git
cd Dynamic_Programming-Deep_Reinforcement_Learning
jupyter notebook Dynamic_Programming_Git.ipynb
```

Then execute the cells top to bottom. You'll:

1. Build the FrozenLake MDP and inspect its state/action spaces.
2. Evaluate a uniform-random policy and **visualize** the resulting value function as a heat map.
3. Derive action values from it.
4. Solve the lake **twice** — via Policy Iteration and via Value Iteration — and confirm both recover the same optimal policy and value function.

### Minimal example

```python
import numpy as np
from frozenlake import FrozenLakeEnv
from plot_utils import plot_values

env = FrozenLakeEnv()

# Solve the MDP with Value Iteration
policy, V = value_iteration(env)

print("Optimal Policy (LEFT=0, DOWN=1, RIGHT=2, UP=3):")
print(policy)
plot_values(V)   # heat map of the optimal state-value function
```

---

## 📊 Visualizing the Value Function

`plot_utils.plot_values(V)` reshapes the 16-element value vector into the 4×4 lake and renders it as a labelled heat map — brighter tiles are worth more, and you can literally *see* value flowing outward from the Goal toward the Start as the algorithm converges.

---

## 💡 Key Takeaways

- **Bellman equations turned into code.** Each function is a direct, readable translation of the expectation/optimality equations — no hidden machinery.
- **Model-based planning.** DP assumes the dynamics are known; this project is the natural conceptual precursor to model-free RL (Monte-Carlo, TD, Q-Learning, DQN).
- **Two algorithms, one answer.** Seeing Policy Iteration and Value Iteration converge to the identical optimal policy cements *why* both are guaranteed to work.
- **Stochasticity matters.** Because the ice is slippery, the "optimal" policy is often counter-intuitive — sometimes steering *away* from a hole is the safest way toward the goal.

---

## 📚 References & Further Reading

This project was built while studying the following excellent resources:

1. [David Silver — UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) · [DP lecture slides (PDF)](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/DP.pdf)
2. [OpenAI Gym — FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0/)
3. [Udacity — Deep Reinforcement Learning](https://github.com/udacity/deep-reinforcement-learning)
4. [Alzantot — Deep RL Demystified: Policy Iteration, Value Iteration & Q-Learning](https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa)
5. [Karpathy — ReinforceJS Gridworld DP (interactive)](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html)
6. [Kaelbling et al. — Reinforcement Learning: A Survey](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/node19.html)
7. [Policy Iteration vs. Value Iteration (Quora)](https://www.quora.com/How-is-policy-iteration-different-from-value-iteration)
8. [Q-Learning tutorial (Hartford, PDF)](http://uhaweb.hartford.edu/compsci/ccli/projects/QLearning.pdf)

---

## 🗺️ Where to Go Next

Dynamic Programming is the foundation. Natural follow-ups once the environment's model is *unknown*:

- **Monte-Carlo methods** — learn from complete episodes
- **Temporal-Difference learning** — SARSA, Q-Learning
- **Function approximation & Deep RL** — DQN, Policy Gradients, Actor-Critic

---

<sub>Built as a hands-on study of the algorithms behind Reinforcement Learning. Contributions, corrections, and questions are welcome.</sub>
