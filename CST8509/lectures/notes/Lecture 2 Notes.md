# Reinforcement Learning Lecture Notes: Markov Decision Processes & Q-Learning

---

## 1. From Markov Chains to MDPs — A Layered Framework

Reinforcement learning builds on a hierarchy of increasingly complex mathematical objects. Each layer adds a new component:

| Object | States | Transitions | Rewards | Actions |
|---|---|---|---|---|
| Markov Chain (MC) | ✓ | ✓ | ✗ | ✗ |
| Markov Reward Process (MRP) | ✓ | ✓ | ✓ | ✗ |
| Markov Decision Process (MDP) | ✓ | ✓ | ✓ | ✓ |

---

## 2. Markov Chains (Markov Processes)

A **Markov Chain** is simply a sequence of states with probabilistic transitions between them. There are no rewards and no actions — just states and the probabilities of moving from one state to another.

### Example: Two-State Chain

Consider two states, **A** and **E**:

- From A: 60% chance of staying in A, 40% chance of transitioning to E
- From E: 30% chance of staying in E, 70% chance of transitioning to A

A **sample** from this chain might look like: A → A → E → E → E → A → A → E ...

Each sample is one possible realization of the stochastic process governed by these transition probabilities. Over a large number of transitions, the observed frequencies converge to the defined probabilities (law of large numbers).

### The Student Example (David Silver)

A more elaborate Markov chain models a student's behavior:

- **States:** Class 1, Class 2, Class 3, Facebook, Pub, Sleep (terminal)
- From Class 1: 50% → Class 2, 50% → Facebook
- From Facebook: 90% → Facebook (continue scrolling), 10% → Class 1
- From Class 3: 60% → Pass, 40% → Pub
- Sleep is the **terminal state** (absorbing state — once entered, the process stays there)

Sample trajectories:
- Class 1 → Class 2 → Class 3 → Pass → Sleep
- Class 1 → Class 2 → Sleep

**Note on start states:** There is no formal "start state" symbol in a Markov chain. The process can in principle begin in any state. By convention, examples often begin at a particular state, but mathematically the chain is defined by its transition matrix, which describes conditional transitions between all pairs of states.

---

## 3. The Markov Property

The **Markov property** states that the future depends only on the present state, not on the history of how we arrived there.

Formally: the probability distribution over the next state depends only on the current state, not on any prior states.

### Intuition with Grid Worlds

In a 5×5 grid world, if the agent is at cell (3,3), it can move up, down, left, or right. Its future depends only on its current position and chosen action — not on the sequence of cells it visited previously. There are potentially many prior states it could have come from, but none of that matters.

### When a Single Observation Isn't Markov

Sometimes a single observation doesn't satisfy the Markov property. The solution is to **include more history in the state representation**.

**Atari example (DeepMind):** A single video game frame may not be Markov (e.g., you can't determine the direction a ball is moving from one frame). DeepMind's solution was to define the state as the **last 4 frames** — this was sufficient to capture velocity and direction, making the state effectively Markov.

In the worst case, you could include the entire history from the start of the episode as the state. This is always Markov by definition, but it becomes impractical (e.g., 60 frames/second would be unwieldy). The choice of how much history to include is a design decision, often found through trial and error.

---

## 4. Histories in Reinforcement Learning

A **history** is the full sequence of interactions from the start of an episode:

$$H_t = S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_t$$

Each step consists of an **action**, a **reward**, and an **observation** (new state). When studying Markov problems, we seek a state representation that satisfies the Markov property so that we can discard the rest of the history.

---

## 5. Markov Reward Process (MRP)

A **Markov Reward Process** extends a Markov chain by adding two components:

- **Reward function R:** Assigns an immediate reward to each state transition
- **Discount factor γ:** Controls how much future rewards are weighted relative to immediate rewards

### Student MRP Example

Adding rewards to the student Markov chain:
- Each class attended: **−2** (studying isn't immediately enjoyable)
- Passing: **+10** (the big payoff)
- Facebook: **−1** per step
- Pub: **+1** (beer tastes good)

There are still **no actions** — the process unfolds according to fixed transition probabilities. The rewards simply attach value judgments to the transitions.

### The Return and Discounting

The **return** $G_t$ from time step $t$ is the cumulative discounted reward:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

Where $\gamma \in [0, 1]$ is the discount factor.

**Why discount?**

- Prevents returns from going to infinity in non-terminating processes
- Reflects a preference for immediate over delayed rewards
- With $\gamma = 1$, there is no discounting (only works if episodes are guaranteed to terminate)
- Different values of γ effectively define different processes with different valuations

### The Value Function and the Bellman Equation for MRPs

The **state value function** $V(s)$ is the expected return starting from state $s$:

$$V(s) = \mathbb{E}[G_t \mid S_t = s]$$

The **Bellman equation** decomposes this into an intuitive recursive form:

$$V(s) = R_s + \gamma \sum_{s'} P_{ss'} V(s')$$

**Intuition:** The value of a state = the immediate reward + the discounted value of the next state. This is a deeply intuitive idea — what a state is worth is what I get now plus what I'll get going forward.

### A Note on Expectation

Expectation appeals to the **law of large numbers**. Example: the expected value of a fair six-sided die is:

$$\mathbb{E}[X] = \frac{1}{6}(1 + 2 + 3 + 4 + 5 + 6) = 3.5$$

You can never actually roll 3.5, but over many rolls, the average converges to this value. Similarly, value functions represent the expected (average) return over many possible trajectories through the Markov process.

---

## 6. Markov Decision Process (MDP)

An MDP adds **actions** to the MRP. Now an agent can make decisions that influence which states it transitions to. This is the core formalism of reinforcement learning.

An MDP is defined by the tuple: **(S, A, P, R, γ)**

- **S:** Set of states
- **A:** Set of actions
- **P:** Transition probability function — $P(s' | s, a)$
- **R:** Reward function — $R(s, a)$
- **γ:** Discount factor

### Deterministic vs. Stochastic Transitions

In the student MDP example, the "pub" action has stochastic outcomes: 40% stay in the same state, 40% transition to one state, 20% to another. This is **non-deterministic**.

In simple grid worlds, transitions are typically **deterministic**: choosing "down" from (3,3) always leads to (3,4) with 100% probability. This simplification makes learning more tractable for introductory study.

### Policies

A **policy** $\pi$ maps states to actions (or to probability distributions over actions):

$$\pi(a | s) = P(A_t = a \mid S_t = s)$$

- **Stochastic policy:** Assigns probabilities to each action in each state (allows exploration)
- **Deterministic policy:** Always selects one specific action per state

In grid world exercises, policies are typically treated as deterministic functions.

### The Action-Value Function (Q-Function)

The **Q-function** $Q^\pi(s, a)$ gives the expected return from taking action $a$ in state $s$ and then following policy $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$$

This is the function we learn in Q-learning. The value of a state under policy $\pi$ can be recovered as $V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)$.

---

## 7. Optimal Policy and Optimal Value Function

The **optimal policy** $\pi^*$ is the policy that maximizes the expected return from every state. Key results:

- There always exists at least one optimal policy
- There may be many optimal policies (all yielding the same value)
- The **optimal value function** $V^*(s)$ is the value function under the optimal policy
- The **optimal Q-function** $Q^*(s, a)$ satisfies the **Bellman optimality equation:**

$$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s', a')$$

**Intuition:** The optimal Q-value for a state-action pair equals the immediate reward plus the discounted maximum Q-value achievable from the next state.

### Multi-Agent Environments

In games like Go, the opponent is part of the **environment**. When the agent places a stone, the environment responds with the opponent's move and the resulting new state. The agent only computes values on its own turns.

### Limitations: Large State Spaces

For simple grid worlds, we can compute Q-values exactly using a Q-table. For games like Go, with an astronomically large state space, this is impossible. The solution is **value function approximation** — using a neural network to approximate Q instead of storing exact values. This also applies to continuous action spaces (e.g., steering angles).

---

## 8. Q-Learning: Temporal Difference Control

### Temporal Difference (TD) Learning

Q-learning is a form of **TD(0)** (temporal difference with one-step updates). Unlike Monte Carlo methods, which require waiting until the end of an episode to update, TD learning updates after **every single step**.

- **TD(0):** Learn from one step of information
- **TD(λ):** Learn from multiple steps
- **Monte Carlo:** Learn only after completing an entire episode

### The Q-Learning Update Rule

$$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma \max_{a'} Q(S', a') - Q(S, A) \right]$$

Where:
- $\alpha$ is the **step size** (learning rate)
- $R$ is the immediate reward
- $\gamma \max_{a'} Q(S', a')$ is the discounted best future value
- $Q(S, A)$ is the current estimate

**Role of the step size α:**

- If $\alpha = 1$: The old value is completely replaced by the new estimate (no memory of previous estimates)
- If $\alpha < 1$: A blend of the old value and the new estimate is retained, creating a smoother learning process

The term $R + \gamma \max_{a'} Q(S', a')$ is the **TD target** — it represents the Bellman equation applied with current estimates.

### ε-Greedy Action Selection

The agent follows an **ε-greedy** policy:
- With probability $(1 - \varepsilon)$: Choose the action with the highest Q-value (**exploit**)
- With probability $\varepsilon$: Choose a random action (**explore**)

Typically, ε **decays** over time, so the agent explores less as it learns more.

---

## 9. On-Policy vs. Off-Policy Learning

### Q-Learning (Off-Policy)

Q-learning updates using $\max_{a'} Q(S', a')$ — the best possible action in the next state — regardless of which action the agent actually takes. It learns the **optimal policy** while following an ε-greedy exploration policy. The learning target and the behavior policy are different, hence **off-policy**.

### SARSA (On-Policy)

SARSA updates using the Q-value of the action **actually chosen** by the policy in the next state:

$$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]$$

Where $A'$ is chosen according to the current policy (including random exploration). This means SARSA's learning accounts for the fact that the agent sometimes takes random actions.

### Cliff Walking Example

The cliff walking environment perfectly illustrates the difference:

- **Q-Learning (off-policy)** finds the **optimal path** — walking right along the edge of the cliff (shortest path, −13 total reward). It ignores the risk of random exploration steps pushing the agent off the cliff.
- **SARSA (on-policy)** finds a **safe path** — walking along the top of the grid, far from the cliff. It accounts for the fact that the ε-greedy policy occasionally takes random actions, which could be fatal near the cliff edge.

SARSA's policy is more conservative because it learns the value of states *under the policy it's actually following*, including the risk of random exploration.

---

## 10. Exploitation vs. Exploration

The **exploration-exploitation tradeoff** is fundamental to reinforcement learning.

- **Exploitation:** Use current knowledge to take the best known action
- **Exploration:** Try new actions to discover potentially better rewards

**The multi-armed bandit** is the simplest illustration: a single-state MDP with multiple actions (levers). If one lever pays $1 consistently, you might keep pulling it. But without exploring other levers, you'll never discover the one that pays $100 — or $1,000,000.

**Restaurant analogy:** You always go to the same restaurant because you know it's good (exploitation). But occasionally trying a new restaurant (exploration) might reveal something even better.

ε-greedy balances this tradeoff by exploiting most of the time while reserving a small fraction of actions for random exploration.

---

## 11. Visualizing the Q-Table During Training

### Representation

For a grid world with 48 states and 4 actions per state, the Q-table can be visualized by dividing each grid cell into four triangular regions representing the Q-values for **up, down, left, right**.

### Observations During Cliff Walking Training

1. **Initialization:** Q-values start between 0 and 100 (or random)
2. **Early episodes:** The agent falls off the cliff repeatedly, learning that cliff-adjacent "down" actions have Q-values around −100
3. **Convergence (~episode 30):** Most Q-values settle near −1 (the per-step reward). The **terminal state** has value 0, which is higher than any other value, creating a gradient that pulls the agent toward it
4. **Cliff cells are never updated** because the agent never occupies those cells — falling off the cliff means transitioning *from* an adjacent cell, not being *on* the cliff
5. **The optimal path emerges** as a gradient: Q-values for "right" actions along the optimal path are slightly higher (less negative) than other directions, guiding the agent toward the goal

### The Gradient Intuition

The terminal state's value of 0 "filters back" along the optimal path, becoming slightly more negative at each step away. This creates a value gradient:

- At the goal: Q ≈ 0
- One step away: Q ≈ −1.0
- Two steps away: Q ≈ −1.01
- And so on...

Simultaneously, a negative gradient near the cliff (Q ≈ −100) repels the agent. These two gradients together define the optimal path.

---

## 12. Roadmap of Key Topics

| Topic | Description |
|---|---|
| **Dynamic Programming** | Policy evaluation (computing values for a given policy) and policy improvement (using values to find better policies). Policy iteration and value iteration converge to the optimal policy. |
| **Monte Carlo Methods** | Similar to Q-learning but updates only occur at the end of an episode. The full trajectory is recorded, then values are updated by backing up from the terminal state. |
| **Temporal Difference Learning** | Updates occur after every step (TD(0)) or after several steps (TD(λ)). Q-learning is off-policy TD control. |
| **Value Function Approximation** | For large/continuous state spaces, use neural networks to approximate Q instead of exact Q-tables. Required for problems like Go or continuous control. |

### Policy Iteration Intuition

The value function and policy are interlinked and can be improved iteratively:

1. **Evaluate** the current policy → get value estimates
2. **Improve** the policy using those value estimates → better policy
3. **Repeat** → both converge to optimal

This interplay between evaluation and improvement is a recurring theme throughout reinforcement learning.
