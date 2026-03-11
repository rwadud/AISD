# Reinforcement Learning — Lecture Notes: Core Concepts & Foundations

---

## 1. The Agent–Environment Interface

Reinforcement learning (RL) is built on a fundamental distinction between two entities:

- **Agent** — the learner and decision-maker.
- **Environment** — everything the agent interacts with, which is external to the agent.

### The RL Cycle

At each discrete time step *t*, the following loop occurs:

1. The agent observes the current **state** (or **observation**) and receives a **reward** from the environment.
2. Based on that information, the agent selects an **action**.
3. The action is performed on the environment.
4. The environment transitions to a new state and produces a new reward.
5. The cycle repeats at time step *t+1*, *t+2*, and so on.

This loop continues for the duration of an **episode** (not "epoch" — that term belongs to supervised learning). Some episodes terminate (e.g., reaching a goal in a maze), while others run indefinitely (e.g., a thermostat continuously regulating temperature). Non-terminating episodes are perfectly valid in RL, and they motivate the need for a **discount factor** (covered later).

> **Key point:** The reward is always a **scalar** value, not a vector. Even in complex real-world trade-off situations, the reward must be distilled into a single number.

---

## 2. Reward

The reward *R_t* is a scalar signal received by the agent at each time step that indicates how well the agent is doing. Key properties:

- **Scalar, not a vector** — all multi-dimensional trade-offs are compressed into one number.
- **Can be negative** — negative rewards (penalties) are useful when you want to encourage shorter paths or faster solutions. The agent will try to minimize the total penalty, which corresponds to finding the shortest or most efficient sequence of actions.
- **Received at every time step** — each action produces a reward from the environment.

### The Reward Hypothesis

> *All goals can be described by the maximization of expected cumulative reward.*

This is a **hypothesis**, not a proven theorem. It is central to RL — all RL algorithms are designed around maximizing cumulative reward. A notable practical issue is **reward hacking**, where an agent discovers ways to increase reward without genuinely advancing toward the intended goal, usually indicating a poorly designed reward function.

### Reward vs. Value (Preview)

- **Reward**: the immediate scalar payoff from a single action.
- **Value**: the expected *total* future reward from a given state onward (see Section 4).

An analogy: investing money involves an immediate negative reward (you lose money), but the expected cumulative reward over time is positive (you gain more money back). A state can have high value even if the next immediate reward is negative.

---

## 3. Policy (π)

A **policy** is the agent's strategy — a function that maps states to actions. It determines how the agent behaves.

### Deterministic Policy

$$\pi(s) = a$$

Given a state *s*, the policy deterministically outputs a single action *a*.

**Example:** In a grid world where the agent can move forward, turn left, or turn right, a simple (though not optimal) policy might be "always step forward regardless of the state." This agent would walk in a straight line until hitting a wall, then keep trying to step forward forever.

### Stochastic Policy (General Case)

In the general case, a policy outputs a **probability distribution** over actions:

$$\pi(a \mid s) = P(A_t = a \mid S_t = s)$$

For example: when in state *s*, take action *a₁* with 20% probability and action *a₂* with 80% probability. The probabilities must sum to 1.

> **In this course**, the focus is primarily on the **deterministic case** for simplicity.

---

## 4. Value Functions

A value function estimates **how good** it is for the agent to be in a given state (or to take a given action in a given state), measured in terms of expected cumulative future reward.

### State Value Function — V(s)

$$V(s) = \text{Expected total reward from state } s \text{ to the end of the episode}$$

This tells you: "Given that I'm in state *s* and I follow my current policy, what total reward can I expect to accumulate?"

- If you're in a **high-value state**, that's a good position to be in overall, even if the next immediate reward might be negative.
- The value can **oscillate** during an episode — it may go up when something good is about to happen, then drop after that reward has been collected and is now "behind" the agent.

### Action Value Function — Q(s, a)

$$Q(s, a) = \text{Expected total reward from state } s \text{, taking action } a \text{, then following the policy}$$

This tells you: "Given that I'm in state *s* and I take specific action *a*, what total reward can I expect?"

The difference from *V(s)* is that *Q* specifies both the state **and** the action being taken.

### Optimal Value Functions

The **optimal** value function, denoted with a star (e.g., *Q\**), corresponds to the best possible policy. **Q-learning** is the process of learning this optimal action value function.

### Interaction Between Policy and Value

These two concepts are deeply intertwined: changing the policy changes the value of states, and knowing the value of states can inform a better policy. The value function for a specific policy is often denoted with a superscript indicating which policy it corresponds to.

---

## 5. Model

A **model** is the agent's internal representation of the environment. It allows the agent to predict what will happen next (state transitions and rewards) without actually performing actions.

- With a model, the agent can do **planning** — reasoning ahead about the consequences of actions to determine the best course of action.
- **Not all RL agents have models.** In model-free RL (e.g., Q-learning in a grid world), the agent has no advance knowledge of how the environment will respond; it must learn through direct interaction.

**Example:** A Blocks World defined in Prolog can serve as both the environment and a model. If the agent has access to the model, it can plan which actions will achieve the goal without trial and error.

---

## 6. Markov Decision Processes (MDPs)

All RL problems studied in this course are formulated as **Markov Decision Processes**. MDPs build up in layers:

1. **Markov Chains** — sequences of states with the Markov property.
2. **Markov Reward Processes** — Markov chains with rewards added.
3. **Markov Decision Processes** — Markov reward processes with actions (decisions) added.

### The Markov Property

> The probability of each possible next state and reward depends **only on the immediately preceding state and action**, not on the full history.

In other words: **the future depends only on the present, not the past.** If a state encodes all relevant information, it is said to be a **Markov state**.

### Example: Constant-Velocity Particle

- **State = position only** → **Not Markov.** Knowing only the current position isn't enough to predict the next position; you'd need to look at past positions to infer direction.
- **State = position + velocity** → **Markov.** Knowing both position and velocity is sufficient to predict the next state.

### Choosing the Right State Representation

The RL practitioner must decide what constitutes the state. This is a design choice.

**Atari Games Example (DeepMind):**
- One video frame alone is **not Markov** (no motion information).
- DeepMind used the **last 4 frames** as the state, which captures enough motion information to satisfy the Markov property.
- At 60 frames per second with groups of 4, the agent makes ~15 decisions per second.

**Worst case:** The entire history is always Markov (it contains everything), but this is impractical. The goal is to find the **minimal sufficient state** representation.

### The Rat Example

Consider a rat encountering sequences of stimuli (lights, bells, levers). Whether the rat gets cheese or electrocuted depends on what is considered the "state" — just the last stimulus? The last two? The entire history? Different state definitions lead to different predictions, illustrating why state design matters.

---

## 7. Categorizing RL Agents

RL agents can be classified by which components they maintain:

| Agent Type | Policy | Value Function | Model |
|---|---|---|---|
| **Value-based** | ✗ (implicit) | ✓ | ✗ |
| **Policy-based** | ✓ | ✗ | ✗ |
| **Actor-Critic** | ✓ | ✓ | ✗ |
| **Model-based** | ✓ or ✗ | ✓ or ✗ | ✓ |

- **Value-based:** Choose actions based solely on the value function (no explicit policy).
- **Policy-based:** Choose actions directly from the policy, without consulting a value function.
- **Actor-Critic:** The policy ("actor") picks actions; the value function ("critic") evaluates them.
- **Model-based:** The agent has a model of the environment and can use it for planning.

---

## 8. Learning vs. Planning

- **Planning:** The agent has a model of the environment and can reason about the consequences of actions internally, without direct interaction. No learning from experience is needed — the model provides all necessary information.
- **Learning:** The agent does **not** have a complete model. It must interact with the environment, observe rewards and state transitions, and improve its policy from experience.

**Dynamic programming** is a family of algorithms for policy improvement that requires full knowledge of the environment (all transition probabilities and rewards). In general RL, this information is unknown and must be learned.

---

## 9. Exploration vs. Exploitation

A fundamental dilemma in RL:

- **Exploitation:** Choose the best-known action (e.g., always go to your favourite restaurant).
- **Exploration:** Try something new to potentially discover a better option (e.g., try a random restaurant — it might be better or worse).

Balancing these two is critical for effective learning. Pure exploitation may miss better strategies; pure exploration wastes time on poor actions.

---

## 10. Q-Learning Preview

**Q-learning** is a specific RL algorithm that learns the optimal action value function *Q\**.

- The Q-function is stored in a **table** (in the tabular case), indexed by state and action.
- In a grid world with 4 possible actions (up, down, left, right), the Q-table is essentially a 3D array: grid rows × grid columns × 4 actions.
- The agent explores the environment, receives rewards, and updates the Q-table using the **Bellman equation** to converge toward the optimal Q-values.
- Once learned, the optimal Q-function directly implies the optimal policy: in any state, choose the action with the highest Q-value.

---

## 11. RL as the Third Paradigm of Machine Learning

Reinforcement learning is considered the **third major type of machine learning**, alongside supervised and unsupervised learning. It differs fundamentally:

- **No labeled data** — unlike supervised learning.
- **Scalar reward signal** — not class labels or target outputs.
- **Sequential decision-making** — the agent's actions affect future states and rewards.
- **Temporal credit assignment** — rewards may be delayed, making it hard to determine which past actions led to the outcome.

RL is remarkably broad, appearing across many disciplines under different names: **optimal control** (engineering), **reward systems** (neuroscience), **classical conditioning** (psychology), **bounded rationality** (economics), and **operations research** (mathematics).

---

## 12. Key Terminology Summary

| Term | Definition |
|---|---|
| **Agent** | The learner/decision-maker |
| **Environment** | Everything external to the agent |
| **State (s)** | A summary of the current situation (must be designed to be Markov) |
| **Action (a)** | What the agent does to the environment |
| **Reward (r)** | Scalar feedback signal at each time step |
| **Episode** | One complete run of the agent–environment interaction |
| **Policy (π)** | Function mapping states to actions |
| **Value Function V(s)** | Expected cumulative reward from state *s* |
| **Action Value Function Q(s, a)** | Expected cumulative reward from state *s* taking action *a* |
| **Model** | Agent's internal representation of the environment |
| **Markov Property** | Next state depends only on current state and action |
| **Bellman Equation** | Recursive relationship used to compute/update value functions |
| **Discount Factor** | Reduces the weight of future rewards (needed for infinite episodes) |

---

*Reference: Sutton & Barto, "Reinforcement Learning: An Introduction" (standard textbook); David Silver's RL Lecture Series (freely available on YouTube).*
