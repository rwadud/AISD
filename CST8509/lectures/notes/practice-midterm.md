# CST8509 Practice Midterm Questions

---

## Part A: Diagrams (Draw / Label)

### Q1. Agent-Environment Interaction Diagram

Draw the agent-environment interaction diagram. Label all arrows with the correct variable names and time subscripts ($A_t$, $R_{t+1}$, $S_{t+1}$, etc.). Briefly explain the cycle in 2-3 sentences.

### Q2. Cliff Walking Optimal Path

On a 4x12 grid, mark the start position (bottom-left), the goal (bottom-right), and the cliff (bottom row between start and goal). Draw the optimal path a Q-learning agent would learn. How many steps does this path take?

---

## Part B: Multiple Choice

### Q3.

The reward hypothesis states that all goals can be described by:

a) Minimizing the number of steps taken
b) Maximizing expected cumulative reward
c) Maximizing immediate reward at every step
d) Minimizing reward over all episodes

### Q4.

The Markov property states that:

a) The next state depends on all previous states and the initial state
b) The next state depends only on the current state and action
c) The agent must observe the full environment state at all times
d) Future rewards are always discounted by gamma

### Q5.

In reinforcement learning, a single run of the agent-environment loop from start to termination is called:

a) An epoch
b) A batch
c) An episode
d) A rollout

### Q6.

What is the correct term for the RL equivalent of "learning rate" in supervised learning?

a) Discount factor
b) Epsilon
c) Step size
d) Exploration rate

### Q7.

Which of the following is true about the reward signal in reinforcement learning?

a) It is a vector of values for each action
b) It is a scalar value
c) It must always be positive
d) It is only given at the end of an episode

### Q8.

Q-learning is classified as:

a) On-policy TD control
b) Off-policy TD control
c) Monte Carlo control
d) Dynamic programming

### Q9.

What happens if the reward is zero for every step and zero at the terminal state in a Q-learning problem?

a) The agent quickly finds the optimal path
b) The Q table values never change and the agent learns nothing
c) The agent learns to avoid all actions
d) The Q table converges to the discount factor

### Q10.

Which of the following is NOT a feature of Stable Baselines 3?

a) Vectorized environments
b) Callbacks for monitoring and saving
c) Built-in Q-table implementations
d) Standard RL algorithm implementations (DQN, PPO, etc.)

---

## Part C: Short Answer

### Q11. Reward vs. Value

Explain the difference between **reward** and **value** in reinforcement learning. Give one concrete example of each using the cliff walking environment.

### Q12. Exploration vs. Exploitation

Define exploration and exploitation. Why must an RL agent balance both? In the epsilon-greedy policy, which parameter controls this balance?

### Q13. Markov State Design

A particle is moving through space. Is the state Markov if the state consists of only the particle's position? Why or why not? What would you add to make it Markov?

### Q14. Why Terminal States Are Zero

In the Q-learning algorithm, the Q values for terminal states are initialized to zero and never updated. Explain why this is correct using the definition of the value function.

### Q15. Episodes vs. Epochs

Why is it incorrect to call a run of the agent-environment loop an "epoch"? What is the correct term in RL, and how does it differ conceptually from an epoch in supervised learning?

### Q16. Three Types of Machine Learning

Name the three types of machine learning. For each, state what kind of feedback the learner receives.

### Q17. Gymnasium Definition

Give a definition of Gymnasium (2 sentences max). Name three methods that every Gymnasium environment must implement.

### Q18. What is a Gymnasium Wrapper?

Explain what a Gymnasium wrapper is and give one concrete example from the course of when a wrapper is needed.

### Q19. Stable Baselines 3

Give a one-sentence definition of Stable Baselines 3. Name two of its key features.

### Q20. Q-Learning vs. DQN

When and why would you use DQN instead of Q-learning? What replaces the Q table in DQN?

---

## Part D: Q-Learning Algorithm

### Q21. Write the Q-Learning Update Rule

Write the Q-table update portion of the Q-learning algorithm in **Python syntax**. Then list each variable used and explain its meaning.

### Q22. Step Size Analysis

Given the Q-learning update rule:

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

a) What happens when $\alpha = 1$? Show the simplification algebraically.
b) What happens when $\alpha$ is close to 0?
c) What does $\alpha = 0.5$ mean intuitively?

### Q23. Discount Factor

In the return formula $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots$:

a) What is the return measuring?
b) What happens as $\gamma$ approaches 1? As it approaches 0?
c) Why is a discount factor necessary for continuing (non-terminating) tasks?

### Q24. Q-Learning vs. SARSA on Cliff Walking

Q-learning and SARSA learn different paths on the cliff walking environment. Describe the path each learns and explain **why** they differ. Use the terms "on-policy" and "off-policy" in your answer.

---

## Part E: Environments and Code

### Q25. Gymnasium step() Return Values

The old OpenAI Gym `step()` method returned 4 values. Gymnasium's `step()` returns 5 values. List all 5 return values and explain the difference between `terminated` and `truncated`.

### Q26. Building a Q-Table

You are given a new Gymnasium environment. Write the Python code to:

a) Query the number of states and actions from the environment
b) Create an appropriately sized Q-table initialized to zeros

### Q27. Epsilon-Greedy Policy

Write the Python code for an epsilon-greedy action selection given a Q-table, a current state, and an epsilon value. Explain what each branch does.

### Q28. Blocks World States

In the 3-block, 4-position Blocks World, the state "C24" means:
- A is on ___
- B is on ___
- C is on ___

How many approximate valid states exist for this configuration? Why does the 6-digit version of the state (current + target) lead to a much larger Q-table?

### Q29. Code Reading: What Does This Do?

```python
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dims = env.action_space.nvec
        self.action_space = gym.spaces.Discrete(np.prod(self.dims))

    def action(self, action):
        return np.unravel_index(action, self.dims)
```

a) What problem does this code solve?
b) Why is it needed for DQN?
c) What does `np.unravel_index` do?

### Q30. Reward Design

In the Blocks World environment, the reward structure is: $-1$ per step, $-10$ for an illegal action, $+100$ for reaching the target configuration.

a) Why is the per-step reward negative instead of zero?
b) What would happen if the per-step reward were $+1$?
c) What is reward hacking, and give one example of how it could occur in Blocks World?

---

## Part F: Conceptual / Applied

### Q31. Agent Taxonomy

An RL agent may have any combination of three components: a **policy**, a **value function**, and a **model**. Fill in the table:

| Agent Type | Policy? | Value Function? | Model? |
|---|---|---|---|
| Value-based | | | |
| Policy-based | | | |
| Actor-critic | | | |
| Model-free | | | |
| Model-based | | | |

Which type is our Q-learning agent?

### Q32. Model and Planning

In the Blocks World assignment, the environment contains a Prolog model, but the agent does not have a model.

a) What would happen if we gave the agent access to the Prolog model?
b) Would reinforcement learning still be needed? Why or why not?
c) When would a model-based approach break down even with a perfect model?

### Q33. Value Function Approximation

What happens in reinforcement learning when there is an unmanageable number of states? Explain the concept and name two algorithms from Stable Baselines 3 that address this problem.

### Q34. Vectorized Environments

What are vectorized environments in Stable Baselines 3? Why are they useful? If you have a machine with 12 CPU cores, what is a reasonable number of parallel environments to create?

### Q35. Observation vs. State

Explain the difference between the environment state and the agent's observation. Give an example from a real-world RL problem where the observation is a subset of the full state.

### Q36. Random Seeds

Why is controlling the random seed important in reinforcement learning experiments? What can go wrong if you do not control it?

---

## Answer Key

### A1.
```
Agent  ──Action A_t──>  Environment
  ^                        |
  |                        |
  +── R_{t+1}, S_{t+1} ───+
```
The agent observes state $S_t$, selects action $A_t$, and sends it to the environment. The environment transitions to a new state and returns reward $R_{t+1}$ and new state $S_{t+1}$. The cycle repeats at the next timestep.

### A2.
The optimal path runs along the bottom row (one row above the cliff) from start to goal. It takes **13 steps**.

### A3. b)
### A4. b)
### A5. c)
### A6. c)
### A7. b)
### A8. b)
### A9. b)
### A10. c)

### A11.
**Reward** is a scalar value received after a single action (immediate, one time step). In cliff walking, the reward is $-1$ per step or $-100$ for falling off the cliff. **Value** is the expected cumulative total reward from a state to the end of the episode (forward-looking). In cliff walking, the value of the state just left of the goal is approximately $-1$ (one step to go), while a state far from the goal has a larger negative value.

### A12.
**Exploration**: trying new, possibly suboptimal actions to discover better strategies. **Exploitation**: always choosing the best known action. The agent must balance both because too much exploitation means it may never discover a better path, and too much exploration wastes time on suboptimal actions. The parameter **epsilon** ($\epsilon$) controls this: it is the probability of taking a random action.

### A13.
No, position alone is not Markov. You cannot predict the next position without knowing the direction and speed of travel. Adding **velocity** (speed and direction) to the state makes it Markov, because position + velocity is sufficient to predict the next position.

### A14.
The value function $Q(s, a)$ represents the expected future reward from state $s$. At a terminal state, there is no future (the episode is over), so the expected future reward is zero by definition. These values are already correct and never need updating.

### A15.
An **epoch** is a supervised learning term meaning one pass over the entire training dataset. In RL, we use **episode**, which means one complete run of the agent-environment interaction loop. An episode may vary in length and does not involve a fixed dataset. RL is a distinct third form of machine learning, and Sutton uses different terminology deliberately.

### A16.
1. **Supervised learning**: labeled examples (correct answer provided for each input)
2. **Unsupervised learning**: no feedback or labels
3. **Reinforcement learning**: scalar reward signal (delayed, evaluative)

### A17.
Gymnasium is an API standard for reinforcement learning with a diverse collection of reference environments. It provides a framework for creating RL environments with a standard interface so that various agents can be applied uniformly. Three required methods: `reset()`, `step(action)`, `render()` (also `close()` and `__init__`).

### A18.
A wrapper is a convenient way to modify an existing environment without altering the underlying code directly. Example: the `DiscreteActionWrapper` wraps the Blocks World environment (which uses tuple actions) and converts its MultiDiscrete action space to a Discrete action space so DQN can use it.

### A19.
Stable Baselines 3 is a set of reliable reinforcement learning algorithm implementations. Two key features: (1) **vectorized environments** for training on multiple environments in parallel, and (2) **callbacks** for monitoring, auto-saving, and custom code during training.

### A20.
Use DQN when the state space is too large for a Q-table (e.g., millions or billions of states). A **neural network** replaces the Q table, approximating the Q function instead of storing it exactly.

### A21.
```python
qtable[state][action] = qtable[state][action] + alpha * (reward + gamma * max(qtable[next_state]) - qtable[state][action])
```

| Variable | Meaning |
|---|---|
| `qtable` | The table of action-values implementing the Q function |
| `state` | The current state |
| `action` | The action taken in the current state |
| `alpha` | Step size (controls how much old value to keep vs. new) |
| `reward` | The reward received from taking the action |
| `gamma` | Discount factor (how much future rewards matter) |
| `next_state` | The state resulting from taking the action |

### A22.
a) When $\alpha = 1$: $Q(s,a) \leftarrow Q(s,a) + 1 \cdot [r + \gamma \max Q(s',a') - Q(s,a)] = r + \gamma \max Q(s',a')$. The old Q value is completely discarded.
b) When $\alpha \approx 0$: the update is tiny, so learning is very slow. Old values are almost entirely preserved.
c) $\alpha = 0.5$ means keep 50% of the old value and blend in 50% of the new information.

### A23.
a) The return $G_t$ measures the total expected future reward from time $t$ onward.
b) As $\gamma \to 1$, the agent becomes more farsighted (values future rewards almost as much as immediate). As $\gamma \to 0$, the agent becomes myopic (only cares about immediate reward).
c) Without discounting, the return for a non-terminating task would be infinite ($R + R + R + \dots = \infty$). Discounting ensures the sum converges.

### A24.
**Q-learning** (off-policy) learns the optimal path right along the cliff edge (13 steps). It updates using $\max Q(s', a')$, always assuming the best next action regardless of what the epsilon-greedy policy might actually do. **SARSA** (on-policy) learns a safer path away from the cliff. It updates using the action the policy *actually selects next*, which includes random exploratory actions. Near the cliff, a random action could step off the cliff ($-100$), so SARSA learns to stay away.

### A25.
`observation, reward, terminated, truncated, info = env.step(action)`

- **terminated**: the episode ended naturally (e.g., reached the goal)
- **truncated**: the episode ended prematurely for an external reason (e.g., hit a step limit, went off track)

### A26.
```python
n_states = env.observation_space.n
n_actions = env.action_space.n
Q_table = np.zeros((n_states, n_actions))
```

### A27.
```python
if np.random.random() < epsilon:
    action = env.action_space.sample()   # Explore: pick a random action
else:
    action = np.argmax(Q_table[state])   # Exploit: pick the best known action
```

### A28.
- A is on **C**
- B is on **position 2**
- C is on **position 4**

Approximately **90-120** valid states. The 6-digit version pairs every possible current state with every possible target state, giving $\sim90 \times 90 = 8{,}100$ states instead of 90. This makes the Q-table about 90 times larger.

### A29.
a) It converts a `MultiDiscrete` action space (tuple actions like (block, destination)) into a `Discrete` action space (single integers).
b) DQN requires a `Discrete` action space. It cannot work with tuple actions directly.
c) `np.unravel_index` converts a flat integer index back into a tuple of coordinates (e.g., integer 7 might become (1, 3) for a 2x4 grid). It reverses the flattening.

### A30.
a) Negative per-step reward encourages the agent to reach the goal in fewer steps, since each extra step costs $-1$.
b) With $+1$ per step, the agent is incentivized to wander forever collecting reward rather than reaching the goal.
c) **Reward hacking** is when the agent finds a way to collect rewards without making real progress. Example: in Blocks World with $+50$ for placing a block at its goal and $-50$ for moving it away, the agent could move a block to the goal ($+50$), move it away ($-50$), then back ($+50$), repeating forever to collect net-zero reward while never finishing.

### A31.

| Agent Type | Policy? | Value Function? | Model? |
|---|---|---|---|
| Value-based | Implicit (derived from V) | Yes | No |
| Policy-based | Yes | No | No |
| Actor-critic | Yes (actor) | Yes (critic) | No |
| Model-free | Maybe | Maybe | No |
| Model-based | Maybe | Maybe | Yes |

Our Q-learning agent is **value-based** and **model-free**. It has a value function (Q table) and derives its policy from it, with no model of the environment.

### A32.
a) The agent could query the Prolog model to know all legal actions and could plan a path directly to the target using Prolog's logical inference.
b) No, reinforcement learning would not be needed. Prolog can compute the exact action sequence to reach any target through planning.
c) When the state space is enormous (high dimensionality), planning becomes computationally infeasible — it would take longer than the age of the universe.

### A33.
When the state space is too large for a Q-table, we use **value function approximation**: replacing the Q table with a function approximator (typically a neural network) that estimates Q values. Two algorithms: **DQN** (Deep Q-Network) and **PPO** (Proximal Policy Optimization).

### A34.
Vectorized environments run multiple independent copies of the environment simultaneously, giving the agent N times as much training data per timestep. They are useful because they dramatically speed up training. With 12 CPU cores, creating approximately **10-12** parallel environments is reasonable (one per core, leaving some headroom for the OS).

### A35.
The **environment state** is everything going on in the environment (including hidden internals). The **observation** is the subset the agent can see. Example: in a floor-walking robot, the environment state includes the structural integrity of the floor beneath the tiles, but the agent only observes its tile position. Or in Atari: the full game state includes internal RAM values, but the agent only observes the pixel frames.

### A36.
Controlling the random seed ensures **reproducibility**: the same seed produces the same sequence of pseudo-random numbers, so training results can be exactly replicated. Without it, you cannot reproduce a good result (you won't know what seed produced it), making it impossible to verify findings or compare experiments fairly.
