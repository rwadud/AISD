# Slide Questions & Answers

## Lecture 1: RL Intro

### "Time to check your learning!" (Slide 37)

**Q: What is the Markov Property?**
A: The probability of each possible value for St and Rt depends only on the immediately preceding state and action, St-1 and At-1.

**Q: What are the possible components in an RL agent?**
A: Policy, Value Function, and Model.

**Q: What is a policy in the context of RL?**
A: A function that maps state to action.

**Q: What is a value function in the context of RL?**
A: Represents the value (how good is it?) of each state or action. It gives the expected future reward from a state — it does not include rewards already received, only future reward.

**Q: What is a model in the context of RL?**
A: The agent's internal representation of the environment. It allows inferences about how the environment will behave — predicts next state (transition model) and next reward (reward model). Used for planning. Not all RL solutions include a model.

---

## Lecture 2: MDP

### "Time to check your learning!" (Slide 18)

**Q: What is a return?**
A: The expected cumulative sum of future reward.

**Q: What is the expression for the expected return at timestep t?**
A: The sum of discounted future rewards (Gt = Rt+1 + γRt+2 + γ²Rt+3 + ...).

**Q: What is the meaning of a state-value function? An action-value function?**
A: A state-value function gives the expected return when starting at a state under a given policy. An action-value function gives the expected return when starting at a state, taking a specific action, and then following a policy.

**Q: What is a policy in the context of RL?**
A: A mapping (function) from states to probabilities of selecting each possible action: π(a|s) = P[At = a|St = s]. A deterministic policy maps states directly to actions: π(s) = a.

**Q: What is an episode in the context of RL?**
A: A complete sequence of agent-environment interaction from start to a terminal state.

### Q-Learning Deep Dive Discussion Questions

**Q: Why does our CliffWalking Example converge on the shortest path?**
A: Because Q-learning is off-policy — it updates the Q-table using the max action value regardless of the exploration policy, so it converges on the optimal (shortest) path.

**Q: How does the reward (besides cliff) affect the eventual path?**
A: Negative reward per step favors shorter episodes. Zero reward has no step penalty. Positive reward could incentivize longer episodes.

**Q: How does the initialization of the Q-table affect convergence?**
A: Randomized vs. initialized to zero affects early exploration behavior and convergence speed.

**Q: Why is SARSA on-policy and Q-learning off-policy?**
A: Both use epsilon-greedy for action selection. The difference: SARSA updates the Q-table using the value of the *actual* next action taken (which may be random), while Q-learning updates using the *max* action value regardless of what action is actually taken.

---

## Lecture 5: DQN

**Q: How can we train DQN on a MultiDiscrete action space?**
A: Wrap it with a `DiscreteActionWrapper` that converts between a single integer action and the multi-dimensional action tuple using `np.unravel_index`.

---

## Lecture 7: Midterm Review

### Sample Written Questions

**Q: Why does our CliffWalking Example converge on the shortest path?**
A: Q-learning is off-policy — it updates using the max action value, so it converges on the optimal path.

**Q: Why does SARSA converge on a different path?**
A: SARSA is on-policy, so it accounts for the exploration noise in its updates, leading it to learn a safer path that avoids the cliff edge.

**Q: How does the initialization of the Q-table affect convergence?**
A: Randomized vs. zero initialization affects exploration. Setting terminal state action-values to zero is important.

**Q: Write down the Q-table update portion of the Q-learning algorithm. List each variable and its meaning.**
A:
```python
qtable[state][action] = qtable[state][action] + alpha * (reward + gamma * max(qtable[next_state]) - qtable[state][action])
```
- `qtable`: table of action-values implementing the action-value function
- `state`: current state
- `action`: current action
- `alpha`: step size
- `reward`: reward received from taking action in state
- `gamma`: discount factor
- `next_state`: the state resulting from taking action in state

**Q: Draw the diagram that represents the primary aspects of an RL problem/solution with agent-environment interaction.**
A: Agent sends Action (At) to Environment. Environment returns Reward (Rt+1) and State (St+1) to Agent. (Standard agent-environment loop diagram.)

**Q: What is Gymnasium?**
A: An API standard for reinforcement learning with a diverse collection of reference environments. A framework for creating RL environments with a standard interface such that various RL algorithms/agents can be applied in a standard way.

**Q: What is a Gymnasium wrapper?**
A: A convenient way to modify an existing environment without having to alter the underlying code directly. You initialize a base environment, then pass it to the wrapper's constructor.

**Q: What is Stable-Baselines3?**
A: A set of reliable RL algorithm implementations that includes features such as vectorized environments (running the algorithm on several copies of the environment simultaneously) and callbacks (monitoring, auto saving, model manipulation, progress bars, etc.).
