# Sample Midterm Questions

## Section A: Short Answer / Definitions

**Q1: What is the Markov Property?**
A: The probability of transitioning to the next state depends only on the current state, not on any previous states. The future is independent of the past given the present.

**Q2: What is the Reward Hypothesis?**
A: All goals can be described as the maximization of expected cumulative reward.

**Q3: What is the difference between a state-value function V(s) and an action-value function Q(s, a)?**
A: V(s) gives the expected return from state s under a given policy. Q(s, a) gives the expected return from state s after taking action a, then following a given policy. V depends only on the state; Q depends on both the state and the action.

**Q4: What is an episode in the context of RL?**
A: A complete sequence of agent-environment interaction from a start state to a terminal state (e.g., one full game, one maze traversal).

**Q5: What is a return?**
A: The cumulative sum of discounted future rewards: Gt = Rt+1 + γRt+2 + γ²Rt+3 + ...

**Q6: What is the discount factor (gamma) and why is it needed?**
A: Gamma (γ) is a value between 0 and 1 that weights future rewards. It prevents returns from going to infinity in continuing (non-terminating) tasks. When γ is close to 1, the agent is farsighted; when γ is small, it focuses on immediate rewards.

**Q7: What is a policy in the context of RL?**
A: A mapping from states to actions. In general, it is a distribution over actions given a state: π(a|s) = P[At = a | St = s]. A deterministic policy maps each state to a single action: π(s) = a.

**Q8: What is a model in the context of RL agents?**
A: The agent's internal representation of the environment. It allows the agent to predict next states (transition model) and rewards (reward model), enabling planning. Not all agents include a model.

**Q9: Name the three possible components of an RL agent.**
A: Policy, Value Function, and Model.

**Q10: Reinforcement learning is often described as a third form of machine learning. What are the other two, and how does RL differ?**
A: Supervised learning (learns from labeled examples) and unsupervised learning (finds structure without labels). RL differs because the agent learns from a scalar reward signal through trial-and-error interaction with an environment, receiving delayed evaluative feedback rather than direct instruction.

---

## Section B: Gymnasium & Stable Baselines 3

**Q11: What is Gymnasium?**
A: An API standard for reinforcement learning with a diverse collection of reference environments. It provides a standard interface so that various RL algorithms/agents can be applied to any environment in a uniform way.

**Q12: Name three methods that every Gymnasium environment must implement.**
A: `reset()`, `step(action)`, and `render()` (also acceptable: `close()` or `__init__`).

**Q13: What does the `step()` method return in Gymnasium?**
A: Five values: `observation, reward, terminated, truncated, info`.

**Q14: What is the difference between `terminated` and `truncated` in Gymnasium?**
A: `terminated` means the episode ended normally (the agent reached a goal or terminal state). `truncated` means the episode ended early for a non-standard reason (e.g., exceeding a time/step limit, going off-track in a race).

**Q15: What is a Gymnasium wrapper?**
A: A convenient way to modify an existing environment without altering the underlying code directly. You initialize a base environment, then pass it to the wrapper's constructor. Example: a DiscreteActionWrapper converts MultiDiscrete actions to Discrete actions so DQN can use the environment.

**Q16: What is Stable Baselines 3?**
A: A set of reliable reinforcement learning algorithm implementations. Key features include vectorized environments (training on multiple copies of an environment simultaneously) and callbacks (mechanisms for monitoring, auto saving, progress bars, etc.).

**Q17: Name two features of Stable Baselines 3.**
A: (1) Vectorized environments — running the algorithm on several copies of the environment at the same time. (2) Callbacks — giving the programmer mechanisms to run custom code for monitoring, auto saving, model manipulation, progress bars, etc.

**Q18: What is a vectorized environment?**
A: A method for stacking multiple independent environments into a single environment. Instead of training on 1 environment per step, the agent trains on n environments per step, increasing training throughput.

**Q19: When would you use `MlpPolicy` vs. `MultiInputPolicy` in Stable Baselines 3?**
A: Use `MlpPolicy` when the observation space is flat/numeric (e.g., `Discrete`, `Box` with a simple array). Use `MultiInputPolicy` when the observation space is a dictionary (e.g., `spaces.Dict` with separate "current" and "target" entries).

---

## Section C: Q-Learning

**Q20: Write down the Q-table update rule in Python syntax and explain each variable.**
A:
```python
qtable[state][action] = qtable[state][action] + alpha * (reward + gamma * max(qtable[next_state]) - qtable[state][action])
```
- `qtable`: the table of action-values implementing the action-value function
- `state`: the current state
- `action`: the current action
- `alpha`: step size (controls how much new information overrides old values)
- `reward`: reward received from taking action in state
- `gamma`: discount factor (how much future rewards matter)
- `next_state`: the state resulting from taking action in state

**Q21: What does the step size (alpha) control? What happens when alpha = 1?**
A: Alpha controls how much of the old Q-value is retained vs. replaced by the new estimate. When alpha = 1, the old Q-value is completely overwritten with the new value. When alpha < 1, some of the old value persists, providing smoother, more stable learning.

**Q22: How should the Q-table be initialized according to the textbook algorithm?**
A: Initialize Q(s, a) to arbitrary values for all states and actions, except set Q(terminal, a) = 0 for all actions at terminal states. Terminal states have zero expected future reward because there is no future from the terminal state.

**Q23: Why does the Q-learning CliffWalking example converge on the shortest path along the cliff edge?**
A: Q-learning is off-policy. It updates the Q-table using the max action value for the next state, regardless of what action the epsilon-greedy policy actually selects. This means it learns the optimal policy (shortest path) without accounting for the risk of random exploration steps.

**Q24: Why does SARSA converge on a different (safer) path than Q-learning in CliffWalking?**
A: SARSA is on-policy. It updates the Q-table using the action actually chosen by the epsilon-greedy policy (which may be random). Near the cliff edge, a random exploratory action could mean falling off (−100 penalty). SARSA accounts for this risk and learns a safer path away from the cliff.

**Q25: What is the difference between on-policy and off-policy learning?**
A:
- **On-policy** (SARSA): Updates the value function based on the action the policy actually takes next (including random exploratory actions).
- **Off-policy** (Q-learning): Updates the value function based on the best possible next action (max Q), regardless of what action was actually taken.

**Q26: What is epsilon-greedy exploration?**
A: With probability (1 − ε), choose the action with the highest Q-value (exploit). With probability ε, choose a random action (explore). This balances exploitation of known good actions with exploration of potentially better alternatives.

**Q27: What happens if all rewards in an environment are zero (including the terminal state)?**
A: No learning occurs. The Q-table update produces no changes because all updates compute 0 + γ × max(0, 0, ...) − 0 = 0. There is no signal to distinguish one path from another, so the agent never learns to prefer any action. Only epsilon-greedy randomness occasionally breaks the agent out of loops.

**Q28: What happens if the step reward is +1 instead of −1 per step in CliffWalking?**
A: The agent is incentivized to wander forever, collecting +1 at each step rather than reaching the goal and ending the episode. Positive step rewards discourage termination.

---

## Section D: Diagrams

**Q29: Draw the RL agent-environment interaction diagram and label all signals.**
A:
```
         ┌──────────┐
         │  Agent   │
         └────┬─────┘
          A_t │ ▲ S_{t+1}, R_{t+1}
              │ │
              ▼ │
         ┌────┴─────┐
         │Environment│
         └──────────┘
```
- The agent sends action A_t to the environment
- The environment returns reward R_{t+1} and new state S_{t+1} to the agent
- This cycle repeats at every timestep

---

## Section E: Value Function Approximation & DQN

**Q30: What happens in reinforcement learning if there is an unmanageable number of states?**
A: We go from exact value function learning (Q-table) to learning an approximation of the value function using a neural network. This is called value function approximation. DQN (Deep Q-Network) is an example that uses a neural network to approximate the Q function.

**Q31: What are the four key components of DQN?**
A:
1. **Q-Network (Policy)**: A neural network that takes the state as input and outputs Q-values for each possible action.
2. **Target Network**: A slowly updated copy of the Q-network that provides stable targets for training.
3. **Replay Buffer**: Stores past (s, a, r, s') transitions and samples random mini-batches to break correlation between consecutive samples.
4. **Epsilon-Greedy Exploration**: Balances exploration (random actions) and exploitation (best-predicted action).

**Q32: Why does DQN use a target network?**
A: To stabilize training by preventing the network from "chasing its own tail." The target network is updated less frequently, providing stable Q-value targets for the loss calculation.

**Q33: Why does DQN require a Discrete action space?**
A: DQN outputs Q-values for every possible action simultaneously. This requires a finite, enumerable set of actions. If the environment uses MultiDiscrete actions, a DiscreteActionWrapper must be applied to flatten the action space into a single integer.

**Q34: Compare Q-learning and DQN.**
A:
| Aspect | Q-Learning | DQN |
|--------|-----------|-----|
| Value storage | Q-table (explicit lookup) | Neural network (function approximation) |
| State input | Table row index | Network input vector |
| Update | Direct table cell update | Gradient descent on network weights |
| Scalability | Limited by table size | Scales to large state spaces |

The underlying logic (Bellman equation, epsilon-greedy) is largely the same.

**Q35: What is TensorBoard used for?**
A: TensorBoard is a comparative graphing facility for monitoring and comparing training experiments. It plots metrics like mean reward, episode length, and exploration rate over training steps, and allows comparison across multiple training runs with different hyperparameters.

---

## Section F: Environment Design & Reward Shaping

**Q36: In the Blocks World environment, what is the reward structure?**
A:
| Condition | Reward |
|-----------|--------|
| Legal action taken (per step) | −1 |
| Impossible action attempted | −10 |
| Target configuration achieved | +100 |

The −1 per step encourages shorter solutions. The −10 discourages illegal moves. The +100 signals goal achievement.

**Q37: What is reward shaping?**
A: The process of designing a reward function that guides the agent toward desired behavior more effectively than a simple sparse reward. For example, giving intermediate rewards for partial progress rather than only rewarding the final goal.

**Q38: What is reward hacking?**
A: When the agent finds a way to collect rewards without making real progress. Example: repeatedly moving a block to its target (+50) then away (−50) then back (+50) in a loop. Strategies to prevent it include asymmetric rewards and directional biases.

**Q39: Why do we limit episode length (e.g., 200 steps) in Blocks World training?**
A: Without a limit, early episodes could run for thousands of steps with the agent making useless moves (e.g., moving blocks back and forth). Truncation forces variety by exposing the agent to many different starting/target configurations, and prevents wasting compute on unproductive episodes.

**Q40: What is the difference between 3-digit and 6-digit state representations in Blocks World?**
A:
- **3-digit**: Only the current block configuration. ~90 states. The agent does not know the target, so it wanders randomly until it accidentally matches.
- **6-digit**: Current configuration + target configuration. ~8,100 states (90²). The agent knows the goal and can learn directed, goal-seeking behavior. Training takes much longer due to the larger Q-table.

---

## Section G: Conceptual / Tricky Questions

**Q41: Why is RL terminology different from supervised learning (e.g., "episodes" not "epochs," "step size" not "learning rate")?**
A: Sutton deliberately uses distinct terms to establish RL as a separate, third form of machine learning — not supervised, not unsupervised. The concepts are related but not identical. Saying "epochs" in an RL context is technically incorrect; the correct term is "episodes."

**Q42: Can the same Q-learning agent work on different environments?**
A: Yes. The same Q-learning algorithm works on CliffWalking, Taxi, Blocks World, etc. Only the environment name changes. The agent queries observation_space.n and action_space.n to set up the Q-table, then learns through the same update rule. Agents and environments are independent.

**Q43: Where do hyperparameters belong — in the agent or the environment?**
A: In the agent. Hyperparameters (alpha, epsilon, gamma) are properties of the learning algorithm, not the problem domain. A common mistake is putting them in the environment class.

**Q44: What is the scalar reward signal? Why is it always a scalar?**
A: The reward in RL is always a single number (not a vector). This is sufficient because the return is defined as a sum of rewards (producing a scalar), and the value function outputs a scalar. All of RL, including complex applications like protein folding, works with a scalar reward signal.

**Q45: If you gave the Blocks World agent a complete Prolog model of the environment, would it still need reinforcement learning?**
A: No. With a complete model, the agent could plan directly — Prolog would compute the exact action sequence to reach any target state. RL is only needed when the agent lacks a complete model and must learn through trial and error. However, at very large scale, even planning becomes computationally infeasible.

**Q46: What error do you get if you forget to adapt from dictionary observations to Discrete observations when switching environments?**
A: "Discrete object is not subscriptable" — because the code tries to index into a Discrete observation as if it were a dictionary.

**Q47: What is the role of the `seed` parameter in `env.reset(seed=42)`?**
A: It sets the random seed for reproducibility. The same seed always produces the same sequence of pseudo-random numbers, so training results can be exactly reproduced later.

**Q48: Given the following code, identify what it does:**
```python
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dims = env.action_space.nvec
        self.action_space = gym.spaces.Discrete(np.prod(self.dims))
    def action(self, action):
        return np.unravel_index(action, self.dims)
```
A: This is a Gymnasium wrapper that converts a MultiDiscrete action space into a Discrete action space. It computes the total number of discrete actions by multiplying the dimensions, then converts a single integer action back into a tuple using `np.unravel_index`. This is needed because DQN requires a Discrete action space.
