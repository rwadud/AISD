# Reinforcement Learning Lecture 4: Blocks World Assignment, Q-Learning Across Environments, and Hyperparameter Experiments

---

## 1. Gym vs. Gymnasium Recap

> **Lecture preview:** This lecture also briefly introduces Stable Baselines 3, which will be explored in more detail later.

OpenAI created the **Gym** library as the standard interface for RL environments. The old version is called **Gym**. The new successor library is called **Gymnasium**.

The standard migration pattern:

```python
import gymnasium as gym
```

This is designed to be a drop-in replacement. Any existing code that depends on `gym` can remain unchanged. You just change the import at the top to `import gymnasium as gym`, and all subsequent code that references `gym` will actually use the new Gymnasium library.

---

## 2. Assignment 1: Blocks World Environment

> **Course note:** Assignment 1 is the Blocks World. It builds directly on the skills from Lab 2 (custom environment creation). The Blocks World environment uses Prolog as its backend logic engine, but deep Prolog knowledge is not required. If you can follow and run the provided instructions, that should be sufficient.

### 2.1 What You Receive

You start with a GitHub repository (named something like `BW-<your-github-username>`) that contains:

- **Prolog code** (`bworld.pl`) with the Blocks World logic, including additional predicates added to support using it as an RL environment
- **PyGame code** (`screen.py`) for rendering the blocks visually
- **Block images** for the PyGame display

### 2.2 What You Create

Your job is to build two things:

1. **A Gymnasium environment** for the Blocks World, based on the Prolog backend
2. **A Q-learning agent** adapted to work with discrete observations (instead of the dictionary observations from Lab 2)

This follows the same process as Lab 2. You created a custom environment for CliffWalking in Lab 2, and now you are creating one for Blocks World in Assignment 1. The same copier command is used to scaffold the starter package.

### 2.3 Folder Structure

The assignment document spells out the expected folder layout:

```
assignment1/
├── .venv/                    # Dedicated virtual environment
└── BW-<github-username>/     # Your checked-out GitHub repo
    ├── .git/
    ├── images/               # Block pictures
    ├── screen.py             # PyGame display (provided)
    ├── bworld.pl             # Prolog Blocks World (provided)
    ├── blocksworld_env/      # Your environment package
    │   └── ...
    └── agent.py              # Your Q-learning agent
```

*(reconstructed from lecture description)*

---

## 3. Interfacing Prolog with Python via pyswip

### 3.1 What Is pyswip?

**pyswip** is a Python library that allows you to run SWI-Prolog from within Python. It starts up a Prolog interpreter, creates a thread, and lets you issue Prolog queries programmatically.

### 3.2 Loading the Blocks World

```python
from pyswip import Prolog

prolog = Prolog()
list(prolog.query("[bworld]"))  # Load bworld.pl (consult)
```

*(reconstructed)*

Key differences from interactive Prolog:

| Interactive Prolog | pyswip in Python |
|---|---|
| `[bworld].` | `prolog.query("[bworld]")` |
| Query ends with a period (`.`) | **No period** at the end of the query string |
| You press `;` to get more answers | All answers are returned automatically |
| You see the Prolog prompt `?-` | No prompt. Results come back as Python data |

The `prolog.query()` method automatically retrieves **all** solutions. In interactive Prolog, you would press semicolon (`;`) to get each successive answer. pyswip does this for you, returning every result at once.

### 3.3 Query Results Are Lists of Dictionaries

Results come back as JSON-like Python structures: a **list of dictionaries**, where each dictionary maps variable names to their values.

```python
# Querying all states
result = list(prolog.query("state(State)"))
# Result looks like:
# [{'State': 'BC2'}, {'State': 'BC3'}, {'State': 'AC1'}, ...]
```

*(reconstructed)*

Each three-digit string (like `'BC2'`, `'BC3'`, `'AC1'`) represents a unique configuration of the three blocks (A, B, C) across the available positions.

> **Two useful queries:** `state(State)` returns **all** possible states. A separate `current_state` query returns just the **current** block configuration at any point during an episode.

### 3.4 The Initial Blocks World Configuration

In the initial situation:

- Block A is on position 1
- Block B is on position 3
- Block C is on block A

You should verify your Prolog setup by running SWI-Prolog directly, issuing a query, and confirming the output matches what the assignment document shows. If it does, Prolog is working correctly with no syntax errors.

---

## 4. Building the State and Action Mappings

### 4.1 State Mapping: Strings to Integers

The Blocks World has approximately **90 discrete states**. Prolog represents each state as a three-character string (e.g., `'BC2'`). Q-learning needs integer state indices (0, 1, 2, ..., 89).

**Solution:** Build a Python dictionary that maps each string to an integer:

```python
# After querying all states from Prolog
result = list(prolog.query("state(State)"))

# Build forward lookup: string -> integer
state_to_int = {entry['State']: i for i, entry in enumerate(result)}

# Build reverse lookup: integer -> string
int_to_state = {i: entry['State'] for i, entry in enumerate(result)}
```

*(reconstructed)*

- **Forward lookup** (`state_to_int`): Given a string like `'EC2'` from Prolog, find the corresponding integer index.
- **Reverse lookup** (`int_to_state`): Given an integer index, find the corresponding Prolog string.

### 4.2 Action Mapping: Prolog Functors to Integers

Actions are queried similarly. The result for each action contains an `args` field and a `functor` field:

```python
result = list(prolog.query("action(Action)"))
# Each result has: {'Action': {'functor': 'move', 'args': ['A', 'C']}}
# This corresponds to move(A, C) — move block A to position/block C
```

*(reconstructed)*

You construct the Prolog action string from the functor and args. For example, if `args` are `['A', 'C']` and `functor` is `'move'`, that corresponds to the Prolog term `move(A, C)`, meaning "move block A from wherever it is to C (assuming the move is legal)."

You build the same kind of bidirectional dictionaries for actions:

```python
action_to_int = {}   # Prolog action string -> integer
int_to_action = {}   # integer -> Prolog action string
```

*(reconstructed)*

The goal of all this mapping is to convert between **discrete integer states/actions** (what Q-learning works with) and **Prolog string representations** (what the Prolog backend works with). In particular, the Prolog action string (e.g., `move(A, C)`) is what gets issued to the **step function in Prolog** when the environment executes an action.

---

## 5. Environment Implementation Details

### 5.1 Rendering

If the `render_mode` is `"human"`, the environment initializes a PyGame display by instantiating the `Display` class from `screen.py` (which is provided). All the PyGame programming has already been done for you.

### 5.2 Testing with a Null Agent

Before hooking up Q-learning, you can test your environment with a **null agent**, an agent that does not learn. It simply creates an instance of the Blocks World environment, tries out some actions, and shows you what the result looks like. This lets you verify that the environment is working correctly (rendering, rewards, state transitions) before introducing the learning algorithm.

### 5.3 The `close()` Method

The Gymnasium `close()` method should **shut down the Prolog server**. This is important because pyswip runs a Prolog interpreter process that needs to be cleaned up when the environment is done.

```python
def close(self):
    # Shut down the Prolog server
    ...
```

*(reconstructed)*

### 5.4 Reward Structure

| Condition | Reward |
|---|---|
| Step predicate returns non-false (legal action taken) | $-1$ |
| Step predicate returns false (impossible action attempted) | $-10$ |
| Target configuration achieved (episode done) | $+100$ |

The $-1$ per legal step encourages the agent to reach the goal in fewer moves. The $-10$ penalty for impossible actions discourages trying illegal moves. The $+100$ terminal reward signals success.

---

## 6. Three-Digit States vs. Six-Digit States

This is the core design evolution of the assignment. There are two versions of the environment, and understanding why we move from one to the other is essential.

### 6.1 Version 1: Three-Digit States (No Target in Observation)

With three-digit states, the observation is just the **current block configuration** (e.g., `'BC2'`). There are approximately 90 unique states, mapped to integers 0 through 89.

**Q table size:** 90 states × ~120 actions = **~10,800 entries**

**Problem:** The agent does not know what the target configuration is. The target is generated randomly, and the agent just moves blocks around until it stumbles upon the target by chance. What the agent actually learns:

- Which actions are impossible (and should be avoided due to $-10$ penalty)
- How to move around "legitimately" (legal moves only)

What it does **not** learn: how to reach a specific target configuration. It has no concept of a goal. It just wanders until it randomly matches the target. This is like a robot flailing its arms randomly in episode one.

### 6.2 Version 2: Six-Digit States (Target Included in Observation)

To do better, we include the **target configuration** as part of the state. The observation now has six characters: the first three are the current configuration, and the second three are the target configuration.

**Q table size:** $90 \times 90$ state combinations × ~120 actions = **~972,000 entries** ($90^2 = 8100$ unique state pairs)

The 90 becomes $90^2$ because for each of the 90 possible current configurations, there are 90 possible targets. Each (current, target) pair occupies its own section of the Q table.

**Consequence:** Training takes **much longer** because:

- The Q table is 90 times larger
- Each training episode only updates values in the section corresponding to the current target
- When a new target is randomly generated, the agent moves to a different section of the table
- If the same target comes up again later, the agent resumes training in that target's section

**Benefit:** The agent can now learn to **actively navigate** toward the target. As episodes continue, it gets better and better at reaching the target efficiently, eventually doing it in three or four steps most of the time. In the lecture demo, by episode eight the agent had noticeably sped up compared to episode one. This is much more satisfying than the random wandering of Version 1.

> **Key insight:** Including the target in the state is what gives the agent the ability to learn goal-directed behavior. Without it, the agent can only learn general navigation skills, not how to reach a specific goal.

### 6.3 Comparison Table

| Property | Three-Digit States | Six-Digit States |
|---|---|---|
| Observation | Current config only | Current config + target config |
| Number of states | ~90 | ~8,100 ($90^2$) |
| Q table size | ~90 × 120 | ~8,100 × 120 |
| Agent knows goal? | No | Yes |
| Learns goal-directed behavior? | No | Yes |
| Training time | Faster (smaller table) | Slower (larger table) |
| Quality of learned policy | Wanders randomly to goal | Navigates efficiently to goal |

---

## 7. Adapting the Agent from Dictionary to Discrete Observations

### 7.1 The Key Difference from Lab 2

In Lab 2 (CliffWalking), the `reset()` and `step()` methods returned observations as **dictionaries** (e.g., `{'current': (x, y), 'target': (tx, ty)}`). The agent had to extract the state number from this dictionary.

In the Blocks World (and in Gymnasium's built-in environments), the observation space is **Discrete**, meaning `reset()` and `step()` return a plain integer state number directly.

### 7.2 What to Change in Your Agent

There is basically **one place** where you need to adjust your agent code:

```python
# Lab 2 (dictionary observation):
obs, info = env.reset()
state = calculate_state_from_dict(obs)  # Had to extract/compute state number

# Assignment 1 / Discrete observation:
obs, info = env.reset()
state = obs  # obs IS the state number directly — no conversion needed
```

*(reconstructed)*

The same change applies inside the training loop where `env.step(action)` returns the next observation. Instead of extracting the state from a dictionary, you receive the integer directly.

### 7.3 Adapting to Gymnasium's Built-in CliffWalking

When switching from our custom CliffWalking environment to Gymnasium's built-in `CliffWalking-v0`, the same adaptation is needed:

1. Change the environment name from the custom namespace to `"CliffWalking-v0"`
2. The observation space changes from `Box`/`Dict` to `Discrete`
3. Read `num_states` from `env.observation_space.n` (just like `num_actions` from `env.action_space.n`)
4. Remove/comment out the dictionary-to-state conversion code
5. Receive states directly as integers from `reset()` and `step()`

The error you get if you forget to adapt: **"Discrete object is not subscriptable"**, because the code tries to subscript into a Discrete observation as if it were a dictionary.

---

## 8. Q-Learning Works Across Many Environments

A central lesson of this lecture is that the **same Q-learning agent code** can be applied to completely different environments with no changes to the algorithm. Only the environment name changes. The agent and the environment are **independent** of each other. The agent embodies the algorithm. The environment embodies the problem domain.

### 8.1 Environments Demonstrated

| Environment | Type | Observation | Actions | Notes |
|---|---|---|---|---|
| Custom CliffWalking | Grid World | Dictionary → Discrete | 4 (up/down/left/right) | Built in Lab 2 from scratch |
| `CliffWalking-v0` | Gymnasium built-in | Discrete (48 states) | 4 | Gymnasium's official version, different visual style |
| `Taxi-v3` | Gymnasium built-in | Discrete (500 states) | 6 (4 movement + pickup + dropoff) | Taxi must pick up a passenger and drop them at a destination |
| Monkey and Banana | Custom | Discrete | Domain-specific | Agent must get the banana to end the episode |
| Coffee World | Custom | Discrete (fluents) | Domain-specific | Fluents: cupboard, open, coffee, stirred. Episode ends when coffee is stirred |
| Blocks World | Custom (Prolog) | Discrete | ~120 | Assignment 1 environment |

In each case, the **exact same Q-learning code** was used. The only changes were the environment name and, if necessary, adapting from dictionary to discrete observations.

> **Key takeaway:** Hyperparameters belong in the agent, not in the environment. A common mistake is putting hyperparameters inside the environment class. Hyperparameters (learning rate, epsilon, gamma, etc.) are properties of the learning algorithm, so they belong in the agent.

### 8.2 The Taxi Environment

`Taxi-v3` is a Gymnasium built-in environment with:

- A grid-based world with a taxi (the agent)
- Actions: four movement directions plus **pickup** and **dropoff**
- The goal: pick up a passenger at one location and drop them off at a specific building
- Both the action space and observation space are discrete, making it a valid Q-learning domain

The taxi behaves randomly at first and takes a long time to complete the task. Over many episodes, Q-learning teaches it efficient routes.

---

## 9. Visualizing the Q Table in Grid World

Grid World (4 × 12 for CliffWalking) is uniquely suited for Q table visualization because:

- The world shape is simple and grid-based
- There are only **four actions** per state (up, down, left, right)
- The Q table can be reshaped to match the exact layout of the world

### 9.1 Q Table Structure

The CliffWalking grid has 48 states (4 rows × 12 columns) and 4 actions per state:

$$Q \in \mathbb{R}^{48 \times 4}$$

Ten of these states are the cliff and are never actually visited, but they still exist in the Q table.

### 9.2 Visual Layout

Each cell in the 4 × 12 grid displays four action values arranged as directional triangles:

```
        ┌───────┐
        │  Up   │
        │       │
  Left  │       │  Right
        │       │
        │ Down  │
        └───────┘
```

*(reconstructed)*

Each triangle is color-coded by its Q value:

| Color | Q Value |
|---|---|
| Yellow | Close to +100 |
| Green | Close to 0 |
| Dark purple | Close to -100 |

### 9.3 What the Visualization Shows

- The Q table is plotted after **every single move** the agent makes
- The agent's current position is also shown on the same grid
- Since the Q table is the same shape as the world, watching the agent move is like watching it traverse the Q table itself

**Initial state:** Values are initialized randomly between 0 and 100 (all yellowish).

**After training begins:** When the agent falls off the cliff, the "down" action value at the cliff edge is updated to approximately $-100$ (turns dark purple), because the Bellman equation calculation incorporates the large negative reward.

> **Why we care about the Q table:** The Q table **is** the Q value function. The Q-learning algorithm optimizes this table to find optimal values. Once the Q table is optimal, it directly gives us the **optimal policy**, because the policy just selects the action with the highest Q value in each state.

> **Note:** This kind of per-cell, per-action visualization only works for environments with very few actions (like Grid World's 4). For Blocks World with ~120 actions per state, it would not be feasible.

### 9.4 Markov Chain Graph

It is also possible to create a Markov chain graph for any environment, showing state transitions, actions, and rewards for each episode. However, the Q table visualization is specifically designed to show how the **value function** evolves during training, which is a different (and arguably more useful) perspective.

---

## 10. Hyperparameter Experiments on CliffWalking

### 10.1 Key Hyperparameters

| Hyperparameter | Symbol | Role |
|---|---|---|
| Epsilon | $\epsilon$ | Degree of randomness (exploration vs. exploitation) |
| Gamma | $\gamma$ | Discount factor (how much future rewards matter) |
| Learning rate | $\alpha$ | How quickly new information overrides old Q values |

### 10.2 Experiment: Changing Gamma

**Default gamma (low, e.g., 0.1):** The future is heavily discounted. Action values quickly settle to reflect mostly the immediate reward ($-1$ per step). Cliff-edge "down" actions drop to approximately $-100$ almost immediately.

**Gamma = 0.9:** The future is **not** discounted as much. Observed effects:

- Values no longer settle to $-1$ as quickly
- Cliff-edge actions are still negative but **less extreme** than $-100$, because the calculation now incorporates future value from neighboring states
- **Convergence takes longer** because more of the future value is being propagated through the table, meaning more updates are needed for everything to stabilize

> **Why this happens:** The Q-learning update rule includes the term $\gamma \cdot \max_a Q(s', a)$. When $\gamma$ is small (0.1), we take only 10% of the best future value, so each state's value is dominated by its immediate reward. When $\gamma$ is large (0.9), 90% of the best future value propagates back, making values depend heavily on states far ahead.

The Q-learning update rule for reference:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

*(added)*

### 10.3 Experiment: Exploration and State Coverage

With the standard hyperparameters, almost all states are explored by the end of the first episode, and every state has been visited at least once by episode two.

**Why exploration matters:** We want every state to be explored because there might be a large reward hidden somewhere. The Q-learning agent does not know the structure of the environment in advance. If the agent zeroes in on the terminal state too quickly, it could miss a hypothetical "pot of gold" in another part of the grid. Epsilon-greedy exploration ensures that, with some probability, the agent tries random actions and discovers the full state space.

### 10.4 Experiment: Zero Reward Everywhere

Setting all rewards to zero (including the terminal state) demonstrates what happens when the agent has **no learning signal** at all.

**What happens:**

1. The Q table is initialized to all zeros
2. Every action in every state has the same Q value (zero)
3. The agent always picks the same action (e.g., the first one, which might be "left"), because `max(0, 0, 0, 0)` returns the first zero
4. The agent gets **stuck**, repeating the same action over and over
5. Only **epsilon-greedy randomness** occasionally breaks it out of the loop, causing it to try a different action by chance
6. No learning occurs because all updates are $0 + \gamma \cdot \max(0, 0, ...) - 0 = 0$

**With zero reward, there is nothing to distinguish one path from another.** Whether the agent takes 20 steps or 100 steps, the total reward is zero either way. There is no gradient of value to attract the agent toward the goal.

**With zero reward everywhere except the terminal state:**  If you give a reward of $+1$ (or any positive value) for the terminal state only, then the first time the agent stumbles upon the terminal state (via random exploration), that positive value will **propagate backward** through the Bellman equation, gradually creating a value gradient that attracts the agent toward the goal in future episodes.

**With zero reward including the terminal state:** Nothing ever changes. There is no value to propagate back. The agent never learns to prefer any direction over another.

> **Key insight:** The Q-learning algorithm is allowed to initialize Q values arbitrarily (zeros, random values, anything). What matters is the update rule. But if the rewards are all zero, the update rule produces no changes, so the initialization is all the agent ever has.

---

## 11. Scaling Beyond Q-Learning: Value Function Approximation

### 11.1 The Problem with Large State Spaces

With 3 blocks and 4 positions, we have ~90 states. The Q table fits comfortably in memory. But what if we scaled up to **10 blocks and 10 positions**?

The number of possible states becomes enormous. Too many states and actions to enumerate, too many to store in a Q table, too many for a computer to handle with tabular methods.

### 11.2 The Solution: Value Function Approximation

When the state space is too large for tabular Q-learning, we switch to **value function approximation**. Instead of storing Q values in a table, we use a function approximator (typically a **neural network**) to estimate Q values.

Algorithms that do this:

- **DQN** (Deep Q-Network): Uses a neural network to approximate the Q function
- **PPO** (Proximal Policy Optimization): A policy gradient method that also uses neural networks

### 11.3 Feature Representation

Neural networks don't accept a single integer as input (like "state 42"). They need **vectors of features**. For a 10×10 Blocks World, the state would need to be encoded as a feature vector that the neural network can process.

> **Course note:** How to represent large state spaces as feature vectors for neural networks will be covered in a later lecture.

---

## 12. Deploying a Trained Agent

After training is complete (e.g., after 50 episodes of Q-learning), the learned Q table contains the knowledge the agent has acquired. To deploy this agent:

1. **Save the Q table** to a file after training
2. **Load the Q table** at the start of the next run
3. The agent now behaves according to the learned policy without any further training

This is the same principle used by Stable Baselines 3 agents. They train for a set number of steps (e.g., 10,000), save the model weights, and then reload them later for inference. For Q-learning, the "model" is simply the Q table.

```python
import numpy as np

# After training
np.save("q_table.npy", q_table)

# At deployment
q_table = np.load("q_table.npy")
# Now always select: action = np.argmax(q_table[state])
```

*(reconstructed)*
