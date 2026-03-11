# Lecture 6: Deep Q Networks, Blocks World, and Reward Shaping

## Estimating State and Action Spaces for Blocks World

### Action Space Upper Bound

In a 5 by 5 Blocks World (5 blocks, 5 positions), each action is a **couple** of (block, destination). To find an upper limit on the number of actions:

- 5 blocks can each be moved to any of the 10 places (5 blocks + 5 positions), giving an upper bound of **50 possible actions**
- Not all 50 are possible at any given time (e.g., moving a block to itself is impossible), but 50 is a safe upper limit
- Removing impossible actions like moving a block to itself would give fewer combinations, but 50 is well within the range of what Q learning can handle

### State Space Upper Bound

The critical question is the number of **distinct states** in 5 by 5 Blocks World:

- The state space grows combinatorially as the number of blocks and positions increases
- Computing the exact number involves summing permutations (e.g., $5! + 4! + \dots$), which results in a very large number
- Using a discrete wrapper to empirically measure the state count, the number of states with 5 by 5 or 6 by 6 exceeds what a Q table can handle in memory
- A typical computer has 32 GB or 64 GB of RAM, corresponding to billions of storable integers. If the state count exceeds this, Q learning is infeasible

### State Representation

The environment encodes block configurations as an array of integers:

- The first $n$ natural numbers (0 through $n-1$) represent **blocks** (displayed as letters A, B, C, ...)
- The next $n$ natural numbers represent **positions** (displayed as numbers)
- Each entry in the state array tells you where a given block is sitting (on another block or on a position)

**Example** *(reconstructed)*: In a 10 by 10 world, if the state array shows A's value is 3, that means A is on block D (block index 3). If B's value is 11, that means B is on position 2 (since 11 > highest block index, it refers to a position).

> **Course note**: This encoding is not human readable. The environment is smart enough to display blocks as letters and positions as numbers in its rendered output, even though internally everything is integer indices.

---

## From Q Learning to Value Function Approximation

### Why Q Learning Has Limits

- If we can do Q learning (manageable number of states and actions), Q learning will crunch through the problem and give a good answer quickly
- But when the state space is too large to store in a Q table, we cannot do exact Q learning
- For Blocks World at 5 by 5 or larger, the state space is unmanageably large

> **Potential midterm question**: What happens in reinforcement learning if there is an unmanageable number of states? **Answer**: We go from exact value function learning to learning an approximation of the value function. This is called **value function approximation**.

### Value Function Approximation

- **Q** (the action value function) maps (state, action) pairs to expected returns
- When we cannot compute Q exactly (via a Q table), we approximate it with a neural network
- **DQN (Deep Q Networks)** is one method that uses a deep neural network to approximate the Q function
- The neural network only sees a fraction of the total state space during training, yet we want it to generalize and behave intelligently on unseen states

> Roughly speaking, all of reinforcement learning boils down to value function approximation.

- David Silver has a dedicated (mathematical) lecture on value function approximation, though it is not assigned for this course

---

## Deep Q Networks (DQN)

### DQN and the Markov Property

All reinforcement learning mathematics is built on **Markov processes**, so the states must be Markov. This means the current state must contain all the information needed to predict future states and rewards. If the states are not Markov, the mathematical guarantees do not hold. You may accidentally get good results, but there is no theoretical backing.

### DQN vs. PPO: Choosing an Algorithm

The instructor has never solved the Blocks World problem before and is not aware of any published papers that have solved the symbolic Blocks World with this kind of dimensionality. This means there is no prior work to reference for the "correct" approach.

| Feature | DQN | PPO |
|---|---|---|
| Full name | Deep Q Networks | Proximal Policy Optimization |
| Type | Value function approximation | Policy gradient with value function approximation |
| Good for | Discrete, symbolic problems (e.g., Rubik's Cube, Blocks World) | Continuous action spaces, robotics |
| Experimental result on 3×4 Blocks World | **Solved** | Did not solve |

- There is no single "correct" algorithm for a given domain. Algorithm selection is trial and error.
- RL research involves running **experiments**: train with one reward structure, observe results, change reward or hyperparameters, train again, compare.
- DQN was chosen for the Blocks World project because it solved the 3 by 4 case, is known to work well on symbolic problems, and is based on Q learning (which students already know).

### Connection Between Q Learning and DQN

DQN is essentially Q learning done with neural networks instead of a table:

| Aspect | Q Learning | DQN |
|---|---|---|
| Value storage | Q table (explicit lookup) | Neural network (function approximation) |
| State input | Table row index | Network input vector |
| Action selection | Look up Q values in table row | Forward pass through network |
| Update rule | Bellman equation on table entries | Gradient descent on network weights |
| Scalability | Limited by table size (state × action) | Scales to large/continuous state spaces |

### DQN Architecture

DQN uses **two neural networks** and a **replay buffer**:

```
┌─────────────────────────────────────────────────┐
│                   DQN Agent                      │
│                                                  │
│  ┌──────────────┐      ┌──────────────────┐     │
│  │  Q Network   │      │  Target Network   │     │
│  │  (updated    │      │  (updated less    │     │
│  │   frequently)│      │   frequently,     │     │
│  │              │      │   provides stable │     │
│  │  state → Q   │      │   targets)        │     │
│  │  values for  │      │                   │     │
│  │  all actions │      │                   │     │
│  └──────────────┘      └──────────────────┘     │
│                                                  │
│  ┌──────────────────────────────────────┐       │
│  │         Replay Buffer                 │       │
│  │  Stores (s, a, r, s') transitions    │       │
│  └──────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
```
*(reconstructed diagram)*

1. **Q Network (Policy)** *(from slides)*: A neural network (often MlpPolicy or CnnPolicy) that takes the state as input and outputs Q values for each possible discrete action. This is the "latest and greatest" network that is updated frequently.
2. **Target Network** *(from slides)*: A slowly updated, identical copy of the Q network used to compute the target Q value, which helps stabilize training by **preventing the network from chasing its own tail**.
3. **Replay Buffer** *(from slides)*: Stores past experiences to **break the correlation between consecutive samples**, allowing the agent to learn from a diverse, random batch of past data.
4. **Epsilon Greedy Exploration** *(from slides)*: The agent balances exploration and exploitation by choosing a random action with probability epsilon, or the best predicted action.

### Policy Types

| Policy | Description | When to use |
|---|---|---|
| **MLP (Multilayer Perceptron)** | Standard feed forward deep learning | When input is a flat array of numbers |
| **CNN (Convolutional Neural Network)** | Uses sliding filters that compute the **dot product** with input regions to detect spatial features. Validated in 2012 at University of Toronto for image recognition. Students covered CNNs in a previous semester course (machine vision). | When input is image like |
| **Multi Input Policy** | Takes a dictionary of observations, unpacks and concatenates them, then feeds through an MLP | When the environment provides multiple observation components (e.g., current config + target config) |

For the Blocks World project, **Multi Input Policy** is used because the environment provides both a current configuration and a target configuration as a dictionary. Under the hood, DQN unpacks the dictionary, concatenates the arrays, and processes them through an MLP.

If moving to CNN policy later, the observation format would need to change to something image like (e.g., a spatial picture of the block configuration). A **one hot stacked** representation might work better with CNN, though this is just a hunch.

### DQN Training Process

*(from slides)*

1. **Interaction and Collection**: The agent interacts with the environment, taking actions and storing transitions in the ReplayBuffer.
2. **Warm up**: For a specified number of steps (`learning_starts`), the agent acts randomly to fill the buffer before learning begins.
3. **Sampling**: After the warm up, the algorithm samples a random mini batch of experiences from the replay buffer.
4. **Target Calculation**: The target network computes the target Q value:

$$y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$$

where $r$ is the reward, $\gamma$ is the discount factor, and $Q_{\text{target}}$ is the target network's estimate for the next state $s'$.

5. **Loss Calculation and Update**: The main Q network computes the current $Q(s, a)$ and updates its weights by minimizing the **Mean Squared Error** between $Q(s, a)$ and $y$.
6. **Target Network Update**: Every `target_update_interval` steps, the main network weights are copied to the target network. This less frequent update is what provides training stability.

### Epsilon Greedy Exploration

DQN uses **epsilon greedy** exploration, identical to Q learning:

- Pick a random number $r \in [0, 1]$
- If $r < \epsilon$, take a **random action** (exploration)
- Otherwise, take the action with the **highest Q value** (exploitation)
- $\epsilon$ is small and **decays** over time, so the agent explores less as training progresses

---

## Action Wrappers: MultiDiscrete to Discrete

### The Problem

The Blocks World environment uses **tuple actions** (block, destination), which is a `MultiDiscrete` action space. DQN only supports **`Discrete`** action spaces (actions numbered 0 through N).

### The Solution: Wrapping

Instead of rewriting the environment, we **wrap** it. A wrapper adds a conversion layer around the environment that translates between tuple actions and single integer actions.

The wrapper is based on `gym.ActionWrapper` and works as follows:

1. Read the `nvec` parameter from the `MultiDiscrete` space. For a 3 by 4 case, `nvec = [3, 4]` meaning 3 possible block values and 4 possible position values.
2. Compute the total number of discrete actions by multiplying: $3 \times 4 = 12$ discrete actions.
3. **Ravel** (tuple to integer): Convert a (block, destination) pair to a single integer index.
4. **Unravel** (integer to tuple): Convert a single integer back to a (block, destination) pair.

> The `nvec` parameter is documented in the official Gymnasium docs for `MultiDiscrete`. It gives "the number of counts in each categorical variable." Always check the official docs as the gold standard for definitive information.

### Ravel/Unravel Formula

For a grid or multi dimensional index, the conversion between coordinates and a flat index uses:

$$\text{flat\_index} = \text{row} \times \text{num\_cols} + \text{col}$$

*(reconstructed)* and the reverse:

$$\text{row} = \text{flat\_index} \mathbin{//} \text{num\_cols}, \quad \text{col} = \text{flat\_index} \bmod \text{num\_cols}$$

This is the same concept used in cliff walking to convert (row, col) grid coordinates to a single state number. In Blocks World, it converts (block, destination) to a single action number.

```python
# From slides: DiscreteActionWrapper
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Assume env.action_space is MultiDiscrete([2, 3])
        self.dims = env.action_space.nvec
        self.action_space = gym.spaces.Discrete(np.prod(self.dims))

    def action(self, action):
        # Convert single integer back to tuple for the inner env
        return np.unravel_index(action, self.dims)
```
*(from slides)*

### Key Insight About Discrete Actions

DQN and Q learning do not care about the **internal structure** of actions. They just want actions numbered 0 through N. The algorithms try all actions and learn their effects. The wrapper handles the translation between the internal tuple structure and the flat integer DQN expects.

**Important clarification**: The action wrapper only converts **actions** (tuples to integers). It does not multiply or combine the state space with the action space. If doing Q learning (not DQN) on this environment, a separate wrapper for **both actions and states** is needed, since Q learning also requires discrete states (for the Q table rows).

> **Potential midterm question**: You may be shown wrapper code and asked to identify what it does. You should be able to recognize it as a wrapper and explain that wrappers can, for example, take multi discrete actions and map them to discrete actions and back. You do not need to memorize or write wrapper code.

---

## Vectorized Environments and Parallelization

### SubprocVecEnv

The environment is created using `SubprocVecEnv` with multiple parallel copies:

```python
# From slides: Applying wrappers to an environment
# Define a function that applies all your wrappers
def make_custom_env():
    import gymnasium as gym
    # using 4 blocks and 4 positions right now
    env = gym.make("blocks_env/BlocksTargetPython-v0",
                    num_blocks=4, num_positions=4)
    # Manually pass kwargs to each wrapper here
    env = TimeLimit(env, max_episode_steps=200)
    env = DiscreteActionWrapper(env)
    return env

# Use the function as the env_id, and create 4 parallel copies
env = make_vec_env(make_custom_env, n_envs=4)
```
*(from slides)*

- The registered environment name is `"blocks_env/BlocksTargetPython-v0"` (available from the Blocks World distribution in the shared area)
- `num_blocks` and `num_positions` are **keyword arguments** to the environment. When omitted, the defaults are **10 by 10** (10 blocks, 10 positions).
- `TimeLimit` wrapper limits episodes to 200 steps
- `DiscreteActionWrapper` converts MultiDiscrete to Discrete
- `make_vec_env` creates vectorized parallel copies. Running 4 environments in parallel means every time step yields 4x the training data

**Vectorized Environments** *(from slides)*: a method for stacking multiple independent environments into a single environment. Instead of training an RL agent on 1 environment per step, it allows us to train it on $n$ environments per step. For simple environments (grid worlds, blocks worlds) running on a VM or loaner laptop, SB3 will use a `DummyVecEnv` for compatibility with VecEnv. For true parallel processing, use `SubprocVecEnv`.

### How Many Parallel Environments?

The number of parallel environments should roughly match the number of **CPU cores** on your machine:

- A Core i9 might have 12 cores. Setting `n_envs = 12` is a sane starting point.
- If you set `n_envs = 400` but only have 12 cores, each core must time slice across many processes. You lose performance to **context switching overhead**.
- One core running one process is faster than one core running two processes (due to context switching).
- The OS also uses some cores, so with 12 cores, maybe 10 environments is slightly better. But 12 is not unreasonable.
- There is no definitive answer. You have to experiment on your specific hardware.

> This matters in RL. Choosing the right parallelization can mean the difference between waiting a month and waiting six months for training to complete.

---

## GPU vs. CPU and CUDA

### CUDA

**CUDA (Compute Unified Device Architecture)**: NVIDIA's system (since ~2006) for general purpose computing on their GPUs.

Setting `device="cuda"` tells DQN to use the GPU. Alternatives *(from slides)*:
- `device="mps"`: Use Apple Metal Performance Shaders (for Mac)
- `device="auto"`: Let the framework choose
- `device="cpu"`: Force CPU usage

### Requirements for Using CUDA

1. An NVIDIA GPU that supports CUDA (e.g., RTX 4070)
2. The **CUDA Development Kit** installed on the system
3. Python packages (PyTorch, TensorFlow, etc.) must be **CUDA aware** (correct versions installed)
4. You can check CUDA availability in PyTorch:

```python
import torch
print(torch.cuda.is_available())  # True if CUDA is set up correctly
```
*(reconstructed)*

On Mac, the equivalent is **MPS (Metal Performance Shaders)**, which may work automatically with Apple Silicon without extra setup.

If no CUDA device is available, training falls back to CPU, which may or may not be significantly slower depending on CPU power (e.g., Apple Silicon CPUs are quite capable).

### GPU vs. CPU Tradeoff in DQN

DQN training involves two types of work:

| Stage | Compute type | Hardware |
|---|---|---|
| Environment interaction (stepping) | CPU bound | CPU |
| Neural network training (backprop) | GPU bound | GPU |

- If environment steps take much longer than training, a powerful CPU matters more than a powerful GPU
- If training dominates, GPU matters more
- The optimal device depends on the specific algorithm, environment complexity, and hardware. Experiment to find out.

---

## DQN Hyperparameters

### Key Hyperparameters

| Hyperparameter | Description | Example value | Notes |
|---|---|---|---|
| `learning_starts` | Number of random steps before learning begins | 100 | No "correct" value. Based on what others have used in similar work. |
| `batch_size` | Number of samples per training batch | 512 | Larger batches save up more work for the GPU, reducing context switching overhead. 256 or 1024 would also work. |
| `buffer_size` | Size of the replay buffer | (default) | How many transitions to store |
| `learning_rate` | Step size for gradient descent | (default) | How fast the network updates |
| `device` | Hardware to use | `"cuda"` | GPU (CUDA), CPU, or auto |

### Batch Size Explained

- **Unit**: Each sample is one (state, action, reward, next_state) transition collected from the environment
- A batch size of 512 means the neural network trains on 512 transitions at a time
- Larger batch sizes reduce the overhead of switching between data gathering and training, but take more GPU memory
- The optimal batch size depends on the hardware. Experiment with 256, 512, and 1024 to compare.

### Stable Baselines 3 Documentation

All DQN hyperparameters are documented in the **Stable Baselines 3 official docs**: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

**SB3 Base RL Class** *(from slides)*: All RL algorithms in SB3 extend a common Base Class (https://stable-baselines3.readthedocs.io/en/master/modules/base.html), providing a common interface. The basic pattern is:

```python
model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
```
*(from slides)*

For guidance on which algorithm to use: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#which-algorithm-should-i-use

The **RL Zoo** provides mechanisms for **automated hyperparameter tuning**, which is the next step after getting a working baseline.

---

## TensorBoard: Monitoring Training

### What is TensorBoard?

**TensorBoard** is a comparative graphing facility originally built for TensorFlow. It is designed for:

- Watching training progress in real time
- Comparing multiple training experiments (different reward structures, different hyperparameters)
- Runs in the browser (similar to Jupyter Notebooks in that way)

### Setting Up TensorBoard

1. Set up directories for logs and models, then create the DQN model with TensorBoard logging:

```python
# From slides: Logs and trained model storage
models_dir = "models/dqn"
logs_dir = "logs/dqn"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# From slides: DQN hyperparameters
model = DQN("MultiInputPolicy", env, learning_starts=100, device="cuda",
            batch_size=512, verbose=1, tensorboard_log=logs_dir)
```
*(from slides)*

Explanation of each parameter *(from slides)*:
- `MultiInputPolicy`: observations are a dictionary with current and target configurations
- `env`: the wrapped environment
- `learning_starts`: number of random actions before learning starts
- `device="cuda"`: use CUDA GPU (would be `"mps"` on a Mac, or `"auto"`, or `"cpu"`)
- `batch_size=512`: batch size for update
- `verbose=1`: print training info to terminal
- `tensorboard_log=logs_dir`: log training progress to the specified directory for viewing with TensorBoard

DQN docs: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

2. After launching training, open a separate terminal and run:

```bash
tensorboard --logdir=./logs/
```

3. TensorBoard provides a URL (e.g., `http://localhost:6006`) where you can view the graphs.

### TensorBoard with PyTorch

Stable Baselines 3 is based on **PyTorch**, not TensorFlow. If you only have PyTorch installed (no TensorFlow), TensorBoard will still work but with a reduced feature set. Install TensorBoard with pip:

```bash
pip install tensorboard
```

### What TensorBoard Shows

Key graphs observed during Blocks World training:

1. **Mean Reward**: Shows how the average reward per episode changes over training steps
   - Example: One training run (green line) started at mean reward of ~$-15{,}000$ and improved to ~$-5{,}000$ over about 70 million steps
   - A previous run (purple line) showed improvement in stages with plateaus and jumps
   - A Q learning run on 3 by 4 quickly reached rewards of ~$+20$ to $+30$

2. **Episode Length**: Shows average episode length over time
   - Episodes start capped at 200 (the time limit)
   - As the agent improves, some episodes terminate early (reaching the goal)
   - The agent occasionally "gets lucky" with shorter episodes, shown as dips in the graph

3. **Exploration Rate**: Shows how epsilon decays over time, just like in Q learning

### Training Time Estimates

- 100 million steps on RTX 4070 hardware: approximately 3 hours
- 1 billion steps: a few days

> **Potential midterm question**: What is TensorBoard used for?

---

## Progress Callbacks

### Custom Callback for Screen Output

**Callbacks** *(from slides)*: A callback is a set of functions that will be called at given stages of the training procedure. You can use callbacks to access internal state of the RL model during training. It allows one to do monitoring, auto saving, model manipulation, progress bars, and more. SB3 includes built in `CheckpointCallback` and `EvalCallback`.

A custom `ProgressCallback` reports training status to the terminal. It is a subclass of `BaseCallback` from Stable Baselines 3 (object oriented Python). The `_on_step` method runs every step, and using modulo (`%`) with the frequency (e.g., 10,000), it prints a report at regular intervals.

SB3 also provides built in callbacks. Example using `EvalCallback` *(from slides)*:

```python
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

# Separate evaluation env
eval_env = gym.make("Pendulum-v1")
# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=500,
                             deterministic=True, render=False)

model = SAC("MlpPolicy", "Pendulum-v1")
model.learn(5000, callback=eval_callback)
```
*(from slides)*

### Connecting the Callback to Training

```python
# From slides: Training the model
# Train for 1,000,000 timesteps with progress reports every 10,000 steps
callback = ProgressCallback(check_freq=10000)
model.learn(total_timesteps=1000000, log_interval=1, callback=callback)
model.save(f"{models_dir}/dqn_blocks_world")
```
*(from slides)*

### Logging Details *(from slides)*

With `tensorboard_log=logs_dir`, SB3 initializes a global logger that handles multiple output formats simultaneously: **terminal (stdout)** and **TensorBoard binary files**.

- **`ProgressCallback`** has access to this same logger via `self.logger`. Any custom metrics recorded in the callback using `self.logger.record("key", value)` will automatically appear in TensorBoard graphs.
- **`log_interval=1`**: For DQN, this tells SB3 to write a data point to TensorBoard every **1 episode**. This includes standard metrics like `rollout/ep_rew_mean` and `train/loss`.
- **`check_freq=10000`**: The callback only triggers its logic every **10,000 timesteps**.
- **The result**: high resolution data in TensorBoard (every episode), while terminal/callback reports will only update in 10,000 step jumps.

After training, `model.save()` stores the trained weights to disk.

### Loading and Running a Trained Model

```python
# From slides: Running trained models
model = DQN.load(f"{models_dir}/dqn_blocks_world", env)
obs = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, info = env.step(action)
```
*(from slides)*

- The model is loaded with the environment passed to `DQN.load()`
- `model.predict(obs, deterministic=True)` returns the action and internal states
- Note: with a VecEnv, `env.step()` returns 4 values (`obs, reward, terminated, info`), not 5. The VecEnv handles resets automatically.
- Set `render_mode="human"` to visualize the agent's behavior

### The No Agent Script (Random Baseline)

The "no agent" script takes **random actions** with no learning. It is useful for verifying the environment works correctly:

```python
# No agent: random actions only (reconstructed)
obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random action from the action space
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs = env.reset()
```
*(reconstructed)*

- `env.action_space.sample()` is the standard Gymnasium method for selecting a random action
- No learning occurs. This just lets you observe the environment behaving with random inputs.

---

## Episode Length Limits

### Why Limit Episode Length

Episodes are capped at **200 steps** using the `TimeLimit` wrapper:

- Without a limit, early episodes could run for thousands or millions of steps (the agent keeps making useless moves like moving block A back and forth)
- Long useless episodes waste compute and provide low quality training data
- 200 steps is enough for 5 blocks to reach the target configuration

### Benefits of Truncation

- Forces **variety**: Each truncated episode starts with a new random configuration and target, so the agent sees diverse situations
- Prevents the agent from spending all its training time on a single unproductive episode
- As the agent improves and starts reaching the goal within 200 steps, the episode length will naturally decrease below the limit. At that point, truncation no longer matters because the episodes are terminating on their own.

---

## State Representation: Index Encoding vs. One Hot

### Current Encoding (Index Based)

Each block's position is stored as a single integer:

- Value 0 through $n-1$: the block is sitting on another block (index = block ID)
- Value $n$ through $2n-1$: the block is sitting on a position (index = position ID offset by $n$)

**Example** *(reconstructed)*: With 10 blocks (A through J, indices 0 through 9) and 10 positions (indices 10 through 19):

| Block | State value | Meaning |
|---|---|---|
| A | 3 | A is on block D |
| B | 11 | B is on position 2 (11 = 10 + 1, offset by num_blocks) |
| C | 10 | C is on position 1 |

### Alternative: One Hot Encoding

Each block would be represented as an array of $2n$ values (one for each possible place), with a 1 in the position it occupies and 0s everywhere else:

**Example** *(reconstructed)*: With 3 blocks and 4 positions (7 total places):
- Block A on position 1: `[0, 0, 0, 0, 1, 0, 0]`
- Block B on block A: `[1, 0, 0, 0, 0, 0, 0]`

The full observation would be a 2D array (stack of one hot vectors), one row per block.

### Comparison

| Aspect | Index encoding | One hot encoding |
|---|---|---|
| Observation size | $n$ integers | $n \times 2n$ binary values |
| Human readability | Low | Low |
| Tested with PPO/DQN | Current approach | Tested, neither PPO nor DQN performed well with it |
| Potential fit | MLP policy | CNN policy (spatial, image like input) |

There is no guarantee which representation is better. It depends on the algorithm and must be determined experimentally. The hunch is that one hot stacked might work better with a CNN policy, but this has not been confirmed.

> This is reinforcement learning. We do not know these things upfront. Hunches are normal. The only way to know is to experiment.

---

## Experimental Results on Blocks World

### Q Learning on 3 by 4

- Using a wrapper for both discrete actions and discrete states, Q learning **solved** the 3 by 4 Blocks World quickly
- This validates that the environment implementation is correct
- Q learning is limited by the **size of the state space**. The action space is manageable, but the state space becomes unwieldy at larger sizes.

### DQN on 4 by 4

- After 10 million training steps, DQN on a 4 by 4 Blocks World was "not great"
- Q learning works better when feasible, but DQN is needed when the state space is too large
- This does not mean DQN fails. It means more training, better hyperparameters, or better rewards may be needed.

> The goal of working with Blocks World is not necessarily to solve it. It is to learn about TensorBoard, Q learning limitations, DQN, reward shaping, and experimental methodology. The plan is to move to real robotics later, which will actually be easier than Blocks World.

---

## The RL Agent Environment Diagram

> **Midterm requirement**: Be able to draw the RL agent environment interaction diagram.

```
         ┌───────────┐
         │   Agent    │
         └─────┬─────┘
           A_t │ ▲ O_{t+1}, R_t
               │ │
               ▼ │
         ┌─────┴─────┐
         │Environment │
         └───────────┘
               │
               ▼
          S_{t+1} (full state,
          may not be fully
          observable)
```
*(reconstructed)*

- $S_t$: The full environment **state** at time $t$
- $O_t$: The **observation** the agent receives (the agent observable part of the state). The agent cannot always see the entire environment. Most of the time, it only gets the observation.
- $A_t$: The **action** the agent takes
- $R_t$: The **reward** the agent receives (note: this is $R$, not $A$)
- $S_{t+1}$: The next state after the action is taken

---

## Improving DQN Performance

Three main approaches to improving DQN results:

### 1. Action Masking

> **Course note**: The **practical tips video** (https://www.youtube.com/watch?v=Ikngt0_DXJg) discusses action masking and other RL advice. This is an assigned/recommended resource from the SB3 slides.

**Action masking** is where the environment filters out impossible or nonsensical actions for the agent in the current configuration. For example, if block A is buried in a stack, actions that try to move A are not possible. Filtering these out reduces the action space the agent must explore. This is a more advanced technique.

### 2. Reward Shaping

### 3. Hyperparameter Tuning (via RL Zoo)

---

## Reward Shaping

### What is Reward Shaping?

**Reward shaping** is the process of designing a reward function that guides the agent toward the desired behavior more effectively than a simple sparse reward (e.g., +1 for solving, 0 otherwise).

> The lecturer cautions against using AI tools (like ChatGPT) for reward design. AI may give detailed but not necessarily good answers. The best approach is to think carefully with pen and paper.

The reward code shown in class included commented out earlier ideas that were abandoned, illustrating that reward shaping is an iterative process of trying ideas, testing, and revising.

### Reward Shaping for Blocks World

The reward shaping logic is implemented by modifying the environment's **`move` function**, so the reward is computed inside the environment step logic (not in the agent or a separate module). The move action is analyzed: first check if the move is possible, then assign rewards based on the conditions below.

The lecturer's reward structure (values are relative magnitudes, not necessarily optimal):

| Condition | Reward | Reasoning |
|---|---|---|
| Any step (default) | $-1$ | Penalize each step to encourage shorter episodes |
| Block moved to its **final destination** | $+50$ | Strongly reward progress toward the goal |
| Block moved **away from** its final destination | $-50$ | Strongly penalize undoing progress |
| Block moved somewhere else when it **could have** gone to its final position | $-50$ | Penalize missed opportunities |
| Block removed from on top of a block that **needs to be moved** (non final stack) | $+3$ | Reward freeing up blocks that are incorrectly placed |

**Final destination** means:
- The block is sitting on a position that matches the target, OR
- The block is sitting on the correct tower, and that entire tower underneath matches the target configuration

### The Reward Hacking Problem

**Reward hacking** occurs when the agent finds a way to collect rewards without making real progress. The most common example: moving a block to its final position ($+50$), then moving it away ($-50$), then moving it back ($+50$), and repeating forever.

Strategies to prevent reward hacking:

1. **Asymmetric rewards**: Penalizing bad moves differently from rewarding good moves so that back and forth cycles are not profitable
2. **Directional bias**: When a block must be freed from a non final stack, reward moving it to a **taller** stack and penalize moving it to a **shorter** stack. This breaks the symmetry so back and forth is not possible (each direction gets a different reward). Moving to the left vs. right was considered but rejected because it does not consistently break symmetry.
3. **Including previous state in the observation**: Add the previous configuration to the current state so the agent can detect it is repeating. However, this only catches 2 cycles. Cycles of length 3, 4, or nested cycles (a 3 cycle within a 2 cycle within a 4 cycle) would require even more history, increasing state dimensionality.
4. **Distance based rewards** (student suggestion): Reward based on reducing the distance between a misplaced block and its target destination. As a block gets closer, the agent gets rewarded. Once the distance is zero, the agent can move on to other blocks. This "chasing the distance" approach might be effective, but must be verified experimentally.
5. **Flattening strategy**: With $n$ blocks and $n$ positions (equal numbers), you can always unstack everything to the ground and rebuild in the target order. Rewarding flattening could work for square configurations, but fails for narrow ones (e.g., 500 blocks with only 3 columns, where there is not enough ground space to flatten).

> With reward shaping, we always have to be on the lookout for reward hacking. It is not a trivial problem.

> Blocks World has never been fully solved (with reinforcement learning at arbitrary scale). If anyone comes up with a good reward structure and solves it, that could be publishable work.

> **Course note**: There is no assignment requiring you to do reward shaping for Blocks World. However, if you come up with good ideas, you are encouraged to implement them and share results with the instructor.

---

## Next Steps

1. **Hyperparameter Tuning**: The **RL Zoo** provides mechanisms for automated hyperparameter exploration. This is the next focus after establishing a working reward structure.
2. **Increased Training Budget**: A 1 billion step training run is in progress (expected to take a few days), but increasing the budget alone is unlikely to solve the problem.
3. **Future direction**: The course will transition from Blocks World to **real robotics**, which will be easier than Blocks World and may not even require reinforcement learning, but RL will be applied for additional learning.

---

## Potential Midterm Topics

> **Course note**: The following types of questions may appear on the midterm.

- **Conceptual**: What happens in reinforcement learning when there is an unmanageable number of states? (Answer: value function approximation)
- **Conceptual**: What is TensorBoard used for? (Answer: comparative graphing facility for monitoring and comparing training experiments)
- **Code identification**: Given a piece of code, identify its purpose. For example, recognizing wrapper code and explaining that wrappers can convert multi discrete actions to discrete actions and back. You do not need to write code from memory, but you should be able to read code and understand what it does.
- **Diagram**: Draw the RL agent environment interaction diagram ($S_t$, $O_t$, $A_t$, $R_t$)
- **Reward shaping**: Propose a reward structure for a given environment (does not need to be as detailed as the lecturer's analysis, just demonstrate understanding of the tradeoffs)
