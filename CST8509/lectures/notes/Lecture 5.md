# Lecture 5: From Q-Learning to Value Function Approximation, Stable Baselines 3, and CartPole

## Course Roadmap

The course follows a progression from RL fundamentals through to applying reinforcement learning on a physical robot:

1. **RL basics** (covered by quizzes)
2. **Q-learning** (Labs 1 and 2, Cliff Walking, Blocks World 3x4)
3. **Value function approximation** with PPO, DQN (Stable Baselines 3 case study)
4. **Gazebo simulator** with the Create 3 robot doing finger following

> **Course note**: Antonin Raffin (the maintainer of Stable Baselines 3) has a practical tips video for reinforcement learning ([video](https://www.youtube.com/watch?v=Ikngt0_DXJg), [tutorial repo](https://github.com/araffin/rl-handson-rlvs21), [Colab exercise](https://colab.research.google.com/github/araffin/rl-handson-rlvs21/blob/main/rlvs_hands_on_sb3.ipynb)). One of the first questions he poses is: "Do you need reinforcement learning for your problem?" For the finger following problem, the answer is no, but we are doing it anyway for learning purposes.

> **Course note**: The scope of this course is an introduction to reinforcement learning. After completing it, you can go on to arbitrarily complex problems.

---

## Q-Learning Review and Limitations

### What Q Represents

**Q** is the value function of states and actions. $Q(s, a)$ equals the value of being in state $s$ and taking action $a$.

### Two Requirements for Q-Learning

To implement a Q-learning agent, two things must happen in order:

1. **Create an instance of the environment**
2. **Create a Q table** to implement the Q function, populated with random values initially (since we don't know the real values yet)

### Why These Requirements Are Limiting

To create a Q table in Python, we need to know:

- **How many states** there are (the state space)
- **How many actions** there are (the action space)

If we cannot answer these questions, or if the numbers are too large (e.g., $10^{60}$ states), Q-learning becomes infeasible. There is no laptop that can hold a Q table with $10^{60}$ entries.

### The Continuous Actions Problem

Continuous actions make Q-learning impossible because the action space becomes infinite. For example, pressing an accelerator pedal on a car: you can press it 0.0001 mm, or 0.00001 mm, or 0.1 mm, or 0.01 mm, or 0.001 mm. There are uncountably infinite possibilities if we use real numbers, as opposed to rationals.

> Many scientists just assume that the real numbers are used in nature, like the velocity of something can be irrational. That's a big assumption, given that humankind cannot measure anything and get an irrational result. All of the results we get from any measurement are rational. We'd have no way of knowing the true nature of that.

### Summary of Q-Learning Limitations

| Requirement | Why It Matters |
|---|---|
| Reasonable number of states | Must fit in memory as a Q table |
| Known number of states | Table dimensions must be declared in advance |
| Reasonable number of actions | Must fit as columns in the Q table |
| Discrete actions | Continuous actions produce an infinite action space |

---

## Blocks World: From 3x4 to 10x10

### Blocks World with 3 Blocks, 4 Positions (Q-Learning)

Blocks World is not a standard Gymnasium environment (unlike Cliff Walking, which is built in). That is why it was chosen for this course.

Two variants were implemented:

- **Without target**: just manipulate blocks
- **With target**: reach a specific goal configuration

With 3 blocks and 4 positions, the world is small enough for Q-learning: roughly 120 actions and 90 states (approximate). Q-learning works well here.

### Scaling to 10 Blocks, 10 Positions

With 10 blocks and 10 positions, the number of states and actions explodes. This forces us away from Q-learning and toward **value function approximation** algorithms.

> **Course note**: Q-learning is the hero for small Blocks World. The instructor has not gotten PPO or DQN to solve Blocks World yet, even after 1,000,000+ training steps. Early experiments show DQN working better than PPO on this type of problem (DQN is a popular choice for combinatorial domains such as the Rubik's Cube). Students in Labs 1 and 2 are not expected to get good results with PPO/DQN on Blocks World. The goal is just to demonstrate that you can run these algorithms, not that they will solve the problem.

---

## Value Function Approximation: PPO and DQN

### Why We Need It

When we cannot create a Q table (too many states or continuous actions), we replace the table with a **neural network**. Instead of updating individual table cells, we use information from action cycles to train the neural network.

The key benefit: we don't need to know the exact number of states up front. We just create a suitably sized network.

### PPO (Proximal Policy Optimization)

A value function approximation algorithm that works with discrete, MultiDiscrete, MultiBinary, and Box action/observation spaces.

### DQN (Deep Q-Network)

The "Q" in DQN is the same Q from Q-learning. DQN is essentially a neural network implementation of the Q table. When the Q table is too large to create, a neural network can approximate the Q function instead.

Key components that make this work *(from slides)*:

- **Q-Network (Policy)**: a neural network (often `MlpPolicy` or `CnnPolicy`) that takes the state as input and outputs Q-values for each possible discrete action
- **Target Network**: a slowly updated, identical copy of the Q-network used to compute the target Q-value, which stabilizes training by preventing the network from chasing its own tail
- **Replay Buffer**: stores past experiences to break the correlation between consecutive samples, allowing the agent to learn from a diverse, random batch of past data
- **Epsilon-Greedy Exploration**: the agent balances exploration and exploitation by choosing a random action with probability $\epsilon$ or the best-predicted action otherwise

The target Q-value is calculated as:

$$y = r + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a')$$

The main Q-network then updates its weights by minimizing the MSE between $Q(s, a)$ and $y$. Every `target_update_interval` steps, the main network weights are copied to the target network. *(from slides)*

> **Constraint**: DQN requires a **Discrete** action space. If the environment uses MultiDiscrete actions (like the Python-based Blocks World), a `DiscreteActionWrapper` must be applied to flatten the action space before DQN can train on it. *(from slides)*

### Stable Baselines 3 Algorithms

| Algorithm | Full Name |
|---|---|
| A2C | Advantage Actor-Critic |
| DDPG | Deep Deterministic Policy Gradient |
| DQN | Deep Q-Network |
| PPO | Proximal Policy Optimization |
| HER | Hindsight Experience Replay |
| SAC | Soft Actor-Critic |
| TD3 | Twin Delayed DDPG |

> **Course note**: PPO, DQN, and HER are the main algorithms that will be explored. SAC and TD3 might also come up. See the official guide [Which Algorithm Should I Use?](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#whichalgorithm-should-i-use) for choosing between them.

---

## Stable Baselines 3 Overview

### Key Advance

The main advance from Stable Baselines to Stable Baselines 3 was making the interface between algorithms and environments **uniform**, so you can apply basically any algorithm to any environment. That does not mean you should, but the plumbing has been worked out.

### Core Features

1. **Vectorized environments**: train on multiple parallel environments every time step instead of just one
2. **Callbacks**: hook into the training loop for evaluation, logging, and model saving
3. **Environment wrappers**: modify environment behavior without changing the environment itself

### Base RL Class

All RL algorithms in SB3 extend a common **[Base Class](https://stable-baselines3.readthedocs.io/en/master/modules/base.html)** with a uniform interface. This means you create a model, call `learn()`, and save/load the same way regardless of algorithm.

```python
# Common interface: works the same for DQN, PPO, A2C, SAC, etc.
model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
```

*(from slides)*

Two common policies:
- **MlpPolicy**: multilayer perceptron, for flat observation spaces (arrays, discrete values)
- **MultiInputPolicy**: for dictionary observation spaces (e.g., separate current and target states)

### Basic Usage Pattern

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment (4 parallel copies)
env = make_vec_env("CartPole-v1", n_envs=4)

# Create model with MlpPolicy (multilayer perceptron neural network)
model = PPO("MlpPolicy", env, verbose=1)

# Train for 25,000 steps
model.learn(total_timesteps=25_000)

# Save and reload
model.save("ppo_cartpole")
del model
model = PPO.load("ppo_cartpole")
```

*(reconstructed from lecture description)*

### Vectorized Environments

**Vectorized Environments** are a method for stacking multiple independent environments into a single environment. Instead of training an RL agent on 1 environment per step, it allows training on *n* environments per step.

Instead of `gym.make()` which gives a single environment, use `make_vec_env` from `stable_baselines3.common.env_util`. For simple environments (grid worlds, Blocks World, running on a VM or laptop), SB3 will turn these into a **DummyVecEnv** for compatibility with the VecEnv interface. The `n_envs` parameter is configurable: 4 for CartPole, or as high as 16 (e.g., `make_vec_env("Pendulum-v1", n_envs=16)`). A [detailed multiprocessing example](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb) is available. *(from slides)*

```python
# Single environment (standard Gymnasium)
env = gym.make("CartPole-v1")

# Vectorized: 4 environments in parallel (Stable Baselines 3)
env = make_vec_env("CartPole-v1", n_envs=4)
```

### Callbacks

A **callback** is a set of functions that will be called at given stages of the training procedure. You can use callbacks to access internal state of the RL model during training. It allows one to do monitoring, auto saving, model manipulation, progress bars, and more.

Built-in callbacks in SB3:

- **CheckpointCallback**: periodically saves the model during training
- **EvalCallback**: evaluates the model during training on a separate evaluation environment and saves the best performing version

Custom callbacks can report average rewards and other metrics at any point during training.

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

Hyperparameter tuning is also available but was not covered in this lecture.

---

## Environment Setup

### Supported Platform

- **Ubuntu 22.04** with **Python 3.10.12** (the default Python on 22.04)
- Mac and Windows can work, but Linux 22.04 is the supported platform
- Version matching is critical

### Virtual Environment Setup

```bash
# Create virtual environment using Python's built-in venv module
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Install Stable Baselines 3 with extras (includes pygame, TensorBoard, etc.)
pip install stable-baselines3[extra]
```

*(reconstructed)*

The `[extra]` flag installs useful dependencies:

- **pygame**: for rendering environments graphically
- **TensorBoard**: for visualizing training metrics
- **PyTorch**: the deep learning backend
- **Gymnasium**: the environment framework
- **OpenCV**: for image processing

### Managing Dependencies

To capture a virtual environment's packages for reproducibility:

```bash
# Export current environment
pip freeze > requirements.txt

# Reproduce on another machine
pip install -r requirements.txt
```

> **Course note**: Files are available on Brightspace: `cartpole.zip` and `python_blocks.zip`.

---

## Gymnasium Spaces

Spaces define the structure of observations and actions. Both Stable Baselines 3 and Gymnasium document these.

### Discrete

A single integer value from 0 to $n-1$.

```python
# Example: 4 actions (up, down, left, right)
action_space = spaces.Discrete(4)  # Values: {0, 1, 2, 3}

# With optional seed for reproducibility (from slides)
action_space = spaces.Discrete(2, seed=42)  # Values: {0, 1}

action_space.sample()  # Returns a random valid action
```

*(reconstructed, seed example from slides)*

Grid worlds (like Cliff Walking) use Discrete action spaces. This is what Labs 1 and 2 use.

### Box

A set of values between specified limits in some dimensional structure. Used for continuous or bounded numeric observations.

```python
# General Box example
obs_space = Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)

# Box for 100x130 pixel images with RGB pixels
obs_space2 = Box(low=0, high=255, shape=(100, 130, 3), dtype=np.uint8)

obs_space.sample()  # returns a random value within the bounds
```

*(from slides)*

- Each pixel has 3 color channels (R, G, B), each ranging from 0 to 255
- (0, 0, 0) is black, (255, 255, 255) is white
- Sampling a Box produces a random value within those bounds (e.g., a random image)

### MultiDiscrete

An array of discrete values, where each element can range independently.

```python
# Example: two independent discrete values
action_space = spaces.MultiDiscrete([10, 5])  # First: 0-9, Second: 0-4
```

*(reconstructed)*

Values can be things like 10, 11, 12, not just 0 or 1.

### MultiBinary

Similar to MultiDiscrete, but each value is restricted to 0 or 1.

```python
# Example: 4 binary flags
observation_space = spaces.MultiBinary(4)  # Each value: {0, 1}
```

*(reconstructed)*

### Dictionary

Combines multiple spaces into a single observation. Used in Lab 2, where the environment gives an observation as a dictionary with a current component and a target component (where you are now versus where you're trying to go).

```python
# Example: Blocks World with separate current and target states
observation_space = spaces.Dict({
    "current": spaces.MultiDiscrete([20] * 10),
    "target": spaces.MultiDiscrete([20] * 10),
})
```

*(reconstructed)*

In the Blocks World assignment, instead of using a Dictionary space, the assignment instructions tell you to concatenate current and target into a single representation and treat it as a Discrete space, because it is easy to query how many states there are with a Discrete space.

---

## CartPole Demo: PPO vs. DQN

### The CartPole Environment

CartPole is a standard Gymnasium environment (`CartPole-v1`). A cart sits on a track trying to balance a pole, like balancing a baseball bat. The cart can move left or right. The longer the pole stays balanced, the better.

### PPO on CartPole

Key implementation details from the demo:

- Uses `make_vec_env` with 4 parallel environments for faster training
- Uses `MlpPolicy` (multilayer perceptron, a deep learning neural network)
- Trains for 25,000 total steps
- The algorithm reports episode length and reward average during training. Longer episodes mean better balance.

**Training vs. Evaluation**: during training, 4 parallel environments run simultaneously. For evaluation, the instructor switches to a single environment with `render_mode='human'` so you can visually watch the agent perform.

```python
# Evaluation: switch to single environment for visual inspection
eval_env = gym.make("CartPole-v1", render_mode="human")
model = PPO.load("ppo_cartpole", env=eval_env)

obs, _ = eval_env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    if terminated or truncated:
        obs, _ = eval_env.reset()
```

*(reconstructed from lecture description)*

### DQN on CartPole

- Uses a **single environment** (not vectorized)
- Produces more verbose training reports (`log_interval=1` writes every episode to TensorBoard, while `ProgressCallback(check_freq=10000)` prints to the terminal in larger jumps)
- In this experiment, DQN performed **much worse** than PPO. Episodes ended almost immediately (the pole fell over so fast you couldn't even see it).

Typical DQN constructor call *(from slides)*:

```python
model = DQN("MultiInputPolicy", env, learning_starts=100,
            device="cuda", batch_size=512, verbose=1,
            tensorboard_log=logs_dir)
```

- `learning_starts=100`: number of random actions before learning begins (warm-up)
- `device="cuda"` for NVIDIA GPU, `"mps"` on Mac, `"auto"`, or `"cpu"`
- `batch_size=512`: mini-batch size for replay buffer sampling
- `tensorboard_log=logs_dir`: enables TensorBoard graphing of training progress

### Comparison

| Aspect | PPO | DQN |
|---|---|---|
| Environments used | 4 (vectorized) | 1 (single) |
| Training speed | ~4x faster due to parallelism | Baseline |
| CartPole result | Kept pole balanced for long episodes | Pole fell almost immediately |
| Winner | **Yes** | No |

> PPO is definitely the winner in this CartPole experiment.

---

## 10x10 Blocks World Representation

The instructor manually designed the representation for the 10x10 Blocks World before handing it to Claude Code for implementation. The human must think through the representation. The AI helps speed up the coding, but the design decisions come from the human.

### Observation Encoding

An observation is an **array of length `num_blocks`** (10), where the value at each index tells you where that block currently is.

- Indices 0 through 9 represent **blocks**
- Values 10 through 19 represent **positions**

```
Index:    [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]
Meaning:  B0   B1   B2   B3   B4   B5   B6   B7   B8   B9
Value:    Where this block is sitting (another block 0-9, or a position 10-19)
```

*(reconstructed)*

**Example**: if index 0 has value 9, that means block 0 is on block 9. If index 9 has value 13, that means block 9 is on position 13. So block 0 is stacked on block 9, which sits on position 13.

### Design Goal

The representation is designed for **fast training**. A move is just setting one array value to a new destination. No complex data structures to manipulate.

### Preconditions for Moves

Drawing on prior coursework about preconditions:

**`position_exists(x)`**: checks whether `x` is a valid block or position.

```python
def block_exists(x):
    return 0 <= x < num_blocks

def position_exists(x):
    return num_blocks <= x < num_blocks + num_positions
```

*(reconstructed)*

**`is_possible(observation, x, y)`**: checks whether block `x` can be moved to destination `y`.

Conditions:
1. Block `x` must exist
2. Destination `y` must exist (either a block or a position)
3. `x` must **not appear** in the observation array (nothing is sitting on top of `x`, so it is clear to pick up)
4. `y` must **not appear** in the observation array (nothing is sitting on top of `y`, so there is room to place something there)

> The key insight: if a value appears in the observation array, it means something is sitting on top of it. If block 9 is at position 13 (value 13 is in the array at index 9), then position 13 is occupied. You cannot move another block to position 13.

**Performing a move**: if the move is possible, update the observation array by setting `observation[x] = y`. Otherwise, return false.

### Observation and Action Spaces

| Space | Type | Structure |
|---|---|---|
| Observation (current state) | MultiDiscrete | Array of 10 values, each in range [0, `num_blocks` + `num_positions`) |
| Observation (target state) | MultiDiscrete | Array of 10 values, same range |
| Action | MultiDiscrete | Two values: (block to move, destination) |

The observation is MultiDiscrete because it contains more than one value. If it were a single value, Discrete would suffice. With MultiDiscrete, each element ranges from 0 to `num_blocks + num_positions - 1`.

Because DQN requires a Discrete action space, the environment must be wrapped before training:

```python
from gymnasium.wrappers import TimeLimit

def make_custom_env():
    env = gym.make("blocks_env/BlocksTargetPython-v0",
                   num_blocks=4, num_positions=4)
    env = TimeLimit(env, max_episode_steps=200)
    env = DiscreteActionWrapper(env)  # Flatten MultiDiscrete -> Discrete
    return env

env = make_vec_env(make_custom_env, n_envs=4)
```

*(from slides)*

### Algorithmic Simplicity vs. RL Difficulty

With 10 positions and 10 blocks, solving this algorithmically is easy: flatten all stacks, then rebuild in the target configuration. It gets more challenging with fewer positions (like Towers of Hanoi with only 3 towers). But solving it with reinforcement learning is a research-level challenge.

---

## The Course End Goal: Create 3 in Gazebo

The ultimate goal of the course is to train the **iRobot Create 3** robot to do **finger following** using value function approximation.

- The Create 3 will be placed in the **Gazebo simulator** with a virtual laptop camera
- In the simulation, the physical laptop is not needed. Gazebo can model robot parts with realistic weights so forces and torques are physically accurate.
- The robot has 3 discrete actions: **turn left, turn right, go straight ahead**

> **Course note**: This is as far as the course will get. The finger following problem does not actually need reinforcement learning, but it is being solved with RL for educational purposes.

---

## RL as Research, Not Just Programming

Solving a reinforcement learning problem is not like typical software development:

- In programming, you get a specification and implement code that meets it. The process is well understood, even if it takes months.
- In RL, you have to **discover** the right representation for the environment, the right algorithm, the right observation encoding, and so on. It is more like research.
- The instructor is running experiments on the 10x10 Blocks World and has not even reached hyperparameter tuning yet.
- RL applications in industry include physical robots and LLMs (reinforcement learning from human feedback is used after pre-training to make models give good answers).

> No computer will ever be fast enough. 20 years ago, this laptop was bigger than any supercomputer available, and today it is quite slow.

---

## AI-Assisted Development with Claude Code

### How the 10x10 Blocks World Was Built

The instructor used a hybrid approach:

1. **Human designs the representation**: the instructor manually wrote the observation encoding, move logic, and space definitions without AI help, because he knew exactly what he wanted
2. **Claude Code implements the rest**: after providing the representation as a prompt, Claude Code registered the environment, created training scripts, and committed everything to Git
3. **Iteration required**: Claude Code did not get it right the first time. The instructor started the process and Claude Code sped up the finishing

### AI Coding Tools

Several AI coding tools were mentioned: **Claude Code** (terminal-based), **Cursor**, and **GitHub Copilot**. They all work in similar ways.

### The Teaching Dilemma

> Using AI coding tools is a little bit like teaching somebody how to turn on Tesla Full Self-Driving when they can't drive yet. It could get them into trouble. Whereas somebody who already knows how to drive can take the wheel. You have to know when to take the wheel when it goes off course.

The key points:

- Expert programmers are very impressed by Claude Code, but you still have to **check everything** it produces, which takes a long time
- You must **know exactly what you want** before using AI tools. The instructor knew the representation and could verify the output.
- Students who are still learning will not get results as immediately, because they are still exploring
- The way to learn how to do it yourself is to **do it yourself first**, then use AI tools to accelerate

> **Course note**: The instructor recommended running Python files from the **command line** rather than through an IDE like VS Code. The IDE terminal is often not a standard terminal and can get in the way, as seen in Labs 1 and 2.

> **Course note**: A course on prompt engineering, agentic AI, managing context, and working with Claude Code would be valuable, but did not exist when this program was created.
