# CST8509 Final Exam Review

## How can Reinforcement Learning be applied when the number of states is huge?

Tabular Q-learning works only when the number of state-action pairs is small enough to store in a Q-table. When the number of states is huge, the agent cannot store an exact value for every state-action pair.

The solution is **value function approximation**. Instead of storing the value function in a table, use a parameterized function, usually a neural network, to approximate it:

$$
\hat{v}(s, w) \approx v(s)
$$

or, for action values:

$$
\hat{q}(s, a, w) \approx q(s, a)
$$

Here, `w` represents the neural network weights. The network takes an observation or state representation as input and outputs an estimated value. This lets the agent generalize from states it has seen to similar states it has not seen exactly before.

Examples from the notes:

- A small cliff-walking world can use a Q-table.
- A large Blocks World becomes too large for a practical Q-table.
- Go has around $10^{170}$ possible states, far beyond tabular storage.
- DQN and PPO are examples of algorithms that use neural networks for approximation.

The tradeoff is that the neural network is only an approximation. It can generalize, but it can also make wrong estimates.

## How DQN does value function approximation

**DQN** means **Deep Q-Network**. It is Q-learning where the Q-table is replaced by a neural network.

In ordinary Q-learning:

- The agent looks up `Q(state, action)` in a table.
- The table has one value for every state-action pair.
- The update changes one table cell directly.

In DQN:

- The neural network receives the state or encoded observation as input.
- It outputs one estimated Q-value for each possible action.
- The agent usually chooses the action with the highest estimated Q-value.
- The network weights are trained so the predicted Q-values move closer to Bellman-style target values.

For example, if the environment has eight actions, the DQN outputs:

$$
Q(s,0), Q(s,1), Q(s,2), \ldots, Q(s,7)
$$

The implied policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

DQN uses several stabilizing ideas:

- **Replay buffer**: stores past transitions such as `(state, action, reward, next_state, done)`.
- **Random batches**: trains on mixed old and new experiences rather than only the most recent transition.
- **Online network**: the network actively being trained.
- **Target network**: a delayed copy of the online network used to compute more stable target values.
- **Backpropagation**: adjusts the network weights to reduce the loss between predicted Q-values and target Q-values.

The high-level idea: DQN still learns action values like Q-learning, but the values come from a neural network prediction instead of a table lookup.

## How PPO does value function approximation

**PPO** means **Proximal Policy Optimization**. It is an **actor-critic** algorithm.

PPO uses two main neural networks:

- **Actor network**: represents the policy. It receives a state and outputs action probabilities.
- **Critic network**: represents the value function. It receives a state and outputs an estimated state value, $V(s)$.

The actor answers:

$$
s \rightarrow \pi(a \mid s)
$$

The critic answers:

$$
s \rightarrow V(s)
$$

The value function approximation part is mainly in the critic. The critic tries to estimate how good a state is by predicting the expected return from that state.

PPO training works roughly like this:

1. Use the actor policy to collect trajectories.
2. Calculate returns, also called rewards-to-go.
3. Use the critic to estimate state values.
4. Calculate **advantage**, which measures whether an action was better or worse than expected:

   $$
   A_t = G_t - V(s_t)
   $$

5. Train the critic by reducing the error between predicted values and observed returns.
6. Train the actor to make better-than-expected actions more likely and worse-than-expected actions less likely.
7. Use **clipping** so the policy does not change too much in one update.

The high-level idea: PPO approximates the state-value function with the critic network and uses that value estimate to improve the actor policy safely.

## Where would a neural network appear in an RL problem or solution?

A neural network normally appears inside the **agent**, not inside the environment.

It can be used as:

- A **value function approximator**, estimating $V(s)$ or $Q(s,a)$.
- A **policy network**, mapping states to action probabilities or actions.
- A **critic network**, evaluating how good states are.
- A **feature extractor**, converting raw observations such as images into useful numeric features.
- In model-based RL, possibly a learned **model** of environment dynamics, though this was not the main focus of the course.

In DQN, the neural network replaces the Q-table.

In PPO, the actor network represents the policy and the critic network approximates the value function.

The environment still supplies observations, rewards, and termination information. The neural network learns from those signals.

## Possible sources of training data for a neural network in RL

In reinforcement learning, the training data comes from interaction with the environment. It is not labelled data in the supervised learning sense.

A typical transition is:

```text
state, action, reward, next_state, done
```

Possible sources include:

- **Direct environment interaction**: the agent acts, receives observations and rewards, and stores the experience.
- **Replay buffer data**: DQN stores transitions and later samples random batches for neural network training.
- **Trajectory or rollout data**: PPO collects batches of full or partial trajectories, then calculates returns and advantages.
- **Simulation data**: for robotics, Gazebo can generate large amounts of experience without damaging a real robot.
- **Real robot data**: after training or testing on hardware, the real robot's sensor observations, actions, and rewards can also be used.
- **Exploratory behavior**: random actions or stochastic policies generate diverse experience early in training.

For the Create 3 red-ball task, training examples might look like:

```text
The camera reports the red ball's x position, for example 180.
The agent chooses a rotational action.
The action is converted into angular.z in a Twist message.
The next observation is the new red-ball x position.
The reward is best when the ball is at x = 320.
The episode continues.
```

The important point: RL training data is produced by the agent-environment loop.

## Assignment 2 strategy: Create 3 robot, simulation, and red-ball tracking

Assignment 2 sets up a reinforcement learning solution for a simulated iRobot Create 3 robot. The example task is red-ball tracking: the robot does not drive toward the ball, it rotates left or right to keep the moving ball centered in the camera view.

The major strategy is to train in simulation first, using a Gymnasium environment that communicates with Gazebo through ROS 2.

Major tools and projects:

| Tool or project | Purpose |
|---|---|
| **Gymnasium** | Provides the standard RL environment interface: `reset`, `step`, `render`, `close`, observation space, and action space. The assignment environment is registered as `aisd_examples/CreateRedBall-v0`. |
| **Stable Baselines 3** | Provides standard RL algorithms such as DQN and PPO. |
| **Q-learning, PPO, or DQN agent** | Learns a policy for choosing rotation actions. Q-learning uses a table; PPO and DQN use SB3. |
| **Non-RL agent** | Computes an action directly from the observation for comparison with learned agents. |
| **ROS 2** | Provides robot communication using topics, nodes, and messages. |
| **Twist messages** | Command robot rotation through `/cmd_vel`, using `angular.z`. |
| **Gazebo Classic / Gazebo 11** | Simulates the robot, physics, sensors, and world. |
| **iRobot Create 3 simulator** | Provides the simulated Create 3 robot. |
| **AWS Small House world** | Provides the simulated house environment. |
| **Virtual camera** | Gives the robot visual observations from inside the simulated world. |
| **Gazebo actor / red ball** | Provides an easy-to-detect moving target. |
| **Docker** | Provides a repeatable Ubuntu/ROS/Gazebo/Python environment. |
| **RViz** | Visualizes robot state and sensor data. |

High-level pipeline:

```text
SB3 agent, PPO or DQN
        ↓ action
Gymnasium environment
        ↓ uses ROS 2 node
Publish rotational Twist message to /cmd_vel
        ↓
Create 3 turns left or right in Gazebo
        ↓
Virtual camera observes the red ball
        ↓
Observation/state is computed from image
        ↓
Reward is calculated
        ↓
Agent trains from state, action, reward, next state
```

The Gymnasium environment sits between the RL agent and Gazebo. The agent does not need to know the details of ROS 2 or Gazebo. It only calls `env.step(action)` and receives the next observation, reward, and done flags.

The actual assignment environment uses:

- `observation_space = spaces.Discrete(640)`.
- The observation is the x-axis pixel column of the red ball in a 640-pixel-wide camera image.
- The center position is `320`, which is the optimal position.
- `action_space = spaces.Discrete(640)`.
- The action is converted into a rotation command:

  $$
  angular.z = \frac{action - 320}{320} \times \frac{5\pi}{12}
  $$

- An action near `320` means little or no rotation.
- Actions far from `320` command larger left/right rotations.
- The episode terminates after 100 steps.
- The reward is:

  $$
  reward = -|observation - 320|
  $$

This reward gives the best value, `0`, when the red ball is centered. Positions farther from center receive more negative rewards.

The Gymnasium environment contains a ROS 2 node called `RedBall`. That node subscribes to the virtual camera topic, detects the red ball with OpenCV, stores the ball's x position, publishes a marked-up `target_redball` image, publishes `Twist` messages to `cmd_vel`, and subscribes to `/stop_status`.

When the agent chooses an action, the environment publishes a `Twist` message. In the red-ball task, the important command is rotational: the simulated Create 3 turns left or right in Gazebo to track the moving ball.

The environment then uses camera data to determine the new observation. For red-ball tracking, the observation is the horizontal position of the red ball in the camera image. The reward encourages the robot to rotate so the moving ball stays centered.

A key synchronization issue is that ordinary ROS 2 nodes often use `spin()`, which runs indefinitely. In the RL loop, the environment must advance one step at a time, so the notes emphasize using `spin_once()` to process ROS messages while keeping the Gymnasium `step()` function synchronized with the robot's rotation action.

The `step()` method publishes the Twist command, calls `rclpy.spin_once()`, waits until `/stop_status` says the Create 3 is stopped, then returns the latest red-ball x position and reward.

The assignment also includes a non-RL comparison agent. Its purpose is to compute a sensible action directly from the observation, then compare its returns against Q-learning and SB3 agents.

In the current `../a2` code, the non-RL agent uses:

```python
action = 640 - observation
```

That mirrors the observed ball position into an action choice. The point is not that this is reinforcement learning; it is a hand-coded baseline for comparison.

Simulation is used because:

- It is safer than random exploration on a real robot.
- It can run many training episodes without a person physically moving a target.
- It avoids damaging hardware during early bad policies.
- It supports sim-to-real transfer: train in simulation, then try the learned behavior on the real robot.

## Improving unsatisfactory SB3 results on a Gymnasium environment

If an SB3 agent gives poor results, explore these areas.

### Environment correctness

Check that the Gymnasium environment is implemented correctly:

- `reset()` returns a valid initial observation.
- `step(action)` returns the correct observation, reward, terminated, truncated, and info values.
- `observation_space` accurately describes the observations.
- `action_space` accurately describes the available actions.
- Termination and truncation are used correctly.
- The environment is actually changing state in response to actions.

### Observation representation

Neural network algorithms do not work well with meaningless integer labels. In Assignment 2, the pixel value is more meaningful than an arbitrary state ID because `0`, `320`, and `639` have geometric meaning. Still, the representation can be improved.

Consider:

- Normalized numeric vectors.
- One-hot encodings for discrete values.
- Dictionary observations with `MultiInputPolicy`.
- Image observations with an appropriate image/CNN policy.
- Including enough information for the state to be Markov.

For CreateRedBall specifically, possible changes include:

- Represent the observation as normalized center error, such as `(x - 320) / 320`, instead of a raw `Discrete(640)` integer.
- Include the previous red-ball position, so the agent can infer whether the ball is moving left or right.
- Include a velocity estimate, such as `current_x - previous_x`.
- Use multiple recent frames or multiple recent x positions to better satisfy the Markov property.

### Reward design

Inspect whether the reward actually encourages the desired behavior:

- Is the reward too sparse?
- Is the reward too noisy?
- Does it accidentally reward the wrong behavior?
- Can the agent exploit the reward without solving the task?
- Should there be step penalties, progress rewards, or terminal rewards?

For CreateRedBall, the current reward is `-abs(observation - 320)`. That correctly makes center optimal, but possible experiments include rewarding improvement toward center, adding a bonus for staying within a centered tolerance range, or scaling/normalizing the reward so learning signals are easier for the algorithm.

Reward shaping is experimental. Try a reward structure, train, inspect results, revise, and repeat.

### Algorithm choice

Try a different algorithm if the problem structure suggests it:

- DQN is value-based and works with discrete actions.
- PPO is often robust and useful in robotics-style setups.
- If many actions are invalid in each state, consider action masking with `sb3_contrib`.
- If the action space is continuous, DQN is not appropriate.

### Hyperparameters

Tune the settings instead of assuming defaults are enough:

- Learning rate.
- Gamma / discount factor.
- Batch size.
- Network architecture.
- Exploration settings.
- `learning_starts`.
- Training frequency.
- Total timesteps.
- Episode length limit.

The notes mention RL Zoo as a tool for automated hyperparameter tuning.

### Training time and randomness

Poor results may simply mean the run has not trained long enough or that one seed was unlucky.

Explore:

- Longer training runs.
- Multiple random seeds.
- Reproducible seed settings.
- Comparing runs in TensorBoard.

The notes emphasize that changing only the seed can significantly change results.

### Action space design

If the action space is too large, the agent may waste most of its experience on bad or impossible actions.

Consider:

- Reducing the number of actions.
- Wrapping a `MultiDiscrete` action space into a `Discrete` action space if needed.
- Removing impossible actions.
- Using action masking.
- Designing actions that match the robot task naturally.

For CreateRedBall, `Discrete(640)` means the agent has 640 possible rotation commands. That is convenient because it mirrors the camera width, but it may be more action resolution than the task needs. A smaller action space, such as turn left, no turn, and turn right, or a small set of rotation strengths, may train more easily.

### Monitoring and debugging

Use tools to see what is happening:

- TensorBoard for reward curves, episode length, and loss.
- SB3 callbacks such as `EvalCallback` and `CheckpointCallback`.
- Environment rendering.
- Random-action tests to confirm the environment responds correctly.
- Logs from ROS 2, Gazebo, or Docker when using the Create 3 setup.

The general strategy is experimental: train, observe what failed, form a hypothesis, change one important thing, and compare the new run.
