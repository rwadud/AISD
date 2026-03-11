# CST8509 Reinforcement Learning - Quiz 1

---

### Question 1

What is Reinforcement Learning (RL)?

- A) None of these answers.
- B) RL is a form of supervised machine learning used for learning to play games.
- C) RL is a third type of machine learning, along with supervised learning and unsupervised learning.
- D) All of these answers.
- E) RL is a form of unsupervised machine learning used in control applications.

---

### Question 2

Which of the following can be considered primary aspects of a Reinforcement Learning problem setup?

- A) Agent, Environment, and Reward.
- B) Reward, Environment, and States.
- C) None of these answers.
- D) Agent, Values, and Step function.
- E) Values, Step function, and Actions.

---

### Question 3

What is a Markov state?

- A) Intuitively, a Markov state has the property that its subsequent states do not depend on its previous states.
- B) None of these answers.
- C) Intuitively, a Markov state has the property that all its previous states completely determine its future states.
- D) All of these answers.
- E) Markov states are states that form a single deterministic chain.

---

### Question 4

What is the relationship between Reinforcement Learning (RL) and Markov Decision Processes (MDPs)?

- A) MDPs are a component of the software used to implement video games.
- B) MDPs are known specific strategies developed for playing games like chess, go, and video games played by RL systems.
- C) All of these answers.
- D) MDPs are a mathematical model of the sequential decision making processes addressed by RL.

---

### Question 5

What is the Reward Hypothesis of Reinforcement Learning?

- A) The Reward Hypothesis basically states that all goals can be thought of as maximizing the expected cumulative value of a scalar reward function.
- B) The Reward Hypothesis basically states that all goals can be thought of as minimizing the number of steps to maximize a scalar reward function.
- C) The Reward Hypothesis basically states that some goals cannot be thought of as maximizing the expected cumulative value of a scalar reward function.
- D) None of these answers.
- E) The Reward Hypothesis basically states that some goals cannot be thought of as minimizing the number of steps to maximize a scalar reward function.

---

### Question 6

What is meant by "episode" in Reinforcement Learning?

- A) An episode is a single run that does not reach the terminal state.
- B) An episode is the number of steps actually taken to reach the terminal state.
- C) None of these answers.
- D) An episode is a single run from the starting state to a terminal (or truncated) state.
- E) An episode is a single cycle of performing an action, receiving a reward, and observing the resulting state.

---

### Question 7

What role does the discount factor γ play in Reinforcement Learning?

- A) γ determines how many times an action is chosen randomly during training.
- B) None of these answers.
- C) γ addresses the problem of infinite cumulative rewards in non-terminating processes.
- D) γ represents the total discount which is subtracted from the reward function cumulative total.
- E) γ represents the weighting of the current goal of a Reinforcement Learning problem.

---

### Question 8

What is a Policy in Reinforcement Learning?

- A) All of these answers.
- B) The Policy is a function that determines the probability of an agent taking an action.
- C) The Policy is a table that assigns a value to each action.
- D) None of these answers.
- E) The Policy is a function that assigns a value to each action-state pair.

---

### Question 9

What is a Value Function in Reinforcement Learning?

- A) None of these answers.
- B) All of these answers.
- C) A Value Function gives a measure of the expected total number of steps to maximize reward.
- D) A Value Function gives a measure of the expected total reward given a state or state-action pair.
- E) A Value Function gives a measure of the expected total reward of an episode.

---

### Question 10

What is the difference between an action value function and a state value function?

- A) Action value functions return the average reward for taking an action, and State value functions return a state's average total future reward.
- B) None of these answers.
- C) State value functions take a state, and action value functions take just actions.
- D) State value functions return total reward to termination, and action-value functions return immediate reward of taking the action.
- E) Action value functions take state-action pairs, whereas state value functions take just states.

---

### Question 11

Which of the following statements is true about the Bellman equation in Reinforcement Learning?

- A) None of these answers.
- B) It breaks the problem of determining the value of a state into smaller problems recursively.
- C) It expresses the relationship between the value of a state or a state-action pair, and the value of the successor states.
- D) It forms the mathematical basis for the Q-Learning algorithm in Reinforcement Learning.
- E) All of these answers.

---

### Question 12

What does "greedy" mean in the context of Reinforcement Learning?

- A) It implies a policy where immediate reward is considered over future reward.
- B) It implies a policy that tries to maximize future reward.
- C) None of these answers.
- D) It implies a policy where future reward is considered over immediate reward.
- E) It implies a policy that tries to maximize total reward.

---

### Question 13

What is a condition for applying Q-learning to a Reinforcement Learning problem?

- A) The complete set of actions and the complete set of possible states must be known.
- B) The complete set of possible states must be known.
- C) None of these answers.
- D) The optimal value function must be known.
- E) The complete set of actions must be known.

---

### Question 14

Which of the following statements is true in the context of Reinforcement Learning?

- A) All of these answers.
- B) Temporal Distance (TD) learning does not require that the agent have a model of the environment.
- C) None of these answers.
- D) Q-learning is a form of Temporal Distance (TD) learning.
- E) Temporal Distance (TD) learning involves learning from differences in time steps as opposed to complete episodes.

---

### Question 15

Which of the following statements is true in the context of Reinforcement Learning?

- A) The results of an action are determined by the agent rather than the environment.
- B) The policy function is implemented in the environment rather than the agent.
- C) None of these answers.
- D) The value function and policy function are implemented in the agent rather than the environment.
- E) The value function is implemented in the environment rather than the agent.

---

## Answer Key

| # | Answer | Explanation |
|---|--------|-------------|
| 1 | **C** | RL is a third type of machine learning, along with supervised learning and unsupervised learning. |
| 2 | **A** | Agent, Environment, and Reward. |
| 3 | **A** | A Markov state's subsequent states do not depend on its previous states. |
| 4 | **D** | MDPs are a mathematical model of the sequential decision making processes addressed by RL. |
| 5 | **A** | All goals can be thought of as maximizing the expected cumulative value of a scalar reward function. |
| 6 | **D** | A single run from the starting state to a terminal (or truncated) state. |
| 7 | **C** | γ addresses the problem of infinite cumulative rewards in non-terminating processes. |
| 8 | **B** | The Policy is a function that determines the probability of an agent taking an action. |
| 9 | **D** | A Value Function gives a measure of the expected total reward given a state or state-action pair. |
| 10 | **E** | Action value functions take state-action pairs, whereas state value functions take just states. |
| 11 | **E** | All of these answers. |
| 12 | **A** | It implies a policy where immediate reward is considered over future reward. |
| 13 | **A** | The complete set of actions and the complete set of possible states must be known. |
| 14 | **A** | All of these answers. |
| 15 | **D** | The value function and policy function are implemented in the agent rather than the environment. |
