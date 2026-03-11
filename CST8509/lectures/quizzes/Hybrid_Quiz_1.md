# Hybrid Quiz 1

---

### Question 1 (1 point)

What is a Markov chain?

- A) A sequence of nodes in a graph without cycles
- B) None of these answers
- C) A sequence of nodes in a graph with cycles
- D) A mathematical model that experiences transition of states with probabilistic rules
- E) A chain with a rubber coating

---

### Question 2 (1 point)

What is a Markov Decision Process?

- A) The underlying logic of a Turing Machine
- B) A process for making a decision between two options
- C) A process for making a decision between more than two options
- D) None of these answers
- E) An extension of the Markov chain with actions and rewards

---

### Question 3 (1 point)

In a Markov Decision Process, taking an action in a state always leads to the same result state.

- A) True
- B) False

---

### Question 4 (1 point)

What is a problem with defining total reward from a starting point to be the sum of all subsequent rewards?

- A) The sum of all subsequent rewards might be negative
- B) The sum of all subsequent rewards might be zero
- C) None of these answers
- D) The sum of all subsequent rewards might be infinite
- E) The sum of all subsequent rewards might be positive

---

### Question 5 (1 point)

What is a policy in Reinforcement Learning?

- A) A function that specifies what action to take in a certain state
- B) A function that gives the list of all the possible actions in a state
- C) A function that specifies the next state to visit
- D) A function that gives the list of all impossible actions in a state
- E) None of these answers

---

### Question 6 (1 point)

What is given by the state value function?

- A) It takes an action and a state and gives the expected total reward we can get from taking that action
- B) It takes an action and gives the state that results from taking that action
- C) None of these answers
- D) It takes a state and gives the expected total reward we can get starting from that state
- E) It takes a state and gives an action that results in the highest reward

---

### Question 7 (1 point)

What is the action value function?

- A) It takes a state and an action and gives the state resulting from taking the action
- B) It takes a state and an action and gives the expected total reward we can get starting from that state and taking that action
- C) It takes a state and an action and gives the immediate reward resulting from taking that action
- D) None of these answers
- E) It takes a state and gives expected total reward we can get starting from that state

---

### Question 8 (1 point)

What is a greedy policy?

- A) A policy that dictates always taking rather than giving
- B) A policy that rotates through all actions
- C) A policy that always specifies the same action
- D) A policy that dictates always taking the action that results in the highest immediate reward
- E) None of these answers

---

### Question 9 (1 point)

What does the Bellman Equation say in the context of Q-Learning?

- A) It says that the value of taking an action *a* in some state *s* is the immediate reward you get for taking that action, plus the **minimum** expected future rewards you can get in the next state.
- B) It says that the value of taking an action *a* in some state *s* is the immediate reward you get for taking that action, **minus** the maximum expected future rewards you can get in the next state.
- C) It says that the value of taking an action *a* in some state *s* is the immediate reward you get for taking that action, plus the **maximum** expected future rewards you can get in the next state.
- D) None of these answers
- E) It says that the value of an action *a* in some state *s* is the immediate reward you get for taking that action, plus the **total past rewards** from the previous next state.

---

## Answer Key

| Question | Answer |
|----------|--------|
| 1 | **D** — A mathematical model that experiences transition of states with probabilistic rules |
| 2 | **E** — An extension of the Markov chain with actions and rewards |
| 3 | **B** — False |
| 4 | **D** — The sum of all subsequent rewards might be infinite |
| 5 | **A** — A function that specifies what action to take in a certain state |
| 6 | **D** — It takes a state and gives the expected total reward we can get starting from that state |
| 7 | **B** — It takes a state and an action and gives the expected total reward we can get starting from that state and taking that action |
| 8 | **D** — A policy that dictates always taking the action that results in the highest immediate reward |
| 9 | **C** — Immediate reward plus the maximum expected future rewards in the next state |
