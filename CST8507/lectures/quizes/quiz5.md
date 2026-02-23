# NLP - Quiz 5

### Question 1

Publicly available datasets, such as news articles, social media posts, and web pages, are commonly used as sources of data for training NLP models, as they provide a diverse range of language usage and context.

- True
- False

---

### Question 2

The core idea of an n-gram language model is to predict the next word by understanding the semantic meaning of entire sentences and applying deep reasoning.

- True
- False

---

### Question 3

When the learning rate is set too low, the training process will become much faster, and the model will reach the optimal solution quickly because small weight updates allow for faster progress.

- True
- False

---

### Question 4

"Stateful computation" in the context of Recurrent Neural Networks (RNNs) refers to maintaining internal memory states across multiple inputs.

- True
- False

---

### Question 5

Suppose we have the following sentence:

"Sunny days make people feel \_\_\_\_\_\_\_\_\_."

Let's assume we have a corpus, and we count the occurrences of the words "feel" and "feel happy" in that corpus.

- Count("feel"): 100 occurrences
- Count("feel happy"): 40 occurrences
- Count("happy"): 30 occurrences

The conditional probability P("happy" | "feel") is:

- A) 0
- B) 0.4
- C) 0.2
- D) 0.3

---

### Question 6

In the context of Recurrent Neural Networks (RNNs), the gradient refers to the rate of change of the loss function with respect to the network's parameters (weights and biases). During the training process, these gradients are computed using backpropagation to adjust the model's parameters in order to minimize the loss and improve the model's performance.

- True
- False

---

### Question 7

What is a significant advantage of Recurrent Neural Networks (RNNs) over traditional feedforward neural networks (FFNs) that makes them particularly suited for natural language processing tasks?

- A) RNNs can process fixed-length input sequences, making them ideal for tasks with static input sizes. RNNs can handle input sequences of varying lengths, unlike FFNs that require fixed-size input vectors.
- B) RNNs maintain an internal state that allows them to model sequential dependencies, which is crucial for tasks like language modeling and machine translation.
- C) RNNs are faster to train than FFNs because they do not require backpropagation.
- D) RNNs only process input in a single forward pass, making them more efficient than FFNs for sequential tasks.

---

### Question 8

Which of the following best explains why LSTMs are able to handle long-term dependencies better than standard RNNs?

- A) Because LSTMs use more hidden layers, which automatically prevent vanishing gradients.
- B) Because LSTMs replace the recurrent connection with a fully connected feedforward network.
- C) Because LSTMs use gating mechanisms that regulate information flow and help preserve gradients over long sequences.
- D) Because LSTMs remove backpropagation and instead rely only on forward propagation.
