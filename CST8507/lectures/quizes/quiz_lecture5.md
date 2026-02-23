# NLP - Quiz Lecture 5

### Question 1

The core idea of an n-gram language model is to predict the next word based on the probability computed from counting how many times word patterns appear together in a corpus.

- True
- False

---

### Question 2

Suppose we have the following counts in a corpus:

- Count("opened their"): 500
- Count("opened their books"): 200
- Count("opened their laptops"): 150

Using a trigram model, what is P("books" | "opened their")?

- A) 0.3
- B) 0.4
- C) 0.7
- D) 0.2

---

### Question 3

A standard feedforward neural network language model does not require a fixed window size because it can naturally handle variable-length input sequences.

- True
- False

---

### Question 4

Which of the following best explains why RNNs are better suited for NLP tasks than standard feedforward neural networks?

- A) RNNs use deeper architectures with more hidden layers
- B) RNNs process input sequentially and propagate information between time steps through a hidden state
- C) RNNs do not require backpropagation during training
- D) RNNs process all input words simultaneously, making them faster

---

### Question 5

The vanishing gradient problem in RNNs occurs when the gradients become very large during backpropagation, causing the model to diverge.

- True
- False

---

### Question 6

Which component of an LSTM is responsible for deciding what information to erase from the previous cell state?

- A) Input gate
- B) Output gate
- C) Forget gate
- D) Candidate cell state

---

### Question 7

In an LSTM, the cell state serves as short-term memory while the hidden state serves as long-term memory.

- True
- False

---

### Question 8

A language model with low perplexity predicts the text well, meaning the text is expected by the model.

- True
- False

---

### Question 9

Consider the following simple neural network language model pipeline. What is the correct order of operations?

- A) One-hot encoding → Hidden layer → Embedding lookup → Softmax
- B) One-hot encoding → Embedding lookup → Hidden layer → Softmax
- C) Embedding lookup → One-hot encoding → Softmax → Hidden layer
- D) Softmax → Embedding lookup → Hidden layer → One-hot encoding

---

### Question 10

The GRU (Gated Recurrent Unit) simplifies LSTM by combining the forget and input gates into a single update gate and merging the cell state and hidden state.

- True
- False
