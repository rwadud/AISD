# NLP - Quiz Lecture 6

### Question 1

In the Seq2Seq encoder-decoder framework, the encoder compresses the entire source sentence into a single fixed-length vector called the context vector, which is then used by the decoder to generate the output sequence.

- True
- False

---

### Question 2

Which of the following best describes the bottleneck problem in Seq2Seq models?

- A) The decoder runs too slowly to generate output in real time
- B) The encoder must compress all source sentence information into a single fixed-length vector, which becomes increasingly difficult for longer sequences
- C) The model cannot handle multiple languages simultaneously
- D) The attention weights are always evenly distributed across all encoder states

---

### Question 3

In a Bidirectional LSTM, the forward and backward LSTMs share the same set of weights to ensure consistency.

- True
- False

---

### Question 4

What is the purpose of the attention mechanism in Seq2Seq models?

- A) To replace the encoder entirely
- B) To allow the decoder to look at all encoder hidden states and focus on the most relevant parts of the input at each decoding step
- C) To reduce the vocabulary size of the model
- D) To prevent the vanishing gradient problem during training

---

### Question 5

In the LSTM cell state update formula $c_t = f_t \odot c_{t-1} + i_t \odot \hat{c}_t$, the first part ($f_t \odot c_{t-1}$) adds new information while the second part ($i_t \odot \hat{c}_t$) filters old information.

- True
- False

---

### Question 6

Which of the following correctly describes the difference between GRU and LSTM?

- A) GRU has more parameters and is slower to train than LSTM
- B) GRU uses 3 gates (forget, input, output) while LSTM uses 2 gates (reset, update)
- C) GRU combines the forget and input gates into a single update gate and merges the cell state and hidden state
- D) GRU maintains both a hidden state and a separate cell state like LSTM

---

### Question 7

In the attention mechanism, softmax is applied to the similarity scores between encoder and decoder hidden states to produce an attention distribution (probability distribution over encoder states).

- True
- False

---

### Question 8

Machine translation is an example of which type of sequence problem?

- A) One-to-one
- B) One-to-many
- C) Many-to-one
- D) Many-to-many

---

### Question 9

Backpropagation Through Time (BPTT) in an RNN is called "through time" because the gradient at each time step depends on the gradients at subsequent time steps, making it time dependent.

- True
- False

---

### Question 10

The Transformer architecture, introduced in the 2017 paper "Attention Is All You Need," relies entirely on recurrent neural network components and does not use self-attention.

- True
- False
