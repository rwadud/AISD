# CST8507: Natural Language Processing

## Week #6 — Sequence to Sequence Models (Seq2Seq)

Developed by Hala Own, Ph.D.
Algonquin College

---

## Slide 2: Lesson Agenda

- Midterm (week 7)
- Bidirectional Long Short Term Memory (Bi-LSTM)
- The encoder-decoder framework
- Attention mechanisms

---

## Slide 3: Midterm

Midterm is on Tuesday, Feb. 23, at 2:00 pm.

- The exam will consist of **30 questions**, including multiple-choice and true/false questions, with no essay questions.
- Material includes from week 1 – week 6
- **You will have 60 minutes to complete the exam.**
- The exam is closed book. However, you may bring **one cheat sheet**: a single **letter-size page (8.5 × 11 inches)** that may be **used on both sides**.
- **Ensure that you leave a 5 cm by 5 cm space in the top-left corner of each side of your cheat sheet for the proctor's signature. If this specific area is missing, you will not be allowed to use any cheat sheet during the exam.**
- **Try to arrive early to allow sufficient time for setup.**

---

## Slide 4: Midterm

- **Read the instruction before starting your exam.**
- Write your **name and ID number** on the spaces provided on the questionnaire and Answer sheet.
- **Make sure to have your ID.**
- Read carefully the ICT exam conduct outline
- Please do not forget to bring your **HB pencils and eraser**.
- Scantron answer sheets will be provided to you before the start of the exam together with the questionnaire.
- Submit **both** the questionnaire and the Scantron answer sheet

---

## Slide 5: How to Prepare

- Lecture summary slides are a good place to start:
  - they don't have all the details, but make sure you understand the details underlying the main points mentioned.
- Do the labs! Make sure you understand the answers you get.
- Code-Examples demonstrated during the lecture (check lecture materials folder on Brightspace).
- Hybrid work

---

## Slide 6: Questions

---

## Slide 7: Recap — N-gram

**N-grams**

$$P(w_2|w_1) = \frac{\text{count}(w_1, w_2)}{\text{count}(w_1)} \longrightarrow \text{Bigrams}$$

$$P(w_3|w_1, w_2) = \frac{\text{count}(w_1, w_2, w_3)}{\text{count}(w_1, w_2)} \longrightarrow \text{Trigrams}$$

$$P(w_1, w_2, w_3) = P(w_1) \times P(w_2|w_1) \times P(w_3|w_2)$$

- Large N-grams to capture dependencies between distant words
- Need a lot of space and RAM

---

## Slide 8: Bigram Probability

*I have a dog whose name is Lucy.*
*I have two cats.*
*they like playing with Lucy.*

$$P(A|B) = \frac{P(A,B)}{P(B)}$$

$$P(\text{have}|\text{I}) = \frac{P(\text{I have})}{P(\text{I})} = \frac{2}{2} = 1$$

$$P(\text{two}|\text{have}) = \frac{P(\text{have two})}{P(\text{have})} = \frac{1}{2} = 0.5$$

$$P(\text{eating}|\text{have}) = \frac{P(\text{have eating})}{P(\text{have})} = \frac{0}{2} = 0$$

$$P(w_2|w_1) = \frac{C(w_1, w_2)}{\sum_w C(w_1, w)} = \frac{C(w_1, w_2)}{C(w_1)}$$

---

## Slide 9: Recap — RNN

Designed to handle sequential data by maintaining a **hidden state** that captures information from previous time steps.

Example: "I called her but she did not ___"

The RNN processes each word sequentially (I → called → her → but → she → did → not), maintaining a hidden state that is passed forward through each step. Weights $W_h$ (hidden) and $W_x$ (input) are shared across time steps.

*Source of image: deeplearning.ai*

---

## Slide 10: Recap — LSTM

**LSTM** networks are a type of RNN that can learn long-term dependencies. They use gates (input, forget, and output gates) to **control the flow of information**, making them effective for tasks requiring memory over long sequences.

Architecture components:
- Previous Cell State → New Cell State (top line)
- Previous Hidden State → New Hidden State (bottom line)
- Input Data $x_t$
- Gates use sigmoid (σ) and tanh activations
- Pointwise multiplication (×) and addition (+)

---

## Slide 11: RNN vs LSTM cell

**RNN**: Simple cell with a single tanh operation.

**LSTM**: Complex cell with multiple gates (σ, σ, tanh, σ) and pointwise operations, providing a cell state path separate from the hidden state.

---

## Slide 12: Types Of Sequence Problems in NLP Task

| Type | Example |
|------|---------|
| One to one | Image classification |
| One to many | Image captioning |
| **Many to one** | **Sentimental analysis** |
| Many to many | Stock Market prediction |
| Many to many | Translation |

(source: http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

---

## Slide 13: Bidirectional Long Short-Term Memory (Bi-LSTM): Motivation

A recurrent cell $A$ processes inputs $x_0, x_1, x_2, \ldots, x_t$ producing hidden states $h_0, h_1, h_2, \ldots, h_t$.

Example: *"The movie was terribly exciting!"*

---

## Slide 14: Bi-LSTM Motivation (continued)

Sentence encoding via element-wise mean/max pooling over hidden states produces the sentence representation, which can be classified (e.g., positive sentiment).

Input tokens: *the movie was terribly exciting !*

Each token produces a hidden state vector; element-wise mean/max pooling combines them into a single sentence encoding → classification (e.g., "positive").

*Slide credit: Daniel Jurafsky*

---

## Slide 15: Bi-LSTM

**Forward RNN** processes tokens left-to-right.
**Backward RNN** processes tokens right-to-left.

The **concatenated hidden states** at each time step combine both forward and backward representations.

> This contextual representation of "terribly" has both left and right context!

For input: *the movie was terribly exciting !*

*Slide credit: Daniel Jurafsky*

---

## Slide 16: Bi-LSTM (continued)

Diagram shows forward arrows $\vec{h}^{<t>}$ and backward arrows $\overleftarrow{h}^{<t>}$ for each timestep.

$\hat{y}^{<t_1>}, \hat{y}^{<t_2>}, \ldots, \hat{y}^{<T>}$ are outputs computed from both directions.

**Information flows from the past and from the future independently**

*Source of image: deeplearning.ai*

---

## Slide 17: Bi-LSTM (formula)

$$\hat{y}^{<t>} = g(W_y[\vec{h}^{<t>}, \overleftarrow{h}^{<t>}] + b_y)$$

The output at time $t$ combines the concatenation of forward and backward hidden states.

---

## Slide 18: Bidirectional RNNs

On timestep $t$:

- **Forward RNN**: $\vec{h}^{(t)} = \text{RNN}_{FW}(\vec{h}^{(t-1)}, x^{(t)})$
- **Backward RNN**: $\overleftarrow{h}^{(t)} = \text{RNN}_{BW}(\overleftarrow{h}^{(t+1)}, x^{(t)})$
- **Concatenated hidden states**: $h^{(t)} = [\vec{h}^{(t)}; \overleftarrow{h}^{(t)}]$

> This is a general notation to mean "compute one forward step of the RNN" — it could be a vanilla, LSTM or GRU computation.

> Generally, these two RNNs have separate weights.

> We regard $h^{(t)}$ as "the hidden state" of a bidirectional RNN. This is what we pass on to the next parts of the network.

*Slide credit: Daniel Jurafsky*

---

## Slide 19: Bidirectional Long Short-Term Memory (Bi-LSTM)

**Single forward LSTM layer** (left): Shows gates $f_i^l, i_i^l, g_i^l, o_i^l$ with sigmoid and tanh activations producing cell state $C_i^l$ and hidden state $\vec{h}_i^l$.

**Bi-LSTM model** (right): Forward LSTM chain (bottom) and Backward LSTM chain (top) operate in parallel, producing bidirectional hidden states.

---

## Slide 20: Bi-LSTM model Architecture for Classification

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

model = Sequential([
    Embedding(input_dim=vocab_size,
              output_dim=embedding_dim,
              input_length=max_len),
    Bidirectional(LSTM(n_lstm)),
    Dense(1, activation='sigmoid')
])
```

---

## Slide 21: Multi-layer RNNs / LSTM

Stacked RNN layers:
- **RNN layer 1** (bottom): takes input embeddings (e.g., *the movie was terribly exciting !*)
- **RNN layer 2** (middle): takes hidden states from layer 1 as input
- **RNN layer 3** (top): takes hidden states from layer 2 as input

> The hidden states from RNN layer $i$ are the inputs to RNN layer $i+1$

*Slide credit: Daniel Jurafsky*

---

## Slide 22: Multi-layer RNNs (continued)

**Layer 0**: RNN Cell 0 processes input $x_0, x_1, \ldots$ producing hidden states $h_{00}, h_{01}, \ldots$

**Layer 1**: RNN Cell 1 takes outputs from layer 0 as input, producing $h_{10}, h_{11}, \ldots$

Initial/final hidden states: $h_{0i}, h_{0f}, h_{1i}, h_{1f}$.

---

## Slide 23: Types of Sequence Problems

- one to many
- many to one
- **many to many** (highlighted)

(source: http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

---

## Slide 24: Introduction to Sequence-to-Sequence (Seq2Seq)

- Seq2Seq is a type of model used to transform one **sequence into another sequence**.
- Commonly used in tasks where the input and output are sequences of **varying lengths**.

---

## Slide 25: (Encoder-Decoder) Model — solution to Seq2Seq task

**High-level view**: INPUT → ENCODER → STATE → DECODER → Output

**Detailed view**:
- **Encoder**: LSTM chain processes input $x_1, x_2, x_3$ producing hidden states $h_1, h_2, h_3$ and a final **Encoder Vector**.
- **Decoder**: LSTM chain initialized from the Encoder Vector generates output tokens $y_1, y_2$.

*Image source: https://pradeep-dhote9.medium.com/seq2seq-encoder-decoder-lstm-model-1a1c9a43bbac*

---

## Slide 26: Machine Translation (MT)

- The sequence-to-sequence model is an example of a **Conditional Language Model**.
  - **Language Model**: task is predicting the next word of the target sentence $y$
  - **Conditional**: predictions are also conditioned on the source sentence $x$

- MT directly calculates:

$$P(y|x) = P(y_1|x) \, P(y_2|y_1, x) \, P(y_3|y_1, y_2, x) \ldots P(y_T|y_1, \ldots, y_{T-1}, x)$$

The final term is the probability of the next target word, given target words so far and source sentence $x$.

*Slide credit: Daniel Jurafsky*

---

## Slide 27: Training a Neural Machine Translation system

$$J = \frac{1}{T} \sum_{t=1}^{T} J^t = J_1 + J_2 + J_3 + J_4 + J_5 + J_6 + J_7$$

Where $J_1$ = negative log prob of "he", $J_4$ = negative log prob of "with", $J_7$ = negative log prob of `<END>`.

- **Encoder RNN** processes source sentence: *il a m' entarté*
- **Decoder RNN** produces target sentence: `<START>` he hit me with a pie
- At each step, $\hat{y}^{(t)}$ is the predicted distribution over the target vocabulary.

> Seq2seq is optimized as a **single system**. Backpropagation operates "*end-to-end*".

*Slide credit: Daniel Jurafsky*

---

## Slide 28: Neural Machine Translation (Testing)

The sequence-to-sequence model at test time:

- **Encoder RNN** produces an encoding of the source sentence (*il a m' entarté*).
- The encoding provides the initial hidden state for the **Decoder RNN**.
- **Decoder RNN** is a Language Model that generates the target sentence, conditioned on encoding.
- At each step, `argmax` selects the next token; the decoder output is fed in as the next step's input.

Output: *he hit me with a pie* `<END>`

*Slide credit: Daniel Jurafsky*

---

## Slide 29: Sequence-to-sequence: bottlenecks problem

The encoding of the source sentence **needs to capture all information about the source sentence**. **Information bottleneck!**

**Problems with this architecture?**
- Fixed-size encoding must compress entire source sequence
- Long sequences lose information
- Decoder has no direct access to source tokens

*Slide credit: Daniel Jurafsky*

---

## Slide 30: Sequence-to-sequence: Limitations

Pair of RNN used for translation. The bottleneck is the single **sentence representation** vector that connects encoder to decoder.

Example: *Maria no baba verde .* → *Mary did not ... witch .* `<END>`

*Natural Language Processing with Transformers, O'Reilly Media, Inc, 2022*

---

## Slide 31: Solution with Attention

*(Illustration of a confused person with question marks, representing the motivating need for attention.)*

---

## Slide 32: What is attention?

- Attention is a **weighted average over a set of inputs**
- How should we compute this weighted average?
  - **Compute** pairwise similarity between **each encoder hidden state** and **decoder hidden state**.
  - **Convert** pairwise similarity scores to probability **distribution** (using softmax) over encoder hidden states and compute weighted average.

---

## Slide 33: Attention

Solution to the **bottleneck problem**.

**Benefits**
- Improved handling of **variable-length input sequences**.
- Enhanced modeling of **long-range dependencies**.
- Better performance in tasks where certain parts of the input sequence are **more relevant** to specific parts of the output sequence.

**Core idea**: on **each step** of the decoder, **use direct connection** to the encoder to focus on a particular part of the source sequence.

---

## Slide 34: Sequence-to-sequence with attention

At the first decoder step (`<START>`), compute attention scores as the **dot product** between the decoder hidden state and the first encoder hidden state.

Source: *il a m' entarté*

*Slide credit: Daniel Jurafsky*

---

## Slide 35: Sequence-to-sequence with attention (continued)

Continue computing dot products between the decoder hidden state and each encoder hidden state, producing an attention score per encoder position.

*Slide credit: Daniel Jurafsky*

---

## Slide 36: Sequence-to-sequence with attention (continued)

Attention scores are computed for all encoder positions (il, a, m', entarté) relative to the current decoder state.

*Slide credit: Daniel Jurafsky*

---

## Slide 37: Sequence-to-sequence with attention (continued)

All four encoder hidden states have corresponding attention scores.

*Slide credit: Daniel Jurafsky*

---

## Slide 38: Sequence-to-sequence with attention (continued)

> Take softmax to turn the scores into a probability distribution (attention distribution).

> On this decoder timestep, we're mostly focusing on the first encoder hidden state ("*he*").

*Slide credit: Daniel Jurafsky*

---

## Slide 39: Sequence-to-sequence with attention (continued)

> Use the attention distribution to take a **weighted sum** of the encoder hidden states.

> The attention output mostly contains information from the hidden states that received high attention.

*Slide credit: Daniel Jurafsky*

---

## Slide 40: Sequence-to-sequence with attention (continued)

> Concatenate attention output with decoder hidden state, then use to compute $\hat{y}_1$ as before.

Output token at step 1: *he*

*Slide credit: Daniel Jurafsky*

---

## Slide 41: Sequence-to-sequence with attention (step 2)

At the second decoder step (input *he*), attention focuses on different encoder positions (e.g., "entarté") to produce output *hit*.

$\hat{y}_2$ → *hit*

*Slide credit: Daniel Jurafsky*

---

## Slide 42: Sequence-to-sequence with attention (step 3)

At decoder step 3 (input *hit*), attention distribution changes to focus on the relevant source token (e.g., "m'"), producing output *me*.

$\hat{y}_3$ → *me*

*Slide credit: Daniel Jurafsky*

---

## Slide 43: Sequence-to-sequence with attention (step 4)

At decoder step 4 (input *me*), attention focuses heavily on "entarté", producing output *with*.

$\hat{y}_4$ → *with*

*Slide credit: Daniel Jurafsky*

---

## Slide 44: Sequence-to-sequence with attention (step 5)

At decoder step 5 (input *with*), attention focuses on "entarté", producing output *a*.

$\hat{y}_5$ → *a*

*Slide credit: Daniel Jurafsky*

---

## Slide 45: Sequence-to-sequence with attention (step 6)

At decoder step 6 (input *a*), attention focuses on "entarté", producing output *pie*.

$\hat{y}_6$ → *pie*

*Slide credit: Daniel Jurafsky*

---

## Slide 46: Attention Mechanism Benefits vs Challenges

How does attention address the temporal bottleneck in sequence-to-sequence models?

---

## Slide 47: Transformers (2017)

**Attention Is All You Need**

Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain / Google Research / University of Toronto)

**Abstract (summary)**: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. The authors propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. The model achieves 28.4 BLEU on WMT 2014 English-to-German translation, improving over existing best results (including ensembles) by over 2 BLEU. On WMT 2014 English-to-French, a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs.

*https://arxiv.org/abs/1706.03762*

---

## Slide 48: What is Transformer

- The Transformer in NLP is a novel architecture that aims to **solve sequence-to-sequence** tasks while handling long-range dependencies with ease.
- The Transformer was proposed in the paper ***Attention Is All You Need***.
- Relying entirely on **self-attention** to compute representations of its input and output.

*https://arxiv.org/abs/1706.03762*

---

## Slide 49: Transformer Architecture

**High-level view**: INPUT (*Je suis étudiant*) → THE TRANSFORMER → OUTPUT (*I am a student*)

**Expanded view**: INPUT → ENCODERS → DECODERS → OUTPUT

---

## Slide 50: Q&A

*"The smartest people are those who ask questions"* — Einstein meme
