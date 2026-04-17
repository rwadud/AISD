# CST8507: Natural Language Processing
## Week #9 - Transformer-Based Language Models

Developed by Hala Own, Ph.D.

---

## Lesson Agenda

- Transformer Architecture
- Self-Attention mechanisms
- Transfer Learning and Fine-Tuning
- Applications of transformer-based language models
- Drawbacks and variants of Transformers

---

## Text Representation Techniques

Three main categories of text representation:

**1. Sparse, Frequency-Based (Count-Based Features)**
- One-Hot Encoding
- Bag of Words
- Bag of N-Grams
- TF-IDF

**2. Dense Word Embeddings (Static) - Dense Vectors Capture Semantics**
- Word2Vec
- GloVe
- FastText

**3. Deep Contextual and Universal Embeddings (Contextual / LLM-based)**
- BERT / GPT / ELMo
- Sentence-BERT / USE / LaBSE
- LLM-Based Universal Embeddings
- Contrastive Learning and Retrieval Tasks

---

## Problem with Static Embeddings (word2vec)

- Fixed embeddings: One word = one vector (no context)
- Fixed at Training Time
- Out-of-vocabulary (OOV) problem
- Morphological Blindness (run, running, runner)

---

## Contextual Embeddings

- Representation of meaning of a word should be different in **different contexts**
- Each word has a **different vector**
- The meanings depend on the **surrounding words**

---

## Self Attention: Motivations

- Build up the **contextual embedding** for a word by selectively integrating information from all **neighboring words**, not equally but **weighted by relevance**.
- Each **word evaluates the importance** of the other words in the sentence and focuses more on those that provide useful context, while giving **less weight to less relevant** words.

---

## What is Self-Attention

Every word in a sequence asks: *"Which other words in this sentence are most relevant to understanding me?"*

Self-attention is the mechanism that answers that question for every word, simultaneously.

Example visualization: For the sentence "The animal didn't cross the street because it was too tired", self-attention shows which words the token "it_" attends to, with stronger connections to "The_", "animal_", and "too_".

---

## Why Multi-Head Attention?

**The Problem with Single-Head Attention**
- One weighted blend of all words.

Example sentence: "The **animal** didn't cross the street because **it was too tired**."

The word "it" needs to resolve three things simultaneously:
- **Coreference**: Refers to the animal (it → animal)
- **Syntactic**: Acts as the subject (subject of "was")
- **Semantic**: Must mean something alive (it → living thing)

A single attention head cannot capture all three relationships at once.

---

## Transformers (2017)

Paper: **Attention Is All You Need**

Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain, Google Research, University of Toronto)

**Abstract summary**: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. The Transformer is a new simple network architecture, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. The model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, the model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs.

Source: https://arxiv.org/abs/1706.03762

---

## What is Transformer

- The Transformer in NLP is a novel architecture that aims to solve **sequence-to-sequence** tasks while handling long-range dependencies with ease.
- The Transformer was proposed in the paper *Attention Is All You Need*.
- Relying entirely on **self-attention** to compute representations of its input and output.

---

## Transformer Architecture

Input: "Je suis étudiant" → THE TRANSFORMER → Output: "I am a student"

Internal structure: ENCODERS → DECODERS

---

## Transformer's Model Architecture

**Encoder (left side):**
- Input Embedding
- Positional Encoding (added via addition)
- N× stacked blocks containing:
  - Multi-Head Attention
  - Add and Norm
  - Feed Forward
  - Add and Norm

**Decoder (right side):**
- Output Embedding (outputs shifted right)
- Positional Encoding
- N× stacked blocks containing:
  - Masked Multi-Head Attention
  - Add and Norm
  - Multi-Head Attention (cross-attention from encoder)
  - Add and Norm
  - Feed Forward
  - Add and Norm
- Linear
- Softmax
- Output Probabilities

Source: https://arxiv.org/abs/1706.03762

---

## Encoder Anatomy

Flow from bottom to top:
1. Input: "Hello I love you"
2. Tokenization and Embedding
3. Positional Encoding (added)
4. N× stacked block containing:
   - Multi-Head Attention
   - Add and Layer Norm
   - Linear
   - Add and Layer Norm

---

## Encoder: Positional Encoding

Inputs like "Thinking" (x₁) and "Machines" (x₂) each get a positional encoding added before entering the self-attention layer.

Flow in Encoder #1:
1. Input embeddings x₁, x₂
2. Positional Encoding added
3. Self-Attention
4. Add and Normalize
5. Feed Forward (per position)
6. Add and Normalize
7. Output: z₁, z₂

---

## Encoder: Positional Encoding (Sigmoid and Sinusoidal)

Comparison visuals:
- **Sigmoid Function**: sig(t) = 1 / (1 + e^(-t)), ranges from 0 to 1
- **Sinusoidal functions**: y = sin(x°) and y = cos(x°), oscillating between -1 and 1

Sine and cosine waves are used because they allow the model to encode position information that the network can learn to attend to based on relative positions.

---

## Encoder: Positional Encoding Formulas

Positional encoding uses different frequencies across dimensions:

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:
- `pos` is the position
- `i` is the dimension index
- `d_model` is the model dimension

Lower dimensions oscillate quickly, higher dimensions oscillate slowly, creating a unique pattern for each position.

---

## Encoder: Positional Encoding (Visualization)

The positional vector is added to the embedding vector of the word at **position 2** in the input sequence.

Example values for positions p0, p1, p2, p3 across dimensions i=0 to i=3:

| Dimension | p0    | p1    | p2     | p3     |
|-----------|-------|-------|--------|--------|
| i=0       | 0.000 | 0.841 | 0.909  | 0.141  |
| i=1       | 1.000 | 0.540 | -0.416 | -0.990 |
| i=2       | 0.000 | 0.638 | 0.983  | 0.875  |
| i=3       | 1.000 | 0.770 | 0.186  | -0.484 |

**Settings**: d = 50. The value of each positional encoding depends on the position (pos) and dimension (d). We calculate result for every index (i) to get the whole vector.

Image source: https://towardsdatascience.com/understanding-positional-encoding-in-transformers-dc6bafc021ab

---

## Positional Encoding: Example

For the sentence "I love NLP" where d=4:

| Word | Position | Calculation | Encoding Vector |
|------|----------|-------------|-----------------|
| "I" | 0 | PE(0,0) = sin(0) = 0.0000<br>PE(0,1) = cos(0) = 1.0000<br>PE(0,2) = sin(0/100) = sin(0) = 0.0000<br>PE(0,3) = cos(0/100) = cos(0) = 1.0000 | [0, 1, 0, 1] |
| "love" | 1 | PE(1,0) = sin(1) ≈ 0.8415<br>PE(1,1) = cos(1) ≈ 0.5403<br>PE(1,2) = sin(1/100) = sin(0.01) ≈ 0.0100<br>PE(1,3) = cos(1/100) = cos(0.01) ≈ 1.0000 | [0.8415, 0.5403, 0.01, 0.9999] |
| "NLP" | 2 | PE(2,0) = sin(2) ≈ 0.9093<br>PE(2,1) = cos(2) ≈ -0.4161<br>PE(2,2) = sin(2/100) = sin(0.02) ≈ 0.0200<br>PE(2,3) = cos(2/100) = cos(0.02) ≈ 1.0000 | [0.9093, -0.4161, 0.02, 0.9998] |

---

## Encoder: Self Attention

The self-attention sub-layer is located between positional encoding and the feed-forward sub-layers in the encoder block.

---

## The Attention Mechanism Workflow

**Left diagram** (simplified flow):
1. Q and K → MatMul
2. → Scale
3. → SoftMax
4. → MatMul with V
5. → Output

**Right diagram** (detailed flow):
- **1st Step**: Query q combined with Keys k1, k2, k3, k4 → Similarity scores s1, s2, s3, s4
- **2nd Step**: Softmax on similarity scores → Weights a1, a2, a3, a4
- Multiply weights with Values V1, V2, V3, V4 and Sum → Attention Value

Image source: *Understanding Large Language Models: Learning Their Underlying Concepts and Technologies* book

---

## Self-Attention

Components:
- **W_Q**: Query weights
- **W_K**: Key weights
- **W_V**: Value weights
- **d_k**: attention head size

Flow:
1. Input Embeddings X (with rows for "I", "love", "tennis")
2. Compute Q = X × W_Q, K = X × W_K, V = X × W_V
3. Attention scores: S = softmax(Q × K_T / √d_k)
4. Example attention score matrix:

| | | | |
|---|---|---|---|
| 0.7 | 0.2 | 0.1 |
| 0.05 | 0.8 | 0.05 |
| 0.05 | 0.2 | 0.75 |

5. Attention output (S × V) → Context aware Embeddings for "I", "love", "tennis"

---

## Scale Dot-Product Motivation: SoftMax Sensitivity

**Small changes in logits can drastically change probabilities due to exponentiation.**

Softmax formula: w_ij = exp(w'_ij) / Σj exp(w'_ij)

**High Weight Values kill the gradient and slow down learning.**

Examples showing softmax outputs for different logit magnitudes:

For small d (logits [1, 2, 3, 4]):
- Softmax: [0.0321, 0.0871, 0.2369, 0.6439]

For medium d (logits [5, 10, 15, 20]):
- Softmax: [0.0000, 0.0000, 0.0067, 0.9933]

For large d (logits [10, 20, 30, 40]):
- Softmax: [0.0000, 0.0000, 0.0000, 1.0000]

---

## Scale Dot-Product Motivation: SoftMax Sensitivity (continued)

**Self Attention Mechanism (Scaling based on Square Root of Embedding Vector Dimension)**

√d is a good choice for scaling.

Example with d=100, √d=10:

Without scaling (logits [10, 20, 30, 40]):
- Softmax: [0.0000, 0.0000, 0.0000, 1.0000]

With scaling by √d ([10, 20, 30, 40] / 10 = [1, 2, 3, 4]):
- Softmax: [0.0321, 0.0871, 0.2369, 0.6439]

Scaling prevents the softmax from becoming too peaked, preserving useful gradients.

---

## Scale Dot-Product: Why?

Comparison of distributions:
- **Original Distribution**: Histogram of values ranging from roughly -400 to 400 (wide spread)
- **Scaled Distribution**: Histogram of values ranging from roughly -4 to 4 (narrow spread, normalized)
- **Softmax of Original**: Highly peaked with only 1-2 spikes dominating (near 0.6 and 0.4)
- **Softmax of Scaled**: Much more distributed probabilities across the range (values around 0.0005 to 0.0025)

The scaled version produces gradients across many positions instead of concentrating on a few.

---

## Encoder: Self (Scale Dot-Product) Attention

D = 64

For inputs "Thinking" and "Machines":
- Embeddings: x₁, x₂
- Queries: q₁, q₂ (computed via W^Q)
- Keys: k₁, k₂ (computed via W^K)
- Values: v₁, v₂ (computed via W^V)

Each vector has dimension 64 (typical in the original paper).

Slide Credit: Jay Alammar

---

## Transformer Architecture - Score

Steps to compute attention:
1. **Calculate scores**: the dot product of the query vector with the key vector of the respective word.
2. **Divide the scores by √64**
3. **Calculate Softmax**
4. **Multiply each value vector by the softmax score**
5. **Sum up the weighted value vectors.**

Worked example for "Thinking" attending to "Thinking" and "Machines":

| Step | Thinking | Machines |
|------|----------|----------|
| Embedding | x₁ | x₂ |
| Queries | q₁ | q₂ |
| Keys | k₁ | k₂ |
| Values | v₁ | v₂ |
| Score | q₁ · k₁ = 112 | q₁ · k₂ = 96 |
| Divide by 8 (√d_k) | 14 | 12 |
| Softmax | 0.88 | 0.12 |
| Softmax × Value | v₁ (weighted) | v₂ (weighted) |
| Sum | z₁ | z₂ |

The resulting z₁ is then passed to the feed-forward neural network.

---

## Encoder: Multi-headed Attention

Process:
1. Input X (with rows for "Thinking" and "Machines")
2. Calculating attention separately in eight different attention heads:
   - Attention Head #0 → Z₀
   - Attention Head #1 → Z₁
   - ...
   - Attention Head #7 → Z₇
3. **Concatenate** all the attention heads: [Z₀ Z₁ Z₂ Z₃ Z₄ Z₅ Z₆ Z₇]
4. **Multiply** with a weight matrix W^O that was trained jointly with the model
5. The result would be the Z matrix that captures information from all the attention heads. We can send this forward to the FFNN.

---

## The Feed-Forward Layer (position-wise feed-forward layer)

Formula:
$$FFN(Z) = ReLU(Z \cdot W_1 + b_1) \cdot W_2 + b_2$$

Where:
- **W₁, W₂**: weight matrices
- **b₁, b₂**: bias vectors

The feed-forward layer is applied position-wise (to each token independently) after the attention and Add and Norm step. Flow: Self-Attention → Add and Normalize → LayerNorm(X + Z) → Feed Forward → Add and Normalize.

---

## Residual Connections

- Let each layer **add refinements to the input rather than replace it**

Formula:
$$Output = Input + Layer(Input)$$

Residual connections skip around each sub-layer (Multi-Head Attention and Feed Forward), allowing gradients to flow directly through the network. They are shown as arrows going around the Add and Norm blocks in the architecture diagram.

---

## Add and Norm

**Add and Normalization** combines:
- **X**: Input Embedding (From first step) - the residual path
- **Z**: Output of Previous layer (From attention block)

These are summed and then passed through **Layer Normalization**.

The Add and Norm block appears N× in the encoder, after both the Multi-Head Attention and Feed Forward sub-layers.

---

## Parallel Processing

Transformers transform sequence elements in parallel. Self-attention for words (X₁, X₂, X₃, X₄) is calculated simultaneously rather than sequentially.

Each word's computation involves:
1. Dot Product (K·q)
2. Scaling by √d
3. Softmax
4. Weighted SUM with value vectors
5. Produces output Y₁, Y₂, Y₃, Y₄

All positions run in parallel, which is a key advantage over RNNs.

---

## Parallel Processing (Multi-Head)

**Multi Head Self Attention of Word X₂**

For each of H heads (h=1, h=2, h=3, h=4, h=5, etc.):
- Each head independently computes Similarity → Probabilities → Weighted SUM
- Each head produces its own output Y partial

All head outputs are then:
1. Concatenated into a vector of size 1×dH
2. Passed through a Linear Transformation [T] of size dH×d
3. Produces final output Y₂ of size 1×d

Each head learns different representation subspaces.

---

## Parallel Computation of Query, Key, and Value in Self-Attention

Three parallel matrix multiplications:

**Query computation:**
- X (N × d) × W^Q (d × d_k) = Q (N × d_k)
- Rows: Input Token 1, 2, 3, 4 → Query Token 1, 2, 3, 4

**Key computation:**
- X (N × d) × W^K (d × d_k) = K (N × d_k)
- Rows: Input Token 1, 2, 3, 4 → Key Token 1, 2, 3, 4

**Value computation:**
- X (N × d) × W^V (d × d_v) = V (N × d_v)
- Rows: Input Token 1, 2, 3, 4 → Value Token 1, 2, 3, 4

All three projections happen in parallel as single matrix operations.

---

## Decoder

The Decoder consists of:
1. Output Embedding (with outputs shifted right)
2. Positional Encoding (added)
3. N× stacked blocks containing:
   - **Masked Multi-Head Attention** (prevents attending to future tokens)
   - Add and Norm
   - **Multi-Head Attention** (cross-attention with encoder output)
   - Add and Norm
   - Feed Forward
   - Add and Norm
4. Linear
5. Softmax
6. Output Probabilities

---

## Masking the Future in Self-Attention

**Mask out attention to future words by setting attention scores to -∞.**

For a sequence [START], I, love, NLP, the mask forms an upper triangular pattern:

| | [START] | I | love | NLP |
|---|---|---|---|---|
| [START] | | -∞ | -∞ | -∞ |
| I | | | -∞ | -∞ |
| love | | | | -∞ |
| NLP | | | | |

For encoding these words (rows), we can look at these (not greyed out) words (columns to the left of and including the diagonal).

The -∞ values become 0 after softmax, effectively blocking attention to future positions.

---

## Masking the Future in Self-Attention (Matrix Form)

Flow of masked attention computation:

1. **Q (N × d_k)** × **K^T (d_k × N)** = **QK^T (N × N)**
   - Full matrix with all dot products: q1·k1, q1·k2, q1·k3, q1·k4 on row 1, etc.

2. **QK^T masked (N × N)** - upper triangle replaced with -∞:
   - Row 1: q1·k1, -∞, -∞, -∞
   - Row 2: q2·k1, q2·k2, -∞, -∞
   - Row 3: q3·k1, q3·k2, q3·k3, -∞
   - Row 4: q4·k1, q4·k2, q4·k3, q4·k4

3. Multiply by **V (N × d_v)** to get **A (N × d_v)** = [a1, a2, a3, a4]

---

## Cross-Attention: The Bridge Between Encoder and Decoder

In the decoder's second attention sub-layer (cross-attention):
- **Q** comes from the decoder (the Masked Multi-Head Attention output)
- **K** and **V** come from the encoder output

Formula:
$$CrossAttention(Q_{dec}, K_{enc}, V_{enc}) = softmax\left(\frac{Q_{dec} K_{enc}^T}{\sqrt{d_k}}\right) V_{enc}$$

This allows the decoder to attend to relevant parts of the input sequence while generating each output token.

---

## How the Encoder and the Decoder Stack Works

- The word embeddings of the input sequence are passed to the **first encoder**
- These are then transformed and propagated to the next encoder
- **The output from the last encoder in the encoder-stack is passed to all the decoders in the decoder-stack**

Example flow: Input "Komm bitte her" (German) → Encoder 1 → Encoder 2 → passed to all decoders → Decoder 1 (Self-Attention, Encoder-Decoder Attention, Feed Forward) → Decoder 2 → Output "Please come here"

Image source: https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/

---

## The Final SoftMax Layer

Pipeline for generating the next token:

1. **Decoder stack output**: a vector
2. **Linear**: projects to logits (size = vocab_size)
3. **Softmax**: produces log_probs (probabilities over vocabulary)
4. **Argmax**: Get the index of the cell with the highest value (example: index 5)
5. Look up which word in our vocabulary is associated with this index (example: "am")

---

## The Hugging Face Ecosystem

**Hugging Face Hub** (top level):
- Models
- Datasets
- Metrics
- Docs

**Core Libraries** (connected bidirectionally to Hub):
- Tokenizers
- Transformers
- Datasets
- Accelerate (connects with Transformers)

Reference: *Natural Language Processing with Transformers*, O'Reilly Media, Inc, 2022

---

## The Hugging Face Hub

The Hub provides a searchable interface with:

**Tasks** available:
- Fill-Mask
- Question Answering
- Summarization
- Table Question Answering
- Text Classification
- Text Generation
- Text2Text Generation
- Token Classification
- Translation
- Zero-Shot Classification
- Sentence Similarity (+12 more)

**Libraries**: PyTorch, TensorFlow, JAX (+19 more)

**Datasets**: wikipedia, common_voice, bookcorpus, dcep europarl jrc-acquis, glue, squad

**Example Models** (25,493 total models shown):
- bert-base-uncased (Fill-Mask, 27.5M downloads, 42 likes)
- xlm-roberta-base (Fill-Mask, 5.88M downloads, 9 likes)
- roberta-large (Fill-Mask, 5.26M downloads, 15 likes)
- distilbert-base-uncased (Fill-Mask, 4.86M downloads, 22 likes)
- gpt2 (Text Generation, 4.64M downloads, 15 likes)

---

## Transformer Applications: Text Classification

Example text:
> "Dear Amazon, last week I ordered an Optimus Prime action figure from your online store in Germany. Unfortunately, when I opened the package, I discovered to my horror that I had been sent an action figure of Megatron instead! As a lifelong enemy of the Decepticons, I hope you can understand my dilemma. To resolve the issue, I demand an exchange of Megatron for the Optimus Prime figure I ordered. Enclosed are copies of my records concerning this purchase. I expect to hear from you soon. Sincerely, Bumblebee."

Code:
```python
from transformers import pipeline
classifier = pipeline("text-classification")

import pandas as pd
outputs = classifier(text)
pd.DataFrame(outputs)
```

Output:

| | label | score |
|---|---|---|
| 0 | NEGATIVE | 0.901546 |

---

## Transformer Applications: Named Entity Recognition

Code:
```python
ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs = ner_tagger(text)
pd.DataFrame(outputs)
```

Output:

| | entity_group | score | word | start | end |
|---|---|---|---|---|---|
| 0 | ORG | 0.879010 | Amazon | 5 | 11 |
| 1 | MISC | 0.990859 | Optimus Prime | 36 | 49 |
| 2 | LOC | 0.999755 | Germany | 90 | 97 |
| 3 | MISC | 0.556569 | Mega | 208 | 212 |
| 4 | PER | 0.590256 | ##tron | 212 | 216 |
| 5 | ORG | 0.669692 | Decept | 253 | 259 |
| 6 | MISC | 0.498350 | ##icons | 259 | 264 |
| 7 | MISC | 0.775361 | Megatron | 350 | 358 |
| 8 | MISC | 0.987854 | Optimus Prime | 367 | 380 |
| 9 | PER | 0.812096 | Bumblebee | 502 | 511 |

---

## Transformer Applications: Question Answering

Code:
```python
reader = pipeline("question-answering")
question = "What does the customer want?"
outputs = reader(question=question, context=text)
pd.DataFrame([outputs])
```

Output:

| | score | start | end | answer |
|---|---|---|---|---|
| 0 | 0.631291 | 335 | 358 | an exchange of Megatron |

---

## Transformer Applications: Summarization

Code:
```python
summarizer = pipeline("summarization")
outputs = summarizer(text, max_length=80, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])
```

Output:
> Bumblebee ordered an Optimus Prime action figure from your online store in Germany. Unfortunately, when I opened the package, I discovered to my horror that I had been sent an action figure of Megatron instead.

---

## The Transformer Tree of Life

**Transformer** splits into three main branches:

**Encoder-only branch** (from Encoder):
- BERT → DistilBERT, RoBERTa
- XLM → XLM-R
- ALBERT
- ELECTRA
- DeBERTa

**Encoder-Decoder branch** (from both Encoder and Decoder):
- T5
- BART
- M2M-100
- BigBird

**Decoder-only branch** (from Decoder):
- GPT
- GPT-2 → CTRL
- GPT-3
- GPT-Neo → GPT-J

Reference: *Natural Language Processing with Transformers*, O'Reilly Media, Inc, 2022

---

## Challenges with Transformers

- *Language*
- *Data availability*
- *Working with long documents*
- *Transparency*
- *Bias*

---

## Summary

**Attention** is a mechanism in neural networks that focuses on a specific part of the input and computes its context-dependent summary. It works like a "soft" version of a key-value store.

**Self-attention** is an attention mechanism that produces the summary of the input by summarizing itself.

The **Transformer model** applies self-attention repeatedly to gradually transform the input.

---

## Q&A