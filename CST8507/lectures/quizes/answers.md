# NLP - Quiz Answers

## Quiz Lecture 1

1. **False** — NLP converts unstructured documents into structured knowledge, not the reverse.
2. **C) Attachment ambiguity** — The phrase "with a telescope" can attach to either "saw" or "the man."
3. **False** — Zipf's Law states frequency is *inversely* proportional to rank.
4. **C) Machine translation** — Translation requires both understanding input (NLU) and generating output (NLG).
5. **False** — In NLP, rare words can be critically important (e.g., "fuzzy logic" may appear once but carry significant meaning).
6. **False** — Summarization is one of the *hardest* NLP applications.
7. **A)** — Common knowledge refers to facts humans take for granted that must be explicitly encoded for machines.
8. **False** — ELIZA (1960s) was rule-based, not deep learning.
9. **B)** — DL ⊂ ML ⊂ AI, and NLP intersects with both ML and DL.
10. **False** — Extractive QA locates and extracts the answer from the text; it does not generate from scratch.

---

## Quiz Lecture 2

1. **False** — `re.match()` checks only at the beginning; `re.search()` searches anywhere.
2. **C) `*`** — The star matches zero or more occurrences.
3. **True** — `[a-zA-Z]+` matches sequences of letters, splitting on digits.
4. **False** — The lifecycle is iterative; you can loop back from evaluation to data collection or preprocessing.
5. **B)** — Stop words are high-frequency words that typically do not convey meaningful content.
6. **False** — SpaCy does not offer stemming; it only provides lemmatization.
7. **B)** — Stemming uses rule-based suffix stripping (may produce invalid words); lemmatization uses dictionary lookup (always valid).
8. **True** — NER identifies people, places, organizations, phone numbers, emails, etc.
9. **True** — `#[A-Z]\d+` matches `#A123` and `#B456`.
10. **C) Tokenization** — The preprocessing pipeline is: tokenization → noise removal → normalization.

---

## Quiz Lecture 3

1. **True** — Each word gets a binary vector with one 1 and the rest 0s; dimension = vocabulary size.
2. **B) 0.052** — TF = 8/200 = 0.04, IDF = log₁₀(500/25) = log₁₀(20) ≈ 1.301, TF-IDF = 0.04 × 1.301 ≈ 0.052.
3. **False** — BoW ignores word order; both sentences have the same words, so they produce identical vectors.
4. **C) 5** — The standard edit distance between "intention" and "execution" is 5 operations.
5. **False** — Cosine similarity measures the angle; Euclidean distance measures straight-line distance. The statement reverses them.
6. **B) 2** — (1)(2) + (3)(0) + (0)(4) = 2.
7. **True** — `ngram_range=(2, 3)` extracts bigrams and trigrams only, no unigrams.
8. **C)** — All frequency-based techniques fail to capture semantic meaning between words.
9. **False** — TF-IDF gives *lower* weight to words that appear across many documents (high IDF = rare = higher weight).
10. **B)** — Cosine similarity of 0 means the vectors are perpendicular, indicating no similarity.

---

## Quiz Lecture 4

1. **True** — The distributional hypothesis: words in similar contexts have similar meanings.
2. **False** — CBOW predicts the target (center) word from context (surrounding) words. The statement describes Skip-gram.
3. **C) FastText** — FastText uses character n-grams to generate approximate vectors for OOV words.
4. **False** — A hypernym is the *general* term. "Animal" is a hypernym of "dog," not the other way around.
5. **B) king - man + woman = queen** — The classic word analogy equation.
6. **True** — GloVe combines prediction-based learning with global co-occurrence statistics.
7. **False** — `sg=0` selects CBOW; `sg=1` selects Skip-gram.
8. **A) `app`** — Valid trigrams for "apple": `<ap`, `app`, `ppl`, `ple`, `le>`.
9. **True** — SGNS maximizes similarity with positive samples and minimizes with negative samples.
10. **C)** — Static embeddings assign one fixed vector per word regardless of context in a sentence.

---

## Quiz Lecture 5

1. **True** — N-gram models predict the next word by counting how often word patterns co-occur.
2. **B) 0.4** — P("books" | "opened their") = 200/500 = 0.4.
3. **False** — Feedforward neural network LMs still require a fixed window size for fixed-length input.
4. **B)** — RNNs process input sequentially and propagate information through a hidden state.
5. **False** — Vanishing gradient means gradients become very *small* (not large), causing the model to stop learning.
6. **C) Forget gate** — The forget gate decides what information to erase from the previous cell state.
7. **False** — It is reversed: cell state = long-term memory, hidden state = short-term memory.
8. **True** — Low perplexity means the model predicts well (less confused).
9. **B)** — One-hot encoding → Embedding lookup → Hidden layer → Softmax.
10. **True** — GRU combines forget+input gates into one update gate and merges cell/hidden state.

---

## Quiz Lecture 6

1. **True** — The encoder produces a context vector; the decoder uses it to generate the output.
2. **B)** — The encoder must compress everything into one fixed-length vector, which is hard for long sequences.
3. **False** — The forward and backward LSTMs have separate weights.
4. **B)** — Attention lets the decoder look at all encoder hidden states and focus on relevant parts at each step.
5. **False** — The first part (f_t ⊙ c_{t-1}) *filters old info*; the second part (i_t ⊙ ĉ_t) *adds new info*. The statement reverses them.
6. **C)** — GRU merges forget+input into an update gate and combines cell/hidden state.
7. **True** — Softmax converts attention scores into a probability distribution over encoder states.
8. **D) Many-to-many** — Translation takes a sequence in and produces a sequence out.
9. **True** — BPTT is time-dependent; each step's gradient depends on subsequent steps.
10. **False** — The Transformer relies entirely on self-attention and does *not* use recurrence.
