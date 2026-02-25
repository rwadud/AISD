# NLP - Quiz Answers

## Quiz 1

1. **D)** NLP is a subset of AI focused on language tasks; ML is an approach to achieve AI; DL is a type of ML.
2. **D) Text Classification** — It is NLU (understanding), not NLG (generation).
3. **False** — Variation refers to multiple ways of expressing the same meaning, not frequency skew (that's Zipf's Law).
4. **False** — AI includes learning-based approaches; ML uses data to make predictions and is a key method within AI.
5. **True** — Documents are raw/semi-structured text; knowledge is structured, interpreted information (facts, entities, relationships).
6. **True** — NLP develops algorithms to understand, interpret, and generate human language.
7. **True** — Correct description of the Turing Test: a human evaluator cannot reliably distinguish the machine from a human.
8. **False** — The statement contradicts itself; summarization must maintain overall meaning and coherence.
9. **True** — Sentiment analysis requires deep understanding of human emotions, language, and context.
10. **B) Transformer** — Introduced in 2017 with "Attention Is All You Need."

---

## Quiz 2 

1. **C) Tokenization** — Converting raw text into a sequence of units a model can process.
2. **True** — Text cleaning removes noise (special characters, irrelevant symbols) and standardizes text (e.g., lowercasing).
3. **True** — SpaCy does not provide built-in stemming; it only offers lemmatization.
4. **D)** When accuracy is preferred more than speed — Lemmatization produces valid dictionary words unlike stemming.
5. **False** — The regex `r'\b\w+[-]\w+\b'` matches hyphenated words (e.g., "well-known"), not words ending with a hyphen.
6. **B)** `[s for s in states if s[0].lower() in 'aeiou' and s[-1] in 'aeiou']` — Checks first (`s[0]`) and last (`s[-1]`) characters.
7. **False** — The regex `[a-zA-Z]\w*d+` matches substrings ending in 'd'; output would include `['and', 'read', 'stud']`, not just `['and']`.
8. **B) Poetry Analysis** — Word forms carry meaning in poetry, so stemming/lemmatization should be avoided.
9. **B)** helped, helps → help — Stemming strips suffixes via rules. The first option (was, am, is → be) is lemmatization.

---

## Quiz 3 (7/8)

1. **C) 0.8421** — dot = 0.06+0.08+0.24+0.35 = 0.73; |w1| = sqrt(0.66) ≈ 0.8124; |w2| = sqrt(1.14) ≈ 1.0677; cosine = 0.73/0.8674 ≈ 0.8421.
2. **False** — Bag of Words ignores word order; each word is treated independently.
3. **True** — Cosine similarity close to 1 means the word vectors are highly similar in meaning.
4. **C) 5** — The minimum edit distance between "intention" and "execution" is 5 operations.
5. **C)** It does not consider the context and semantic relationships between words — A key limitation of TF-IDF.
6. **False** — Correct ascending TF*IDF order is [d2, d1000, d3, d9, d1]; the proposed order swaps d1 and d9.
7. **False** — The vocabulary from `ngram_range=(1,2)` has 10 features, not 8; the output array dimension and values are wrong.
8. **True** — IDF is calculated by dividing the total number of documents by the number of documents containing the word.

---

## Quiz 4

1. **True** — Self-supervision uses surrounding words as implicit training data, avoiding hand-labeled supervision.
2. **True** — TF-IDF's high-dimensional nature makes it difficult to use efficiently for deep learning-based NLP.
3. **True** — Word2Vec consists of two techniques: CBOW (Continuous Bag of Words) and Skip-gram.
4. **False** — Most modern NLP algorithms DO use embeddings as the representation of word meaning.
5. **True** — GloVe is global (considers entire corpus) and local (co-occurrence within a limited context window).
6. **A) 100** — The default dimensionality of word embeddings in Gensim Word2Vec.
7. **False** — Embedding vectors do not need to match vocabulary size; lower dimensions capture meaning more efficiently.
8. **False** — Skip-Gram predicts surrounding context words from the center word (the reverse is CBOW).
9. **B)** e_boy - e_girl ≈ e_brother - e_sister — The gender relationship is consistent across word pairs.

---

## Quiz 5

1. **True** — Publicly available datasets (news, social media, web pages) provide diverse language data for training NLP models.
2. **False** — N-gram models predict the next word using statistical co-occurrence counts, not semantic understanding or deep reasoning.
3. **False** — A low learning rate makes training slower, not faster; small weight updates mean slower convergence.
4. **True** — "Stateful computation" in RNNs refers to maintaining internal memory states across multiple inputs.
5. **B) 0.4** — P("happy" | "feel") = Count("feel happy") / Count("feel") = 40/100 = 0.4.
6. **True** — In RNNs, gradients are the rate of change of the loss w.r.t. parameters, computed via backpropagation.
7. **B)** RNNs maintain an internal state that allows them to model sequential dependencies, crucial for language modeling and translation.
8. **C)** Because LSTMs use gating mechanisms that regulate information flow and help preserve gradients over long sequences.
