# NLP - Quiz Lecture 4

### Question 1

The distributional hypothesis states that words appearing in similar contexts tend to have similar meanings.

- True
- False

---

### Question 2

In the CBOW (Continuous Bag of Words) model, the input is the target (center) word and the output is the surrounding context words.

- True
- False

---

### Question 3

Which of the following word embedding techniques can generate approximate vector representations for out-of-vocabulary (OOV) words?

- A) Word2Vec
- B) GloVe
- C) FastText
- D) TF-IDF

---

### Question 4

In WordNet, a hypernym is a specific term within a broader category. For example, "dog" is a hypernym of "animal."

- True
- False

---

### Question 5

Which of the following word analogy equations would you expect a well-trained word embedding to satisfy?

- A) king - queen = woman - man
- B) king - man + woman = queen
- C) king + man - woman = queen
- D) king - woman + man = queen

---

### Question 6

GloVe differs from Word2Vec primarily because it combines prediction-based learning with statistical co-occurrence information from the entire corpus.

- True
- False

---

### Question 7

When training a Word2Vec model using Gensim, setting the parameter `sg=0` selects the Skip-gram algorithm.

- True
- False

---

### Question 8

FastText represents each word as the sum of its character n-gram vectors. For the word "apple" with trigrams, which of the following is a valid character n-gram?

- A) `app`
- B) `apple`
- C) `ap`
- D) `pplee`

---

### Question 9

In Skip-gram with Negative Sampling (SGNS), the goal is to maximize the similarity between the target word and its positive context samples while minimizing the similarity with randomly sampled negative words.

- True
- False

---

### Question 10

Which of the following is a limitation common to all static word embedding methods (Word2Vec, GloVe, FastText)?

- A) They cannot produce dense vectors
- B) They require manually labeled training data
- C) They assign a single fixed vector per word regardless of the word's context in a sentence
- D) They cannot handle large vocabularies
