# NLP - Quiz Lecture 3

### Question 1

In a One Hot Encoding representation, each word is represented as a vector where only one element is set to 1 and all others are 0. The dimension of the vector equals the size of the vocabulary.

- True
- False

---

### Question 2

Given a corpus of 500 documents, if the word "algorithm" appears 8 times in a document that has 200 total words, and it appears in 25 documents across the corpus, what is the TF-IDF score for "algorithm" in that document?

- A) 0.04
- B) 0.052
- C) 1.3
- D) 0.4

---

### Question 3

In Bag of Words representation, the sentences "The cat chased the dog" and "The dog chased the cat" produce different vector representations.

- True
- False

---

### Question 4

What is the Levenshtein distance between the words "intention" and "execution"?

- A) 3
- B) 4
- C) 5
- D) 7

---

### Question 5

Cosine similarity measures the straight-line distance between two vectors, while Euclidean distance measures the angle between them.

- True
- False

---

### Question 6

Given two vectors w⃗₁ = (1, 3, 0) and w⃗₂ = (2, 0, 4), what is the dot product w⃗₁ · w⃗₂?

- A) 0
- B) 2
- C) 6
- D) 10

---

### Question 7

Using `CountVectorizer(ngram_range=(2, 3))` on the text "I love NLP" would extract both bigrams and trigrams but not unigrams.

- True
- False

---

### Question 8

Which of the following is a limitation shared by all frequency-based text representation techniques?

- A) They require GPUs for computation
- B) They cannot handle English text
- C) They do not capture semantic meaning between words
- D) They require labeled training data

---

### Question 9

In TF-IDF, a word that appears frequently across many documents in the corpus will receive a higher weight than a word that appears in only a few documents.

- True
- False

---

### Question 10

If the cosine similarity between two document vectors is 0, which of the following is true?

- A) The two documents are identical
- B) The two vectors are perpendicular, indicating no similarity
- C) The two documents are very similar but not identical
- D) The cosine similarity calculation failed
