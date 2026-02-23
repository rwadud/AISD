# NLP - Quiz 3

### Question 1

Suppose that you have two word vectors; w₁ and w₂ with the following TF_IDF values:

w⃗₁ = (0.2, 0.2, 0.3, 0.7) and w⃗₂ = (0.3, 0.4, 0.8, 0.5)

What is the cosine similarity between the two words:

- A) 0.932
- B) 0.748
- C) 0.8421
- D) 0.947

---

### Question 2

In a Bag of Words representation, the order of words in a document is crucial, and each word is treated as dependent on its surrounding words.

- True
- False

---

### Question 3

If the cosine similarity between the word vectors for Word A and Word B is close to 1, it means that Word A and Word B are considered highly similar in meaning.

- True
- False

---

### Question 4

Given the words "intention" and "execution," what is the minimum number of operations required to transform "intention" into "execution"?

- A) 3
- B) 6
- C) 5
- D) 7

---

### Question 5

One of the disadvantages of using TF-IDF is:

- A) It can be used in various NLP tasks such as text classification, document retrieval, and information extraction.
- B) It helps to identify the most important words in a document
- C) It does not consider the context and semantic relationships between words

---

### Question 6

Consider a corpus with 60000 documents. The word "cat" occurs in some documents, with the following frequency:

TF_d₁ = ²⁵⁄₁₂₇, TF_d₂ = ³⁄₂₅₀, TF_d₃ = ²⁰⁄₆₅₀, TF_d₉ = ¹⁵⁄₁₂₅, and TF_d₁₀₀₀ = ²⁰⁄₈₀₀

If the total number of words in the corpus is 50000, then if we arrange the documents according to the rank in the ascending order using TF × IDF for the word "cat" will result:

[d₂, d₁₀₀₀, d₃, d₁, d₉]

- True
- False

---

### Question 7

Consider the following code snippet:

```python
cv = CountVectorizer(ngram_range=(1,2)).fit(["I love NLP", "He love NLP","good man"])
cv.transform(['love']).toarray()
```

The output will be:

`array([[0, 0, 1, 0, 0, 0, 0, 0]], dtype=int64)`

- True
- False

---

### Question 8

The inverse document frequency (IDF) of a word is calculated by dividing the total number of documents by the number of documents containing the word.

- True
- False
