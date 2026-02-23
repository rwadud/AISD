# NLP - Quiz Lecture 2

### Question 1

The `re.match()` function searches for a pattern anywhere in the string, while `re.search()` only checks at the beginning.

- True
- False

---

### Question 2

Which of the following metacharacters matches zero or more occurrences of the preceding element?

- A) `+`
- B) `?`
- C) `*`
- D) `.`

---

### Question 3

Consider the following code snippet:

```python
import re

text = "abc123def456ghi"
result = re.findall(r'[a-zA-Z]+', text)
print(result)
```

The output is: `['abc', 'def', 'ghi']`

- True
- False

---

### Question 4

The NLP development lifecycle is strictly a linear process: once you move from evaluation to deployment, you should never go back to data collection or preprocessing.

- True
- False

---

### Question 5

Which of the following is a correct reason to remove stop words during preprocessing?

- A) Stop words are always misspelled and introduce noise
- B) Stop words are high-frequency words that typically do not convey meaningful content
- C) Stop words increase model accuracy and should never be removed
- D) Stop words only appear in spoken language and not in written text

---

### Question 6

SpaCy provides both stemming and lemmatization as built-in functions.

- True
- False

---

### Question 7

What is the key difference between stemming and lemmatization?

- A) Stemming uses dictionary lookup while lemmatization applies rule-based suffix stripping
- B) Stemming applies rule-based suffix stripping and may produce invalid words, while lemmatization uses dictionary lookup and always produces valid words
- C) Both methods always produce valid words, but stemming is slower
- D) Lemmatization removes the entire word and replaces it with a numeric ID

---

### Question 8

Named Entity Recognition (NER) identifies and tags entities in text such as people, places, organizations, and phone numbers.

- True
- False

---

### Question 9

Consider the following code snippet:

```python
import re

text = "Order #A123 and #B456 today"
result = re.findall(r'#[A-Z]\d+', text)
print(result)
```

The output is: `['#A123', '#B456']`

- True
- False

---

### Question 10

In the text preprocessing pipeline, which step should be performed first?

- A) Normalization (stemming/lemmatization)
- B) Noise removal
- C) Tokenization
- D) Feature extraction
