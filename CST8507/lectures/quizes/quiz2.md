# NLP - Quiz 2

### Question 1

The process of converting raw text into a sequence of units that a model can process.

- A) Lemmatization
- B) Stemming
- C) Tokenization

---

### Question 2

Text cleaning removes noise (like special characters, irrelevant symbols, and unnecessary spaces) and standardizes the text (e.g., converting to lowercase), is essential for improving the quality of the data and the performance of NLP models.

- True
- False

---

### Question 3

SpaCy does not provide a built-in function for Stemming

- True
- False

---

### Question 4

When might you use lemmatizing over stemming?

- A) When non-dictionary words are allowed to appear in the output
- B) When the data file contains a large number of simple words
- C) When speed is preferred more than accuracy
- D) When accuracy is preferred more than speed

---

### Question 5

The following rgx will match all the words ended with a hyphen(-):

`rgx = r'\b\w+[-]\w+\b'`

- True
- False

---

### Question 6

Consider you have the following list that represents the USA's state names:

```python
states = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado',
          'Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho',
          'Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana',
          'Maine','Maryland','Massachusetts','Michigan','Minnesota',
          'Mississippi','Missouri','Montana','Nebraska','Nevada',
          'New Hampshire','New Jersey','New Mexico','New York',
          'North Carolina','North Dakota','Ohio','Oklahoma','Oregon',
          'Pennsylvania','Rhode Island','South Carolina','South Dakota',
          'Tennessee','Texas','Utah','Vermont','Virginia','Washington',
          'West Virginia','Wisconsin','Wyoming']
```

Which python expression outputs which state names start and end with a "vowel" character?

- A) `[s for s in states if s[1].lower() in 'aeiou' and s[-1] in 'aeiou']`
- B) `[s for s in states if s[0].lower() in 'aeiou' and s[-1] in 'aeiou']`
- C) `[s for s in states if s[0].lower() in 'aeiou' and s[1] in 'aeiou']`

---

### Question 7

Consider the provided code snippet:

```python
Text='I love NLP and I am read9y to study in 5 hours per Day'
regex='[a-zA-Z]\w*d+'

print(re.findall(regex, Text))
```

The output is: `[and]`

- True
- False

---

### Question 8

For which of the following tasks we shouldn't do stemming/lemmatization?

- A) Text Classification
- B) Poetry Analysis
- C) Sentiment Analysis

---

### Question 9

Pick the stemming action

- A) was, am, is, are → be
- B) helped, helps → help
- C) troubled, troubling, trouble → trouble
