# CST8507: Natural Language Processing

## Week #10 — BERT & Introduction to Question/Answering

Developed by Hala Own, Ph.D.
Algonquin College

---

## Slide 2: Lesson Agenda

- BERT Architecture
  - BERT Variants
  - BERT for Text Classification
- What is Question Answering?
- Extractive Question Answering (Reading Comprehension)
- Open Domain Question Answering
- Closed Domain Question Answering
- Generative Question Answering

---

## Slide 3: The Transformer Tree of Life

**Transformer** (Encoder + Decoder)

- **Encoder-only branch (BERT family):**
  - BERT → DistilBERT
  - RoBERTa
  - XLM → XLM-R
  - ALBERT
  - ELECTRA
  - DeBERTa
- **Encoder-Decoder branch:**
  - T5
  - BART
  - M2M-100
  - BigBird
- **Decoder-only branch (GPT family):**
  - GPT
  - GPT-2 → CTRL
  - GPT-3
  - GPT-Neo → GPT-J

*Source: Natural Language Processing with Transformers, O'Reilly Media, Inc, 2022*

---

## Slide 4: Transformer TimeLine

*"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" — Jacob Devlin et al.*

- **2017 JUN** — Transformers
- **2018 JUN** — GPT
- **2018 OCT** — BERT
- **2019 FEB** — GPT-2
- **2019 OCT** — T5
- **2020 MAY** — GPT-3
- **2021 SEP** — FLAN
- **2022 MAR** — GPT-3.5 / InstructGPT
- **2022 NOV** — ChatGPT
- **2023 FEB** — LLaMA
- **2023 MAR** — GPT-4
- **2024 MAR** — GPT-4o
- **2024 APR** — LLaMA-3.1 405B
- **2024 DEC** — OpenAI-o1, DeepSeek-V3
- **2025 JAN** — DeepSeek-R1

*Image source: https://medium.com/@lmpo/a-brief-history-of-lmms-from-transformers-2017-to-deepseek-r1-2025-dae75dd3f59a*

---

## Slide 5: Bidirectional Encoder Representations from Transformers (BERT)

Evolution layers (bottom to top): RNN / LSTM → Encoder-Decoder / Bi-LSTM → Attention → Transformer → BERT

BERT was trained on:

- **Data:** English Wikipedia, around 2.5 billion words; Book Corpus (11,000 books, around 800 million words).

---

## Slide 6: BERT: Google Search

Context example:
- "We went to the river bank."
- "I need to go to bank to make a deposit."

**Example search query:** "2019 brazil traveler to usa need a visa"

- **Before:** Result about U.S. citizens traveling to Brazil without a visa (Washington Post, 2019/03/21).
- **After:** Result about Tourism & Visitor | U.S. Embassy & Consulates in Brazil — "In general, tourists traveling to the United States require valid B-2 visas…"

*Source: http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/*

---

## Slide 7: BERT: Google Search…

**Example search query:** "parking on a hill with no curb"

- **Before:** "Parking on a Hill. Uphill: When headed uphill at a curb, turn the front wheels away from the curb and let your vehicle roll backwards slowly until the rear part of the front wheel rests against the curb using it as a block. Downhill: When you stop your car headed downhill, turn your front wheels toward the curb." — Parking on a Hill - DriversEd.com
- **After:** "For either uphill or downhill parking, if there is no curb, turn the wheels toward the side of the road so the car will roll away from the center of the road if the brakes fail. When you park on a sloping driveway, turn the wheels so that the car will not roll into the street if the brakes fail." — Parking on a Hill

*Source: http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/*

---

## Slide 8: BERT Architecture

- **BERT-base:** 12 layers, 768 hidden size, 12 attention heads, 110M parameters
- **BERT-large:** 24 layers, 1024 hidden size, 16 attention heads, 340M parameters

---

## Slide 9: BERT Architecture …

Architecture components:
- Inputs → WordPiece + MLM/NSP → Input Embedding
- Positional Encoding + Segment Encoding
- N × (Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm)
- Two output heads:
  - **NSP Output Probabilities:** Linear → Softmax
  - **MLM Output Probabilities:** Linear → Softmax

---

## Slide 10: Model Input

BERT stack of 12 encoders processes input tokens:
- Position 1: `[CLS]`
- Position 2: Help
- Position 3: Prince
- Position 4: Mayuko
- ...
- Position 512

---

## Slide 11: Input Processing

### WordPiece algorithm
- Tokenizer trained on a training set beforehand
- Vocabulary size: ~30,500

### Adding new Tokens
- Add `[CLS]` token at the beginning of the input
- Separate consecutive segments with the `[SEP]` token and put another one at the end
- Use `[MASK]` to mask inputs

---

## Slide 12: BERT: Tokenization

Using DistilBertTokenizer on: "a visually stunning rumination on love"

1. **Break words into tokens:** `a | visually | stunning | rum | ##ination | on | love`
2. **Add [CLS] and [SEP] tokens:** `[CLS] | a | visually | stunning | rum | ##ination | on | love | [SEP]`
3. **Substitute tokens with their ids:** `101 | 1037 | 17453 | 14726 | 19379 | 12758 | 2006 | 2293 | 102`

*Source: http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/*

---

## Slide 13: Token embedding

Special token IDs:
- `[PAD]` → 0
- `[UNK]` → 100
- `[CLS]` → 101
- `[SEP]` → 102
- `[MASK]` → 103

---

## Slide 14: BERT: Input Embedding

Input: `[CLS] my dog is cute [SEP] he likes play ##ing [SEP]`

Each input position is the sum of three embeddings:

- **Token Embeddings:** E_[CLS], E_my, E_dog, E_is, E_cute, E_[SEP], E_he, E_likes, E_play, E_##ing, E_[SEP]
- **Segment Embeddings:** E_A, E_A, E_A, E_A, E_A, E_A, E_B, E_B, E_B, E_B, E_B
- **Position Embeddings:** E_0, E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8, E_9, E_10

*Image adapted from Jacob Devlin, Stanford CS224N*

---

## Slide 15: Model Outputs

The `[CLS]` token after processing is used as the **representation for the entire input sentence**.

BERT: 12 encoder layers process input tokens from position 1 `[CLS]` through position 512.

---

## Slide 16: How was BERT Trained

- It happens in two main phases: **pre-training** and **fine-tuning**.
- In **pre-training** phase, BERT learns from a massive amount of text data **without any specific task** in mind.
- In **fine-tuning** phase, we take the pre-trained BERT model and add a small layer on top tailored to a specific task:
  - Masked Language Modelling (MLM)
  - Next Sentence Prediction (NSP)

Pretraining uses large data → Base Model. Fine-tuning uses small data → Fine-tuned Model.

---

## Slide 17: How was BERT Trained: Masked LM (MLM)

BERT is an Encoder-Only Transformer stacking 12 Transformer Encoder Blocks.

Input example: `[CLS] Deep [MASK] is very Power [SEP]`
Output tokens: `[CLS] Deep Learning is very Power [SEP]` (predicting the masked "Learning")

Each encoder contains:
- Layer Norm → Multi-Head Self-Attention (Q, K, V) → Add
- Layer Norm → Feed-Forward → Add

Positional encoding is added to token embeddings before entering the first encoder block.

*Source: https://medium.com/@lmpo/bert-unleashed-the-model-that-redefined-language-understanding-afecf5545295; https://arxiv.org/abs/1810.04805*

---

## Slide 18: Masked LM (MLM)

**15% of the words to predict.**

- **80% of the time, replace with [MASK]**
  `went to the store → went to the [MASK]`
- **10% of the time, replace random word**
  `went to the store → went to the running`
- **10% of the time, keep same**
  `went to the store → went to the store`

---

## Slide 19: How was BERT Trained: Masked LM (MLM)

- Token + Positional Embeddings → Bidirectional Transformer Encoder → hidden states z_1..z_8
- For each masked position, an **LM Head with Softmax over Vocabulary** predicts the original token.
- **CE Loss** is computed only at masked positions, e.g., `−log y_long`, `−log y_thanks`, `−log y_the`.

Example:
- Input: `So [mask] and [mask] for all apricot fish`
- Target: `So long and thanks for all the fish`

---

## Slide 20: How was BERT Trained: Two-sentence Tasks

- Input tokens: `[CLS] Cancel my flight [SEP] And the hotel [SEP]`
- Each token receives Token + Segment + Positional Embeddings.
- `h_CLS` (output of `[CLS]`) passes through an **NSP Head** with weights `W_NSP`.
- Cross-entropy loss `−log y_1` is computed against the IsNext/NotNext label.

---

## Slide 21: How was BERT Trained…

Predict likelihood that sentence B belongs after sentence A (FFNN + Softmax over `IsNext` / `NotNext`).

Example prediction: 1% IsNext, 99% NotNext.

Input format:
- `[CLS] the man [MASK] to the store [SEP] penguin [MASK] are flightless birds [SEP]`
- Sentence A and Sentence B are separated by `[SEP]`.

**Examples:**
- Sentence A = "The man went to the store." / Sentence B = "He bought a gallon of milk." → Label = `IsNextSentence`
- Sentence A = "The man went to the store." / Sentence B = "Penguins are flightless." → Label = `NotNextSentence`

*Source: http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/*

---

## Slide 22: BERT pre-training: Putting Together

During pre-training, BERT jointly predicts:
- **NSP** (from `[CLS]` output representation `C`)
- **Mask LM** (from masked token outputs `T_1 … T_N`, `T_1' … T_M'`)

Inputs: Unlabeled Sentence A and B pair → embeddings `E_[CLS], E_1, …, E_N, E_[SEP], E_1', …, E_M'` → BERT → `C, T_1, …, T_N, T_[SEP], T_1', …, T_M'`.

**Total Loss = MLM Loss + NSP Loss**

*Image adapted from Jacob Devlin, Stanford CS224N*

---

## Slide 23: Transfer Learning: Quick Overview

1. **Model architecture with random weight** (No knowledge of language)
2. → *Training* → **Pretrained Model** (Very good understanding of language)
3. → *Fine-Tune* → **Fine-tuned model** (Different NLP Tasks)

---

## Slide 24: BERT: Transfer Learning

1. **Pretrain on large dataset** (Wikipedia): BERT with objective of masked language modeling.
2. **Fine-tune for downstream task:** Classification, Named Entity Recognition, Paraphrase Identification.

*Image source: Hands-on Large Language Models, Book*

---

## Slide 25: BERT for Text Classification

Input: `[CLS] I like to draw [SEP] [PAD] [PAD]` with attention mask `1 1 1 1 1 1 0 0`.

- Pad sentence to MAX_LEN; use "Attention Mask" to ignore pads.
- Stack of Transformer Layers 1..12.
- Only the `[CLS]` output vector is passed to the Classifier → Prediction (other token outputs are discarded).

*Source: Natural Language Processing with Transformers, O'Reilly Media, Inc, 2022*

---

## Slide 26: DEMO

*(Demo slide — no content)*

---

## Slide 27: List of the released pre-trained BERT models

| Model | Details |
|-------|---------|
| BERT-Base, Uncased | 12-layer, 768-hidden, 12-heads, 110M parameters |
| BERT-Large, Uncased | 24-layer, 1024-hidden, 16-heads, 340M parameters |
| BERT-Base, Cased | 12-layer, 768-hidden, 12-heads, 110M parameters |
| BERT-Large, Cased | 24-layer, 1024-hidden, 16-heads, 340M parameters |
| BERT-Base, Multilingual Cased (New) | 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters |
| BERT-Base, Multilingual Cased (Old) | 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters |

---

## Slide 28: BERT Variants

**Exploring Variants of BERT:**
- ALBERT
- RoBERTa
- ELECTRA
- DistilBERT — *achieves 97% of BERT's performance while using 40% less memory and being 60% faster*
- SpanBERT
- TinyBERT

*Source: https://www.scaler.com/topics/nlp/bert-variants/*

---

## Slide 29: Q/A System

*(Section title slide)*

---

## Slide 30: Taxonomy of Q/A System

Question (Q) → [System] → Answer (A)

- **Information source**
- **Question types**
- **Answer type**

---

## Slide 31: Information Source

- **Structured Data Sources**
  - Databases (SQL, NoSQL)
- **Unstructured Text Sources**
  - Web Documents & Articles (e.g., Wikipedia, news websites), Research Papers & Scientific Literature (e.g., arXiv, PubMed): Critical for academic and medical Q/A systems.
  - Product Manuals & Documentation.
  - Books & Digital Libraries (e.g., Project Gutenberg, Google Books)
- **Conversational Data Sources**
  - Customer Support Logs & Chat Transcripts
  - Community Forums & Q/A Sites (e.g., Stack Overflow, Quora)
  - Social Media Feeds (e.g., X): For real-time Q/A on trends, events, and opinions.

---

## Slide 32: Types of Questions in Modern Systems

- Factoid Questions
- Open domain Questions
- Closed domain Questions
- Complex (narrative) Questions

---

## Slide 33: Answer type

- **Extractive Answers** (Span-Based Answers)
- **Abstractive (Generative) Answers**
- **Factoid Answers** (Knowledge-based) — The system provides short factual answers, such as names, dates, numbers, or locations.

---

## Slide 34: Question Answering Paradigms

- Extractive QA (SQuAD, BERT-based models)
- Knowledge-based QA
- Hybrid approaches QA
- Generative QA
- Retrieval-Augmented QA (RAG)

---

## Slide 35: 2011: IBM Watson beat Jeopardy champions

Image: IBM Watson Jeopardy match scoreboard showing three contestants with final totals of $24,000, $77,147, and $21,600. Final Jeopardy responses all answered "Who is (Bram) Stoker?".

---

## Slide 36: IBM Watson architecture

*Won Jeopardy on February 16, 2011!*

Pipeline:
1. **Question Processing** — Focus Detection, Lexical Answer Type Detection, Question Classification, Parsing, Named Entity Tagging, Relation Extraction, Coreference.
2. **Candidate Answer Generation**
   - *From Text Resources:* Document and Passage Retrieval → passages → Answer Extraction (Document titles, Anchor text) → Candidate Answers.
   - *From Structured Data:* Relation Retrieval from DBPedia, Freebase → Candidate Answers.
3. **Candidate Answer Scoring** — Evidence Retrieval and scoring using Text Evidence Sources, Time from DBPedia, Answer Type, Space from Facebook → Candidate Answer + Confidence.
4. **Confidence Merging and Ranking** — Merge Equivalent Answers → Logistic Regression Answer Ranker → Answer and Confidence.

*Slide credit: Dan Jurafsky*

---

## Slide 37: Reading Comprehension (extractive QA)

Reading comprehension = comprehend a passage of text and answer questions about its content (P, Q) → A

**Passage:**
> Tesla was the fourth of five children. He had an older brother named Dane and three sisters, Milka, Angelina and Marica. Dane was killed in a horse-riding accident when Nikola was five. In 1861, Tesla attended the "Lower" or "Primary" School in Smiljan where he studied German, arithmetic, and religion. In 1862, the Tesla family moved to Gospić, Austrian Empire, where Tesla's father worked as a pastor. Nikola completed "Lower" or "Primary" School, followed by the "Lower Real Gymnasium" or "Normal School."

**Q:** What language did Tesla study while in school?
**A:** German

*Slide credit: Dan Jurafsky*

---

## Slide 38: Stanford question answering dataset (SQuAD)

- 100k annotated (passage, question, answer) triples
- Passages are selected from English Wikipedia, usually **100~150 words**.
- Questions are **crowd-sourced**.
- Each answer is a **short segment of text** (or span) in the passage.
- SQuAD remains the most popular reading comprehension dataset.

**Example passage (meteorology):**
> In meteorology, precipitation is any product of the condensation of atmospheric water vapor that falls under **gravity**. The main forms of precipitation include drizzle, rain, sleet, snow, **graupel** and hail… Precipitation forms as smaller droplets coalesce via collision with other rain drops or ice crystals **within a cloud**. Short, intense periods of rain in scattered locations are called "showers".

**Q1:** What causes precipitation to fall? → **gravity**
**Q2:** What is another main form of precipitation besides drizzle, rain, snow, sleet and hail? → **graupel**
**Q3:** Where do water droplets collide with ice crystals to form precipitation? → **within a cloud**

*Slide credit: Dan Jurafsky*

---

## Slide 39: Stanford Question Answering dataset (SQuAD)

**Evaluation:** Exact Match EM (0 or 1) and F1 (partial credit).

- For development and testing sets, **3 gold answers are collected.**
- Compare the predicted answer to each gold answer and take **max scores**.
- Take the **average** of all the examples for both exact match and F1.

**Q:** What did Tesla do in December 1878?
**Answer:** {`left Graz`, `left Graz ans`, `left Graz and severed all relations with his family`}
**Prediction:** {`left Graz and severed`}

- Exact Match: max{0, 0, 0} = 0
- F1: max{0.67, 0.67, 0.61} = 0.67

*Slide credit: Dan Jurafsky*

---

## Slide 40: Neural Models For Reading Comprehension

**Problem formulation**
- Input: `C = (c_1, c_2, …, c_N)`, `Q = (q_1, q_2, …, q_M)`, `c_i, q_i ∈ V`
- Output: `1 ≤ start ≤ end ≤ N`

*M < N; answer is a span in the passage*

- A family of **LSTM-based models** with attention (2016–2018)
- Fine-tuning **BERT-like models** for reading comprehension (2019+)

*Slide credit: Dan Jurafsky*

---

## Slide 41: BiDAF: the Bidirectional Attention Flow model

Exact Match (EM): 71.3%; F1 score: 81.2%

Layer stack (bottom to top):
- **Encoding**
  - Character Embed Layer
  - Word Embed Layer
  - Phrase Embed Layer (LSTM over context tokens x_1..x_T and query tokens q_1..q_J)
- **Attention**
  - Attention Flow Layer: Query2Context and Context2Query Attention → g_1, g_2, …, g_T
- **Modeling**
  - Modeling Layer: LSTM over attention outputs → m_1, m_2, …, m_T
  - Output Layer: Dense + Softmax (Start) and LSTM + Softmax (End)

---

## Slide 42: BERT for Reading Comprehension…

- **Question** = Segment A
- **Passage** = Segment B
- **Answer** = predicting two endpoints in segment B

Input = `[CLS] Question tokens [SEP] Reference tokens` with Segment Embeddings A for question, B for reference.

**Example**
- **Question:** How many parameters does BERT-large have?
- **Reference Text:** BERT-large is really big… it has 24 layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance.

**Loss:**
- `L = −log p_start(s*) − log p_end(e*)`
- `p_start(i) = softmax_i(w_start^T · H)`
- `p_end(i) = softmax_i(w_end^T · H)`
- where `H = [h_1, h_2, …, h_N]` are the hidden vectors of the paragraph returned by BERT.

*Image credit: https://mccormickml.com/ ; Slide credit: Dan Jurafsky*

---

## Slide 43: BERT for Reading Comprehension: Predict start

- Stack of Transformer Layers 1..12 produces per-token hidden vectors (length 768).
- A single **start** weight vector (length 768) is dot-producted with **every** token's vector.
- Softmax over all positions yields a distribution over possible start indices.

Example tokens: `… BERT large has 340 M params total ! …`
The highest softmax probability identifies the predicted start token.

*Image credit: Chris McCormick*

---

## Slide 44: BERT for Reading Comprehension: Predict end

Analogous to the start prediction:
- A separate **end** weight vector is dot-producted with every token's hidden vector.
- Softmax yields a distribution over possible end indices.

Example tokens: `… BERT large has 340 M params total ! …`
The highest softmax probability identifies the predicted end token.

*Image credit: Chris McCormick*

---

## Slide 45: Comparisons between BiDAF and BERT models on SQuAD 2.0

| Model | F1 | EM |
|-------|------|------|
| BiDAF | 77.3 | 67.7 |
| BERT-base | 88.5 | 80.8 |
| BERT-large | 90.9 | 84.1 |
| XLNet | 94.5 | 89.0 |
| RoBERTa | 94.6 | 88.9 |
| ALBERT | 94.8 | 89.3 |

*Source: https://rajpurkar.github.io/SQuAD-explorer/*

---

## Slide 46: Dealing With Long Passages

Example question: *"Why is the camera of poor quality?"*
Passage: *"Item like the picture, fast deliver 3 days well packed, good quality for the price. The camera is decent (as phone cameras go). There is no flash though…"*

- The question and passage are combined as `[CLS] question [SEP] passage [SEP]`.
- For long passages, apply a **stride** to generate multiple overlapping windows so each segment still contains the question plus a chunk of the passage.

*Source: Natural Language Processing with Transformers, O'Reilly, 2021*

---

## Slide 47: Tokenizing Questions and Contexts for BERT-based Question Answering

```python
inputs = tokenizer(
    examples["question"],
    examples["context"],
    max_length=500,
    truncation="only_second",
    stride=25,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
    return_tensors="pt",
    return_attention_mask=True,
    add_special_tokens=True
)
```

---

## Slide 48: DEMO

*(Demo slide — no content)*

---

## Slide 49: Open Datasets available for Question Answering

- **S**tanford **Qu**estion **A**nswering **D**ataset (SQuAD)
- **WikiQA** dataset
- The **TREC-QA** dataset
- **NewsQA** dataset
- Google (NQ) dataset

---

## Slide 50: Retriever-Reader Architecture

Flow:
- User question (e.g., "When did Marie Curie win her first Nobel Prize?")
- → **Retriever** pulls relevant documents from unstructured documents (Wikipedia, internet, …)
- → *Document postprocessing* → Relevant documents
- → **Reader** extracts the answer
- → *Answer postprocessing* → returns answer (e.g., "A: 1903") to the user.

*Source: Natural Language Processing with Transformers, O'Reilly, 2021*

---

## Slide 51: Retrieval document stores

| Retriever | In memory | Elasticsearch | FAISS | Milvus |
|-----------|-----------|---------------|-------|--------|
| TF-IDF | Yes | Yes | No | No |
| BM25 | No | Yes | No | No |
| Embedding | Yes | Yes | Yes | Yes |
| DPR | Yes | Yes | Yes | Yes |

---

## Slide 52: Embeddings in Information Retrieval

- Word2Vec
- GloVe
- BERT

---

## Slide 53: Dense Passage Retrieval (DPR)

- **Train a separate encoder** for both queries (questions) and passages (documents) to optimize their embeddings for retrieval tasks.
  - Dual Encoder Architecture
  - End-to-End Training

---

## Slide 54: Dense Passage Retrieval (DPR)

Dual-encoder pipeline:
- **Question Encoder** converts the query (e.g., "Why is the camera of poor quality?") → Question vector
- **Passage Encoder** converts each candidate passage (e.g., "Item like the picture… there is no flash…") → Passage vector
- **Dot Product Similarity** between the two vectors → Document Score

*Hugging Face; Source: Natural Language Processing with Transformers, O'Reilly, 2021*

---

## Slide 55: Haystack library

- Developed by **deepset**
- Based on the retriever-reader architecture
- Abstracts much of the complexity
- Integrates tightly with Transformers
  - Document store
  - Pipeline

---

## Slide 56: DEMO

*(Demo slide — no content)*

---

## Slide 57: Other frameworks similar to Haystack

- DeepPavlov
- DrQA

---

## Slide 58: Evaluating the Reader

- *Exact Match (EM)*
- *F1-score*

---

## Slide 59: Going Beyond Extractive QA

- Retrieval-augmented Generation (**RAG**)
  - Based on LLM

---

## Slide 60: Q&A

*(Closing slide — meme: "The smartest people are those who ask questions.")*
