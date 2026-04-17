# CST8507: Natural Language Processing

**Week #11 — LLM & RAG**

Developed by Hala Own, Ph.D.
Algonquin College

---

## Lesson Agenda

- Introduction to LLM
- LLMs Components
- How ChatGPT trained
- Limitations of LLMs
- How Retrieval Augmented Generation **RAG** works
- LangChain

---

## What is LLMs

- Trained on **massive amounts of text data**.
- Transformer architecture
- Understanding and generating human-like texts.
- Capable of performing a wide range of NLP tasks with high proficiency, including text completion, summarization, ...

---

## A Timeline Of Existing Large Language Models

Publicly available models over time:

- **2019**: T5, GPT-3
- **2020**: GShard, mT5, Codex
- **2021**: PanGu-α, PLUG, Ernie 3.0, Jurassic-1, CPM-2, T0, HyperCLOVA, FLAN, Yuan 1.0, LaMDA, AlphaCode, Chinchilla, Anthropic, WebGPT, Ernie 3.0 Titan, Gopher, InstructGPT, CodeGen, GLaM, MT-NLG
- **2022**: OPT, CodeGeeX, GPT-NeoX-20B, Tk-Instruct, GLM, Cohere, AlexaTM, WeLM, BLOOM, mT0, BLOOMZ, Galactica, OPT-IML, UL2, PaLM, YaLM, Sparrow, Flan-T5, Flan-PaLM, Luminous, NLLB, ChatGPT
- **2023**: Pythia, Vicuna, PaLM2, Falcon, MOSS, PanGu-Σ, Bard, LLaMA, GPT-4, LLaMA2, InternLM, Qwen, Mistral, Deepseek, Mixtral
- **2024**: Baichuan-4, Baichuan-3, InternLM2, Qwen2, DeepSeek-V2, LLaMA3, MiniCPM, Gemma, YuLan-Chat, StarCoder, CodeGen2, ChatGLM, DeepSeek-V3, Qwen2.5, Gemma-2, YuLan-Mini

Source: marktechpost.com (Wayne Xin Zhao et al.)

---

## LLMs Components

Central concept: **LLMs component**, surrounded by:

- Transformer architecture
- Token
- Context / window length
- Parameters

---

## Comparison of Popular Large Language Models

| Model | Parameters | Size on Disk | Memory Usage (Inference) | Learning Data Size |
|---|---|---|---|---|
| **BERT (Large)** | 340M | ~1.3 GB (FP32) | ~1.5–2 GB (FP16) | 3.3B words (~16 GB) |
| GPT-4o | ~200B | ~350 GB (FP32) | ~400 GB (FP16, single GPU) | 570 GB (~300B tokens) |
| LLaMA (13B) | 13B | ~26 GB (FP32) | ~26 GB (FP16) | ~1T tokens |
| LLaMA (70B) | 70B | ~140 GB (FP32) | ~140 GB (FP16) | ~1T tokens |
| **BLOOM (176B)** | 176B | ~352 GB (FP32) | ~352 GB (FP16) | 1.6T tokens |
| Mistral 7B | 7B | ~14 GB (FP32) | ~14 GB (FP16) | ~1T tokens |
| **Mixtral 8x7B** | 56B | ~112 GB (FP32) | ~112 GB (FP16) | Unknown (large corpus) |
| Grok (xAI) | Unknown (est. ~70B) | Est. ~140 GB (FP32) | Est. ~140 GB (FP16) | Unknown (large) |
| **PaLM (540B)** | 540B | ~1 TB (FP32) | ~1 TB (FP16) | 780B tokens |

Source: Recent Survey on large language models

---

## LLM Tokenizer

https://tiktokenizer.vercel.app/

---

## Context Length

The **maximum number of tokens** an LLM can process at once.

Includes:

- prompt
- conversation history
- generated answer

---

## Context Length: Comparison of Various LLMs

| Model | Context Length |
|---|---|
| Gemma 2 | 32K Tokens |
| Mistral | 32K Tokens |
| LLaMA 3.1 | 128K Tokens |
| Claude 3.5 | 200K Tokens |
| GPT-4.1 | 1M Tokens |
| Gemini 1.5 Pro | 1–2M Tokens |

Source: Recent survey on LLM

---

## Types of large language models

- A general-purpose language model (**GPT series**)
- Instruction tuned language models (**AllenNLP's**)
- Dialogue-tuned language models (**Microsoft's DialoGPT**)
- Domain specific language models (**BioBERT**, Bidirectional Encoder Representations from Transformers for Biomedical Text Mining)

---

## How ChatGPT is Trained

### Step 1: Collect demonstration data, and train a supervised policy.

- A prompt is sampled from our prompt dataset. (e.g., "Explain the moon landing to a 6 year old")
- A labeler demonstrates the desired output behavior. (e.g., "Some people went to the moon...")
- This data is used to fine-tune GPT-3 with supervised learning (**SFT**).

### Step 2: Collect comparison data, and train a reward model.

- A prompt and several model outputs are sampled.
- A labeler ranks the outputs from best to worst (e.g., D > C > A = B).
- This data is used to train our reward model (**RM**).

### Step 3: Optimize a policy against the reward model using reinforcement learning.

- A new prompt is sampled from the dataset (e.g., "Write a story about frogs").
- The policy generates an output (**PPO**).
- The reward model calculates a reward for the output.
- The reward is used to update the policy using PPO (r_k).

---

## The Modern Upgrade: (Claude and GPT4o)

### Constitutional AI (CAI)

**1. Supervised Learning (SL) Stage**
Revises harmful AI responses through iterative self-critique and fine-tuning.

Steps:
1. Generate responses to harmfulness prompts
2. Critique and revise response
3. Fine-tune with SL on the final revised responses

Guided by constitutional principles: helpful, harmless, ethical, virtuous.

**2. Reinforcement Learning (RL) Stage**
Uses AI evaluations of responses according to constitutional principles to generate preference data for harmlessness, and uses it to train a new model via Reinforcement Learning from AI Feedback (RLAIF).

Steps:
4. AI generates dataset of preferences for harmlessness — "Which is better according to constitutional principles?"
5. Train preference model
6. Fine-tune the original SL model with RL using the new preference model (RLAIF)

Image source: https://www.anthropic.com/news/claudes-constitution

---

## LLM Limitations

### Knowledge of LLM constrained to pretraining data

Example — GPT-5 model card (screenshot taken Nov 9, 2025):

- **Reasoning**: Higher
- **Speed**: Medium
- **Price**: $1.25 input · $10 output
- **Input**: Text, image
- **Output**: Text
- 400,000 context window
- 128,000 max output tokens
- **Sep 30, 2024 knowledge cutoff**
- Reasoning token support

GPT-5 is OpenAI's flagship model for coding, reasoning, and agentic tasks across domains.

### Limited context size

GPT-5 has a 400,000 token context window.

### Pricing is per input/output token

GPT-5: $1.25 input · $10 output

---

## Other LLM Challenges

- Hallucination
- Lack of specialized information
- Lack of transparency
- Privacy & Security Risks
- Energy Consumption & Environmental Impact
- High Computational & Memory Costs

(e.g., Galactica)

---

## RAG: Motivations

**RAG = Retrieval-Augmented Generation**

- Domain-specific accurate answering
- Frequent updates of data
- Traceability and explainability of generated content
- Controllable Cost
- Privacy protection of data

---

## Prerequisite: Create knowledge base

Three-step pipeline:

1. **Collect** — gather documents (Document 1, Document 2, ...)
2. **Divide** — split each document into chunks
3. **Embed** — turn each chunk into a vector (e.g., [0.48, 0.33, ..., -0.51])

---

## RAG systems

- Retrieval augmented generation **RAG**: Augmented LLM with specialized and mutable knowledge base.

Flow:
1. User sends **prompt** to Q/A System.
2. Q/A System performs **retrieval** — context data, real-time data, etc. — from **External Data Stores** (Vector DB, Feature Store, etc.).
3. Q/A System sends augmented prompt query to **LLM**.
4. LLM response returned to user.

Reference: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
https://www.hopsworks.ai/dictionary/retrieval-augmented-generation-llm

---

## How RAG works

```
Input Text (Question)
        ↓
   Retriever (BM25, dense vector, etc.)  ←→  External Knowledge
        ↓ Ranked Context
    Reader (LM: BERT, GPT, etc.)
        ↓
   Output Text (Answer)
```

Retriever tools: TFiDF, chroma, Weaviate, Milvus, Qdrant, Elasticsearch, Faiss

Source: Hands-On Large Language Models, Book, O'Reilly, 2024

---

## How RAG works...

Pipeline:

1. **Parsing** — Document Corpus → Chunked Documents
   Example corpus text: "LLMs are first pre-trained, then they undergo alignment via SFT and RLHF. After this, applying them in practice requires a combination of fine-tuning and in-context learning."
   → split into chunks.

2. **Indexing** — Chunks → Embedding Model → Chunk Embeddings → Vector DB.

3. **Semantic search** — User prompt/input (e.g., "What are the three primary training techniques for an LLM?") → Search Engine queries Vector DB → Retrieved Chunks.

4. **Add Data to the Prompt!** — Retrieved chunks (e.g., "LLMs are first pre-trained, then they undergo", "alignment via SFT and RLHF. After this,") augment the prompt given to the LLM.

Source: https://cameronrwolfe.substack.com/p/a-practitioners-guide-to-retrieval

---

## Chunking Methods (Fixed Size Chunking)

Input: `Llama 2 was trained on 40% more data than Llama 1`

- **Character split** (every 15 characters): `Llama 2 was tra` | `ined on 40% mor` | `e data than Lla` | `ma 1` — *No overlap*
- **Token split** (every 5 tokens): `Llama 2 was trained on` | `40% more data than Llama` | `1`
- **Token split** (every 5 tokens with 1 overlapping token): `Llama 2 was trained on` | `on 40% more data than` | `than Llama 1` — *Overlapping tokens*

Source: Hands-On Large Language Models, Book, O'Reilly, 2024

---

## Chunking Methods (split on natural boundaries)

- **Each sentence is a chunk** (Chunk 1 vector, Chunk 2 vector, ..., Chunk 15 vector)
- **Each paragraph is a chunk** (Chunk 1, 2, 3 vectors)
- **Overlapping window of sentences** (Chunk 1, 2, 3 vectors)

Source: Hands-On Large Language Models, Book, O'Reilly, 2024

---

## Prompt Template

```
Prompt_Templet="answer the question based only the following context:{context}
---
Answer the question based on the above context :{question}"
```

---

## Vector database

### Dedicated vector databases

- **Open source (Apache 2.0 or MIT license)**: chroma, marqo, vespa, Qdrant, LanceDB, Milvus
- **Source available or commercial**: Weaviate, Pinecone

### Databases that support vector search

- **Open source**: OpenSearch, ClickHouse, PostgreSQL, Cassandra
- **Source available or commercial**: Elasticsearch, Redis, Rockset, SingleStore

Source: https://blog.det.life/why-you-shouldnt-invest-in-vector-databases-c0cd3f59d23c

---

## Quantify performance of retrieval

### Normalized Discounted Cumulative Gain at k (NDCG@k)

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i+1)}$$

- `rel_i` = relevance score at position i
- `log2(i+1)` = penalty for lower ranks
- with `rel_i ∈ {0, 1}`

$$\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

where IDCG@k = DCG@k if ranking was perfect.

---

## NDCG@k: Example

| Rank | Chunk | Relevance (rel_i) |
|---|---|---|
| 1 | Chunk text 1 | 1 |
| 2 | Chunk text 2 | 3 |
| 3 | Chunk text 3 | 2 |
| 4 | Chunk text 4 | 0 |
| 5 | Chunk text 5 | 1 |

**DCG:**
- 1/log2(2) = 1
- 3/log2(3) = 1.89
- 2/log2(4) = 1
- 0/log2(5) = 0
- 1/log2(6) = 0.39
- **Total DCG ≈ 4.28**

**The ideal sorted relevance:** [3, 2, 1, 1, 0]

**IDCG:**
- 3/1 = 3
- 2/1.585 = 1.26
- 1/2 = 0.5
- 1/2.32 = 0.43
- **Total IDCG ≈ 5.19**

**nDCG = DCG / IDCG = 4.28 / 5.19 ≈ 0.82**

---

## Quantify performance of retrieval...

### Reciprocal Rank at k (RR@k)

$$RR = \frac{1}{\text{rank}}$$

where `rank` = rank of the first relevant chunk.

---

## Quantify performance of retrieval...

### Recall at k

$$\text{Recall@}k = \frac{|\text{relevant in top } k|}{|\text{relevant}|}$$

---

## Quantify performance of retrieval...

### Precision at k

$$\text{Precision@}k = \frac{|\text{relevant in top } k|}{k}$$

---

## Settings to keep in mind

- One important setting is controlling how deterministic the model is when generating completion for prompts.
- **Temperature** and **top_p** are two important parameters to keep in mind.
- Generally, keep these low if you are looking for exact answers.
- Keep them high if you are looking for more diverse responses.

---

## RAG Frameworks

- There are many tools, libraries, and platforms with different capabilities and functionalities.
- Capabilities include:
  - Developing and experimenting with prompts
  - Evaluating prompts
  - Versioning and deploying prompts

Tools shown: Dyno (Prompt Engineering IDE), **LangChain**, haystack, LlamaIndex, DUST, PROMPTABLE

More tools: https://github.com/dair-ai/Prompt-Engineering-Guide#tools--libraries

---

## LangChain framework

- **Modular architecture** for flexible and adaptable LLM integrations.
- **Chaining together** multiple services beyond just LLMs.
- Goal-driven agent interactions instead of isolated calls.
- **Memory and persistence** for statefulness across executions.
- **Open-source access** and community support.

https://python.langchain.com/docs/get_started/introduction

---

## LangChain...

LangChain offers 507 integrations across categories:

- Document Loaders (157)
- Vector Stores (57)
- Embedding Models (43)
- Chat Models (19)
- LLMs (73)
- Callbacks (26)
- Tools (101)
- Toolkits (18)
- Message Histories (13)

Popular Document Loaders: AirbyteJSONLoader, ApifyDatasetLoader, UnstructuredHTMLLoader, UnstructuredPDFLoader, UnstructuredCSVLoader, UnstructuredURLLoader, OnlinePDFLoader, UnstructuredMarkdownLoader, UnstructuredFileLoader, UnstructuredExcelLoader, UnstructuredFileIOLoader, UnstructuredODTLoader, UnstructuredAPIFileIOLoader, UnstructuredAPIFileLoader, UnstructuredEPubLoader.

---

## Open Source LLMs

### Core general-purpose models

- Meta Llama 3
- Mistral AI
- Qwen
- Gemma 3
- Deepseek R1
- Microsoft Phi-4
- Gemma 3N (powerful and efficient; runs natively on phones, tablets and laptops)
- Falcon 3 (Redefining AI with Lightweight Power)

---

## Open Source LLMs — 8 GB RAM (Laptop / Low-End PC)

| Model | Size on Disk | Ollama Command |
|---|---|---|
| **Llama 3.3 8B** | 4.9 GB | `ollama pull llama3.3:8b` |
| **Mistral 7B** | 4.1 GB | `ollama pull mistral:7b` |
| **DeepSeek R1 7B** | ~5 GB | `ollama pull deepseek-r1:7b` |
| **Phi-4** | ~4 GB | `ollama pull phi4` |

Source: https://wp.astera.com/type/blog/open-source-vs-closed-source-llms/

---

## Open Source LLMs... — 16 GB RAM (Mid-Range Workstation)

| Model | Size on Disk | Ollama Command | Q&A Strength |
|---|---|---|---|
| DeepSeek R1 14B | ~9 GB | `ollama pull deepseek-r1:14b` | Chain-of-thought Q&A |
| Qwen 2.5 14B | ~9 GB | `ollama pull qwen2.5:14b` | Multilingual Q&A |
| Phi-4 14B | ~8 GB | `ollama pull phi4:14b` | Analytical reasoning |

---

## Choosing a Platform to Run LLMs

Tools to Run and Host Open-source LLMs:

- **Ollama**
- vLLM
- Hugging Face
- **LM Studio**

---

## Comparison between Ollama, vLLM, Hugging Face

| Factor | Ollama | vLLM | Hugging Face |
|---|---|---|---|
| **Ease of Use** | Very user-friendly | Requires more technical setup | User-friendly with extensive docs |
| **Performance** | Optimized for smaller models | Highly optimized for large-scale models | Good performance with scaling options |
| **Scalability** | Limited | High | Scalable with managed services |
| **Integration & Flexibility** | Limited flexibility | High flexibility for advanced users | High flexibility, integration into multiple tools |
| **Model Support** | Popular open-source models | Popular open-source models | Extensive (Transformers, etc.) |
| **Deployment** | Local | Cloud/On-premise | Cloud/On-premise |
| **Use Case** | Ideal for small to medium-scale | Best for high-throughput, enterprise-level use | Great for research, fine-tuning, and both small and large-scale use |

---

## LLM Models open source

- https://ollama.com/library
- https://lmstudio.ai/models

---

## Leaderboard: Community-driven Evaluation for Best LLM

| Model | Overall | Expert | Hard Prompts | Coding | Math | Creative Writing | Instruction Following | Longer Query |
|---|---|---|---|---|---|---|---|---|
| claude-opus-4-6-thi... | 1 | 2 | 1 | 1 | 2 | 1 | 1 | 1 |
| claude-opus-4-6 | 2 | 1 | 2 | 2 | 3 | 5 | 2 | 2 |
| gemini-3.1-pro-prev... | 3 | 4 | 3 | 3 | 4 | 4 | 3 | 3 |
| grok-4.20-beta1 | 4 | 22 | 5 | 6 | 17 | 2 | 10 | 13 |
| gemini-3-pro | 5 | 10 | 7 | 11 | 5 | 3 | 8 | 6 |
| gpt-5.4-high | 6 | 3 | 4 | 4 | 1 | 10 | 4 | 7 |
| grok-4.20-beta-0309... | 7 | 12 | 6 | 7 | 16 | 7 | 16 | 18 |
| gpt-5.2-chat-latest... | 8 | 16 | 8 | 8 | 12 | 15 | 11 | 15 |
| gemini-3-flash | 9 | 14 | 12 | 19 | 7 | 9 | 15 | 14 |
| claude-opus-4-5-202... | 10 | 7 | 9 | 5 | 8 | 6 | 5 | 4 |
| grok-4.1-thinking | 11 | 25 | 18 | 25 | 28 | 22 | 33 | 30 |

Source: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard

---

## Comparing Different LLM Models

**HELM — Holistic Evaluation of Language Models**

Takes Scenarios + Models → HELM → ranked evaluation output.

---

## HELM Evaluation Dimensions

- Accuracy
- Calibration
- Robustness
- Fairness
- Bias
- Efficiency

---

## Q&A

*"The smartest people are those who ask questions."* — Einstein meme

