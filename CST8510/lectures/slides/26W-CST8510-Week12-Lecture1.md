# Artificial Intelligence Software Development

**CST8510 Week 12**
Dr. Hari M Koduvely

---

## Agenda for Today

- **Theory: 5:30PM – 7:30PM**
  - LLM Fine-Tuning
- **Lab: 7:30PM – 9:30PM**
  - Standup Meetings

---

## LLM Fine-Tuning

- Process of tuning a broad general-purpose LLM to perform exceptionally well on specific domain or task.
- Foundation models or Base models are pre-trained on a vast amount of data to perform well on a general tasks.
- However, they would fail miserably when asked about very domain specific questions.
- Example Home Cook Vs Sushi Chef

---

## LLM Fine-Tuning (Diagram)

The diagram illustrates the fine-tuning concept using a cooking analogy with three stages:

**The Home Cook (Pre-Trained LLM) — "Homer"**
- Knows Basic Cooking (General Knowledge: Language, Grammar, Common Facts)
- Can make: Steak, Soup, Pasta, Tacos
- Represents the Pre-trained Model (Generalist)

**LLM Fine-Tuning (Specialized Training)**
- Input: General Cooking Data → Pre-trained Model (Generalist)
- Input: Japanese/Sushi Data → Specific, Domain Data
- Goes through the Fine-Tuning Process
- Output: Refined Model (Specialist) with Sushi Expertise

**The Sushi Chef (Fine-Tuned LLM) — "Kenji"**
- Masters One Cuisine (Domain-Specific Knowledge: Deep Understanding of Sushi)
- Can make: Delicate Omakase

*From General Knowledge to specialized Expertise through targeted data and training*

---

## When to Fine-Tune LLMs?

- **LLM needs to perform Complex Tasks**
  - E.g. Code like an expert software developer
- **Cost and Latency need to be optimized**
  - Achieve the same performance using a smaller and more efficient model.
- **Critical applications** where accuracy and reliability are paramount.
- **Adjust Tone, Style & Behaviour** to your customer audience
- **Enforce Structure and Format**
- **Domain Adaptation**

---

## Do these before Fine-Tuning

- **Prompt Engineering**
  - Few shot learning examples
  - Chain-of-Thought Prompting
  - Self-consistent and Threshold based prompting
  - Prompt optimization using OPRO
- **Advanced Retrieval Augmented Generation (RAG)**
  - Semantic Chunking
  - Query Transformation
  - Fusion Retrieval
  - Re-ranking of the retrieved context

---

## When Not to Do Fine-Tuning

- **Data changes too frequently**
  - E.g. Stock
- **Simple tasks**
- **Low-quality data**
- **Privacy Constraints**

---

## Different Types of Fine-Tuning

- **Full Fine-Tuning**
  - Update all the parameters of an LLM.
  - Powerful, but expensive
  - Could result in catastrophic forgetting
- **LoRA (Low-Rank Adaptation)**
  - Freezes the original model weights
  - Weight in selected layers are updated using small, trainable "adapter" modules.
  - Only the adapters are trained, which is more memory and computationally efficient process.
  - Adapters can be switched easily to adapt to different domains/tasks.

---

## Low Rank Adaptation (LoRA) (Diagram)

The diagram compares the model before and after fine-tuning:

**Before Fine-Tuning (General Knowledge)**
- Base LLM (W_orig) with Billions of Parameters
- Knowledge from Diverse Datasets, Common Language
- Original Frozen Weights (W_orig) — size d × d
- General Responses for a wide range of questions: Writing poetry, explaining history, summarizing news.

**LoRA Fine-Tuning Process (Specialized Training)**
- Training Process applied with New, Domain-Specific Dataset (e.g., Medical Texts)

**After Fine-Tuning (Specialized & General Knowledge)**
- Original Frozen W_orig (d × d) combined with LoRA Adapter (Small Task-Specific)
- Custom Task-Specific Responses with Domain-Specific Terms
- Formula: **W_new = W_orig + (A × B)**
- Produces Fine-Tuned, Domain-Specific LLM (Medical Diagnosis) with Responses for Medical Diagnosis / Code Generation

**Fine-Tuning Comparison:**
- *Traditional Fine-Tuning*: Update All Parameters → High Cost
- *LoRA Fine-Tuning*: Add Low-Rank Matrices → Low Cost, Fused, Specialized Model

**Component Summary Table:**

| Component | Function | Update Status | Size (Parameters) |
|-----------|----------|---------------|-------------------|
| Base LLM | Core Knowledge (Generalist) | FROZEN | BILLIONS |
| LoRA Adapter | New Tasks/Domain Skill (Specialist) | UPDATABLE | MILLIONS (Very Small Fraction) |

---

## Different Fine-Tuning Frameworks

| Feature | Unsloth | Axolotl | LLaMA-Factory | HF PEFT |
|---------|---------|---------|---------------|---------|
| **Primary Goal** | Raw Speed / Low VRAM | Reproducibility / Scaling | Ease of Use / All-in-one | Core Logic / Integration |
| **Interface** | Python / No-code Studio | YAML Config | Web UI (LlamaBoard) | Python API |
| **Best Hardware** | Single GPU (Consumer) | Multi-GPU (H100/A100 clusters) | Flexible | Any |
| **Difficulty** | Low (Studio) to Med | High | Low | Med-High |
| **Multi-GPU?** | Yes (Pro/Enterprise) | Native / Best | Yes | Yes |

---

## Workout Example Using Colab

Fine-Tuning Llama Model with Cybersecurity Domain Data

---

## Summary of Today's Learning

- What is Fine-Tuning LLMs.
- When to and not to fine-tune.
- LoRA and PEFT
- Colab implementation of Llama fine-tuning using Cybersecurity Data

---

## Thank You
