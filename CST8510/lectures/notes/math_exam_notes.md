# CST8510 – Math Exam Notes

A focused cheat-sheet of every math-heavy concept from the lectures, with simple worked examples. Formula → intuition → concrete numeric example.

---

## 1. Evaluation Metrics

### 1.1 ROUGE (Recall-oriented)

```math
\text{ROUGE-Recall} = \frac{\text{overlapping n-grams}}{\text{n-grams in reference}}
```

**Example (unigrams):**
- Reference: "the cat sat on the mat" → 6 unigrams
- Generated: "the cat sat on a rug"
- Overlap: {the, cat, sat, on} = 4
- $\text{ROUGE-1} = 4/6 \approx \mathbf{0.667}$

### 1.2 BLEU (Precision-oriented)

```math
\text{BLEU-Precision} = \frac{\text{overlapping n-grams}}{\text{n-grams in generated}}
```

Same example: $4/6 \approx \mathbf{0.667}$.

> Memory hook: ROUGE looks at the **reference** (recall); BLEU looks at the **generated** (precision).

### 1.3 F1 Score (harmonic mean)

```math
F_1 = \frac{2 \cdot P \cdot R}{P + R}
```

**Example:** $P = 0.8,\ R = 0.6$

```math
F_1 = \frac{2 \cdot 0.8 \cdot 0.6}{0.8 + 0.6} = \frac{0.96}{1.4} \approx \mathbf{0.686}
```

### 1.4 G-Eval Weighted Score

```math
\text{Score} = \sum_{s=1}^{S} s \cdot P(s)
```

**Example:** $P(5)=0.6,\ P(4)=0.3,\ P(3)=0.1$

```math
\text{Score} = 5(0.6) + 4(0.3) + 3(0.1) = 3.0 + 1.2 + 0.3 = \mathbf{4.5}
```

### 1.5 RAGAS — Faithfulness

```math
\text{Faithfulness} = \frac{\text{supported statements}}{\text{total statements}}
```

**Example:** 4 of 5 claims supported → $4/5 = \mathbf{0.8}$.

### 1.6 RAGAS — Context Precision

```math
\text{Precision} = \frac{\text{relevant sentences}}{\text{total retrieved sentences}}
```

**Example:** 1 relevant of 10 retrieved → $1/10 = \mathbf{0.1}$ (noisy retrieval).

### 1.7 RAGAS — Answer Relevance

```math
\text{Relevance} = \frac{1}{N} \sum_{i=1}^{N} \mathrm{sim}(q_i^{\text{gen}},\ q^{\text{orig}})
```

Generate $N$ candidate questions from the answer; average cosine similarity to the original.

---

## 2. Feature Scaling & Transformations

### 2.1 Min-Max Scaling

```math
X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}} \in [0, 1]
```

**Example:** ages $[20, 30, 50, 80]$, $X_{\min}=20,\ X_{\max}=80$

```math
X=50 \Rightarrow \frac{50-20}{80-20} = \frac{30}{60} = \mathbf{0.5}
```

### 2.2 Z-score (Standard Scaling)

```math
Z = \frac{X - \mu}{\sigma}
```

**Example:** $\mu=100,\ \sigma=15,\ X=130 \Rightarrow Z = (130-100)/15 = \mathbf{2.0}$ (2 σ above mean).

### 2.3 Box-Cox Transformation

```math
X' = \begin{cases} \dfrac{X^{\lambda} - 1}{\lambda} & \text{if } \lambda \neq 0 \\[6pt] \ln(X) & \text{if } \lambda = 0 \end{cases}
```

**Example:** $\lambda = 0.5,\ X = 16 \Rightarrow (16^{0.5} - 1)/0.5 = 3/0.5 = \mathbf{6}$.

> **Leakage rule:** split FIRST, then fit the transformer on train, apply to test.

---

## 3. Vectors, Embeddings & Similarity

### 3.1 Dot Product

```math
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i\, b_i
```

**Example:** $\mathbf{a}=[1,2,3],\ \mathbf{b}=[4,5,6]$

```math
\mathbf{a}\cdot\mathbf{b} = 1(4) + 2(5) + 3(6) = 4 + 10 + 18 = \mathbf{32}
```

### 3.2 Vector Norm (magnitude)

```math
\lVert \mathbf{a} \rVert = \sqrt{\sum_{i=1}^{n} a_i^2}
```

**Example:** $\mathbf{a}=[3,4] \Rightarrow \sqrt{9+16} = \sqrt{25} = \mathbf{5}$.

### 3.3 Cosine Similarity

```math
\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert\, \lVert \mathbf{b} \rVert} \in [-1, 1]
```

**Example:** $\mathbf{a}=[1,2,3],\ \mathbf{b}=[2,4,6]$
- $\mathbf{a}\cdot\mathbf{b} = 2+8+18 = 28$
- $\lVert\mathbf{a}\rVert = \sqrt{14} \approx 3.742,\ \lVert\mathbf{b}\rVert = \sqrt{56} \approx 7.483$
- $\cos(\theta) = 28 / (3.742 \cdot 7.483) \approx \mathbf{1.0}$ (same direction)

### 3.4 Word2Vec Analogy

```math
\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}
```

### 3.5 Pooling (word → sentence embedding)

- **Average pooling:** element-wise mean of word vectors
- **Max pooling:** element-wise maximum

**Example:** word vectors $[1,2],\ [3,4],\ [5,0]$

```math
\text{Avg} = \left[\tfrac{1+3+5}{3},\ \tfrac{2+4+0}{3}\right] = [3,\ 2] \qquad \text{Max} = [5,\ 4]
```

---

## 4. Probability & Uncertainty

### 4.1 Softmax

```math
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
```

**Example:** logits $= [2, 1, 0]$
- $e^x = [7.39,\ 2.72,\ 1.00]$, sum $= 11.11$
- probs $\approx [0.665,\ 0.245,\ 0.090]$ (sum to 1)

### 4.2 Cross-Entropy Loss

```math
L = -\sum_{i} y_i\, \log(p_i)
```

**Example:** $\mathbf{y}=[1,0,0],\ \mathbf{p}=[0.7,0.2,0.1]$

```math
L = -\log(0.7) \approx \mathbf{0.357}
```

### 4.3 Self-Consistency (majority vote)

Sample $N$ reasoning paths, take the most common answer.

**Example:** $[11\text{AM},\ 11\text{AM},\ 11\text{AM},\ 12\text{PM},\ 12\text{PM}]$ → majority $=\mathbf{11\text{AM}}\ (3/5)$.

### 4.4 Autocorrelation (time series)

```math
\rho(\tau) = \mathrm{Corr}(X_t,\ X_{t+\tau})
```

High $\rho$ → random splits leak information; split by time instead.

---

## 5. Gradient Descent & Training Math

### 5.1 Parameter Update

```math
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla L
```

**Example:** $\theta=5.0,\ \nabla L=2.0,\ \eta=0.1 \Rightarrow \theta_{\text{new}} = 5.0 - 0.1(2.0) = \mathbf{4.8}$.

### 5.2 Effective Batch Size (gradient accumulation)

```math
B_{\text{eff}} = B \times A
```

where $B$ = batch size, $A$ = accumulation steps.

**Example:** $4 \times 4 = \mathbf{16}$.

### 5.3 Batches per Epoch

```math
\text{batches} = \left\lceil \frac{N}{B} \right\rceil
```

where $N$ = sample count, $B$ = batch size.

**Example:** $\lceil 345/4 \rceil = \mathbf{87}$ batches/epoch → 10 epochs = **870 batches** total.

---

## 6. Memory, Quantization & Parallelism

### 6.1 Model Memory Footprint

```math
\text{memory (bytes)} = N_{\text{params}} \times \text{bytes per param}
```

**Example:** LLaMA 7B at different precisions:

| Precision | Bytes/param | Total |
|-----------|------------|-------|
| FP32      | 4          | 28 GB |
| FP16/BF16 | 2          | **14 GB** |
| INT8      | 1          | 7 GB |
| INT4      | 0.5        | 3.5 GB |

Training needs $\approx 2\times$ inference memory → 7B at FP16 needs ~28 GB → won't fit on 24 GB GPU.

### 6.2 50% Utilization Rule

```math
\text{FLOPS}_{\text{real}} \approx 0.5 \cdot \text{FLOPS}_{\text{theoretical}}
```

### 6.3 Gradient Checkpointing

Store every $\sqrt{n}$-th activation; recompute the rest. Trade ~10× memory for ~20% extra compute.

---

## 7. Model Compression

### 7.1 Low-Rank Factorization

```math
A_{M \times N} \approx B_{M \times K} \cdot C_{K \times N}, \quad K \ll M, N
```

**Example:** $M=N=1000,\ K=10$
- Original: $1000 \cdot 1000 = 1{,}000{,}000$ params
- Factored: $1000\cdot 10 + 10\cdot 1000 = \mathbf{20{,}000}$ (98% reduction)

### 7.2 LoRA

```math
W' = W + \Delta W, \quad \Delta W = A \cdot B
```

```math
A \in \mathbb{R}^{m\times k},\quad B \in \mathbb{R}^{k\times n},\quad k \ll \min(m,n)
```

**Example:** $1000\times 1000$ frozen $W$, rank $k=8$

```math
|A| + |B| = 1000\cdot 8 + 8\cdot 1000 = \mathbf{16{,}000}\ \text{trainable}\ (1.6\%\ \text{of original})
```

### 7.3 Knowledge Distillation Loss

```math
L_{\text{soft}} = T^2 \cdot \mathrm{KL}\!\left(\mathrm{softmax}(z_T/T)\ \|\ \mathrm{softmax}(z_S/T)\right)
```

```math
L_{\text{hard}} = \mathrm{CE}(z_S,\ y)
```

```math
L_{\text{total}} = \alpha \cdot L_{\text{soft}} + (1-\alpha) \cdot L_{\text{hard}}
```

Typical $T = 3.0,\ \alpha \approx 0.5$. DistilBERT retains 97% of BERT accuracy with ~40% fewer params.

### 7.4 Pruning

Up to **90% of weights zeroed** with minimal accuracy loss on overparameterized nets.

---

## 8. Hyperparameter Tuning — Cost Comparison

### 8.1 Grid Search — $O(v^k)$ exponential

**Example:** 3 hyperparams × 10 values = $10^3 = \mathbf{1000}$ evaluations.

### 8.2 Random Search — $O(n)$ linear

~50 random samples often matches a full grid.

### 8.3 Bayesian Optimization

```math
\mathrm{EI}(x) = \mathbb{E}\!\left[\max\!\left(f(x) - f(x^{*}),\ 0\right)\right]
```

Converges in ~10–12 iterations using a Gaussian-Process surrogate.

---

## 9. Differentiable NAS (DARTS)

```math
\bar{o}(x) = \sum_{i} \frac{\exp(\alpha_i)}{\sum_{j} \exp(\alpha_j)} \cdot o_i(x)
```

Softmax over candidate operations, trained by gradient descent.

**Example:** $\boldsymbol{\alpha}=[2,1,0]$ → weights $\approx [0.665,\ 0.245,\ 0.090]$. Poor ops' $\alpha$ falls; good ops dominate.

---

## 10. Shapley Values (SHAP)

```math
\varphi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(|N|-|S|-1)!}{|N|!} \cdot \bigl[f(S \cup \{i\}) - f(S)\bigr]
```

Feature $i$'s value = average marginal contribution across all coalitions.

**Intuition:** average of "what does feature $i$ add when included in every possible subset?"

---

## 11. Missing Data Types (MCAR / MAR / MNAR)

| Type | Depends on | Example | Fixable? |
|------|-----------|---------|----------|
| MCAR | Nothing | Survey page torn | ✅ simple methods OK |
| MAR  | Other observed vars | Young folks skip income | ✅ multiple imputation |
| MNAR | The missing value itself | Depressed skip mental-health Qs | ⚠️ biased, needs special models |

> **Golden rule:** split → impute on train → apply same stats to test. Never impute before splitting.

---

## 12. Roblox Inference Optimization — cumulative math

Baseline BERT: 330 ms, <100 msg/s.

| Step | Change | Latency | Throughput |
|------|--------|---------|-----------|
| 0 | BERT | 330 ms | ~100/s |
| 1 | Threads=1 | — | modest |
| 2 | DistilBERT | 171 ms | 185/s |
| 3 | Dynamic shape | 69 ms | 369/s |
| 4 | INT8 quantization | **10 ms** | **3015/s** |
| 5 | + caching | — | 2× extra |

```math
\text{Speedup} = \frac{330}{10} = \mathbf{33\times}
```

---

## 13. Quick-Reference Formula Sheet

| # | Concept | Formula |
|---|---------|---------|
| 1 | ROUGE recall | $\dfrac{\text{overlap}}{\text{ref n-grams}}$ |
| 2 | BLEU precision | $\dfrac{\text{overlap}}{\text{gen n-grams}}$ |
| 3 | F1 | $\dfrac{2PR}{P+R}$ |
| 4 | Min-Max | $\dfrac{X-X_{\min}}{X_{\max}-X_{\min}}$ |
| 5 | Z-score | $\dfrac{X-\mu}{\sigma}$ |
| 6 | Box-Cox | $\dfrac{X^{\lambda}-1}{\lambda}$ |
| 7 | Cosine sim | $\dfrac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert}$ |
| 8 | Softmax | $\dfrac{e^{x_i}}{\sum_j e^{x_j}}$ |
| 9 | Cross-entropy | $-\sum_i y_i \log p_i$ |
| 10 | Grad update | $\theta - \eta\,\nabla L$ |
| 11 | Memory | $N_{\text{params}} \times$ bytes/param |
| 12 | LoRA | $W + A\cdot B$ |
| 13 | Distill loss | $\alpha\, T^2\,\mathrm{KL} + (1-\alpha)\,\mathrm{CE}$ |
| 14 | G-Eval | $\sum s\cdot P(s)$ |
| 15 | Faithfulness | $\dfrac{\text{supported}}{\text{total}}$ |
| 16 | Grid cost | $O(v^k)$ |
| 17 | DARTS mix | $\sum_i \mathrm{softmax}(\alpha_i)\,o_i(x)$ |

---

## 14. Applied / Implicit Math (Latency, Throughput, Cost, Capacity)

The lectures rarely write these as formulas, but exam questions expect you to derive them.

### 14.1 Latency ↔ Throughput duality

```math
\text{throughput (req/s)} \approx \frac{\text{concurrency}}{\text{latency (s)}}
```

For a single worker: $\text{throughput} = 1/\text{latency}$.

**Roblox (Lecture 6):**
- Vanilla BERT: $1/0.330 \approx \mathbf{3}$ req/s per worker → ~100 msg/s across 36 cores
- INT8 DistilBERT: $1/0.010 = \mathbf{100}$ req/s per worker → ~3,015 msg/s across 36 cores
- Speedup: $330/10 = \mathbf{33\times}$

> If you halve latency, you roughly double throughput at fixed concurrency.

### 14.2 Daily Volume Capacity

```math
\text{requests per day} = \text{throughput (req/s)} \times 86{,}400\ \text{s}
```

**Example (Roblox ≈ 2 B msgs/day):**

```math
\frac{2 \times 10^9}{86{,}400} \approx \mathbf{23{,}148}\ \text{msg/s}
```

At 3,015 msg/s per box → need $\lceil 23{,}148 / 3{,}015 \rceil = \mathbf{8}$ machines for the mean (more for peaks).

### 14.3 Latency Budgets

| System | Tolerable latency |
|--------|------------------|
| Online prediction / real-time | $< 100$ ms |
| RAG chatbot | $3$–$5$ s |
| Customer-support agent | $\leq 1$ min |

**RAG pipeline example:** $200 + 300 + 2000 = 2{,}500$ ms → under the 5 s budget.

### 14.4 CPU vs GPU for Inference

```math
\frac{\text{CPU throughput}}{\text{GPU throughput}} = \frac{3000}{450} \approx \mathbf{6.7\times}
```

GPUs aren't always faster when batch = 1 and latency matters; CPU wins for short requests.

### 14.5 Caching Savings

```math
t_{\text{eff}} = p_{\text{hit}} \cdot t_{\text{cache}} + (1 - p_{\text{hit}}) \cdot t_{\text{model}}
```

**Example:** $t_{\text{model}}=10$ ms, $t_{\text{cache}}=0.1$ ms, $p_{\text{hit}}=0.6$

```math
t_{\text{eff}} = 0.6(0.1) + 0.4(10) = \mathbf{4.06}\ \text{ms}\ (\approx 2.5\times\ \text{throughput})
```

### 14.6 Chunking & Overlap

- Chunk size = 512 tokens, overlap = 128 → $\text{overlap ratio} = 128/512 = \mathbf{25\%}$
- Stride $= 512 - 128 = 384$ tokens per step

**Example:** 10,000-token doc

```math
\text{chunks} \approx \left\lceil \frac{10000 - 128}{384} \right\rceil = \lceil 25.7 \rceil = \mathbf{26}
```

### 14.7 Context Precision — noisy retrieval

1 relevant of 10 retrieved → $1/10 = \mathbf{0.10}$ precision. Fix: reduce top-$k$, or rerank with a cross-encoder.

### 14.8 Fine-tuning Data/Compute Budget

**MITRE dataset (Lecture 12):** 345 train samples, batch = 4, accumulation = 4
- Effective batch $= 4 \times 4 = 16$
- Batches/epoch $= \lceil 345/4 \rceil = 87$
- Steps in 10 epochs $= 870$
- Optimizer updates $= 870 / 4 \approx \mathbf{218}$

**LoRA trainable fraction:**

```math
\frac{1.4 \times 10^6}{1 \times 10^9} = \mathbf{0.14\%}\ \ (\text{LLaMA 1B})
```

```math
\frac{2.25 \times 10^6}{6.72 \times 10^8} = \mathbf{0.33\%}\ \ (\text{TinyLlama})
```

### 14.9 Training Memory Rule

```math
\text{memory}_{\text{train}} \approx 2 \cdot \text{memory}_{\text{inference}}
```

**Example:** 7B at FP16
- Inference: $7\times 10^9 \cdot 2\ \text{B} = \mathbf{14}$ GB
- Training: $\approx \mathbf{28}$ GB → needs H100 80 GB

**QLoRA:** quantize frozen base to 4-bit → $7 \times 10^9 \cdot 0.5\ \text{B} = 3.5$ GB + tiny adapter → fits on 16 GB GPU.

### 14.10 Accuracy vs Cost

Simple model (90%, cheap) vs deep model (92–95%, $5$–$10\times$ compute).

| Option | Per request | 1 M requests/day |
|--------|------------|------------------|
| Simple | \$0.0001 | \$100/day |
| Deep   | \$0.001  | \$1,000/day |

$\Delta = \$900/\text{day}$ for $+2\%$ accuracy — worth it only if that gain unlocks > \$900/day in value.

### 14.11 Gradient Checkpointing Trade-off

- Memory: $\approx 10\times$ reduction
- Compute: $\approx 20\%$ slower
- Usually a win if model otherwise OOMs.

### 14.12 Compute Utilization Rule-of-Thumb

```math
\text{FLOPS}_{\text{real}} \approx 0.5 \cdot \text{FLOPS}_{\text{spec}}
```

Plan capacity for $2\times$ what the spec implies.

### 14.13 Hyperparameter Search Iteration Counts

| Method | Iterations (3 params × 10 values) |
|--------|-----------------------------------|
| Grid   | $10^3 = 1000$ |
| Random | $\sim 50$ |
| Bayesian | $\sim 10$–$12$ |

> Grid is $\sim 100\times$ more expensive than Bayesian for equivalent coverage.

### 14.14 Uncertainty Threshold (self-consistency)

```math
\text{decision} = \begin{cases} \text{majority answer} & \text{if unique-output proportion} < 0.5 \\ \text{``I don't know''} & \text{otherwise} \end{cases}
```

**Example:** $[A, A, A, B, C]$ → unique-proportion $= 3/5 = 0.6$ → **reject**.
$[A, A, A, A, B]$ → $2/5 = 0.4$ → **accept A**.

### 14.15 Underfit vs Overfit via Learning Curves

| Observation | Diagnosis | Action |
|------------|-----------|--------|
| Train & val both ~60% | Underfit | Bigger model / more features |
| Train 99%, val 70% (big gap) | Overfit | More data / regularization |
| Train 95%, val 94% (small gap, low) | Bias | Richer model |
| Train 99%, val 98% (small gap, high) | ✅ good | — |

---

## 15. Exam Traps / Gotchas

1. **ROUGE vs BLEU direction** — recall vs precision; know which denominator.
2. **Data leakage** — impute/scale **after** split, not before. Time series: split by time.
3. **7B at FP16 ≠ 14 GB to train** — training needs ~2× for gradients + optimizer.
4. **Cosine sim = 1 doesn't mean semantically identical** — just the same direction.
5. **Softmax outputs must sum to 1** — if not, you computed wrong.
6. **LoRA rank $k$ is small** — typical $k \in [4, 16]$.
7. **Grid is exponential**, random is linear, Bayesian needs ~10–12 iterations.
8. **MNAR is the dangerous one** — biased even with perfect imputation.
9. **Effective batch = batch × accumulation** — not just batch.
10. **50% utilization rule** — always plan for $\sim 2\times$ your theoretical compute.
