# CST8506 – Advanced Machine Learning

## Week 11: Association Rule Mining

Dr. Abbas Akkasi
Winter 2026

---

## Association Rule Mining

Given a set of transactions, find rules that will predict the occurrence of an item based on the occurrences of other items in the transaction.

**Market-Basket transactions**

| TID | Items |
|-----|-------|
| 1 | Bread, Milk |
| 2 | Bread, Diaper, Beer, Eggs |
| 3 | Milk, Diaper, Beer, Coke |
| 4 | Bread, Milk, Diaper, Beer |
| 5 | Bread, Milk, Diaper, Coke |

**Example of Association Rules**

- {Diaper} → {Beer}
- {Milk, Bread} → {Eggs, Coke}
- {Beer, Bread} → {Milk}

> Implication means co-occurrence, not causality!

---

## Definition: Frequent Itemset

- **Itemset**
  - A collection of one or more items
    - Example: {Milk, Bread, Diaper}
  - **k-itemset**
    - An itemset that contains k items

- **Support count (σ)**
  - Frequency of occurrence of an itemset
  - E.g. σ({Milk, Bread, Diaper}) = 2

- **Support**
  - Fraction of transactions that contain an itemset
  - E.g. s({Milk, Bread, Diaper}) = 2/5

- **Frequent Itemset**
  - An itemset whose support is greater than or equal to a *minsup* threshold

---

## Definition: Association Rule

- **Association Rule**
  - An implication expression of the form X → Y, where X and Y are itemsets
  - Example: {Milk, Diaper} → {Beer}

- **Rule Evaluation Metrics**
  - **Support (s)**: Fraction of transactions that contain both X and Y
  - **Confidence (c)**: Measures how often items in Y appear in transactions that contain X

**Example:** {Milk, Diaper} ⇒ {Beer}

$$s = \frac{\sigma(\text{Milk, Diaper, Beer})}{|T|} = \frac{2}{5} = 0.4$$

$$c = \frac{\sigma(\text{Milk, Diaper, Beer})}{\sigma(\text{Milk, Diaper})} = \frac{2}{3} = 0.67$$

---

## Association Rule Mining Task

Given a set of transactions T, the goal of association rule mining is to find all rules having:
- support ≥ *minsup* threshold
- confidence ≥ *minconf* threshold

**Brute-force approach:**
- List all possible association rules
- Compute the support and confidence for each rule
- Prune rules that fail the *minsup* and *minconf* thresholds
- ⇒ **Computationally prohibitive!**

---

## Computational Complexity

Given d unique items:
- Total number of itemsets = 2^d
- Total number of possible association rules:

$$R = \sum_{k=1}^{d-1}\left[\binom{d}{k} \times \sum_{j=1}^{d-k}\binom{d-k}{j}\right] = 3^d - 2^{d+1} + 1$$

**If d = 6, R = 602 rules**

---

## Mining Association Rules

**Example of Rules:**

- {Milk, Diaper} → {Beer} (s=0.4, c=0.67)
- {Milk, Beer} → {Diaper} (s=0.4, c=1.0)
- {Diaper, Beer} → {Milk} (s=0.4, c=0.67)
- {Beer} → {Milk, Diaper} (s=0.4, c=0.67)
- {Diaper} → {Milk, Beer} (s=0.4, c=0.5)
- {Milk} → {Diaper, Beer} (s=0.4, c=0.5)

**Observations:**
- All the above rules are binary partitions of the same itemset: {Milk, Diaper, Beer}
- Rules originating from the same itemset have identical support but can have different confidence
- Thus, we may decouple the support and confidence requirements

---

## Mining Association Rules — Two-step Approach

1. **Frequent Itemset Generation**
   - Generate all itemsets whose support ≥ minsup

2. **Rule Generation**
   - Generate high confidence rules from each frequent itemset, where each rule is a binary partitioning of a frequent itemset

Frequent itemset generation is still computationally expensive.

---

## Frequent Itemset Generation

Itemset lattice for items {A, B, C, D, E} — from `null` at the top down through 1-itemsets, 2-itemsets, 3-itemsets, 4-itemsets, to the 5-itemset {ABCDE}.

**Given d items, there are 2^d possible candidate itemsets.**

---

## Frequent Itemset Generation — Brute-force

**Brute-force approach:**
- Each itemset in the lattice is a **candidate** frequent itemset
- Count the support of each candidate by scanning the database
- Match each transaction against every candidate
- Complexity ~ O(NMw) ⇒ **Expensive since M = 2^d !!!**

Where:
- N = number of transactions
- M = number of candidate itemsets
- w = transaction width

---

## Frequent Itemset Generation Strategies

- **Reduce the number of candidates (M)**
  - Complete search: M = 2^d
  - Use pruning techniques to reduce M

- **Reduce the number of transactions (N)**
  - Reduce size of N as the size of itemset increases

- **Reduce the number of comparisons (NM)**
  - Use efficient data structures to store the candidates or transactions
  - No need to match every candidate against every transaction

---

## Reducing Number of Candidates

**Apriori principle:**
- If an itemset is frequent, then all of its subsets must also be frequent

Apriori principle holds due to the following property of the support measure:

$$\forall X, Y : (X \subseteq Y) \Rightarrow s(X) \geq s(Y)$$

- Support of an itemset never exceeds the support of its subsets
- This is known as the **anti-monotone** property of support

---

## Illustrating Apriori Principle

If an itemset (e.g. AB) is found to be **infrequent**, all of its supersets (ABC, ABD, ABE, ABCD, ABCE, ABDE, ABCDE) can be **pruned** from the search space.

---

## Illustrating Apriori Principle — Example

**Transactions:**

| TID | Items |
|-----|-------|
| 1 | Bread, Milk |
| 2 | Beer, Bread, Diaper, Eggs |
| 3 | Beer, Coke, Diaper, Milk |
| 4 | Beer, Bread, Diaper, Milk |
| 5 | Bread, Coke, Diaper, Milk |

**Items (1-itemsets):**

| Item | Count |
|------|-------|
| Bread | 4 |
| Coke | 2 |
| Milk | 4 |
| Beer | 3 |
| Diaper | 4 |
| Eggs | 1 |

**Minimum Support = 3**

- If every subset is considered: ⁶C₁ + ⁶C₂ + ⁶C₃ = 6 + 15 + 20 = 41
- With support-based pruning: 6 + 6 + 4 = 16

With minsup = 3, prune Coke (count=2) and Eggs (count=1).

**Pairs (2-itemsets)** — No need to generate candidates involving Coke or Eggs:

| Itemset | Count |
|---------|-------|
| {Bread, Milk} | 3 |
| {Bread, Beer} | 2 |
| {Bread, Diaper} | 3 |
| {Milk, Beer} | 2 |
| {Milk, Diaper} | 3 |
| {Beer, Diaper} | 3 |

**Triplets (3-itemsets):**

| Itemset | Count |
|---------|-------|
| {Beer, Diaper, Milk} | 2 |
| {Beer, Bread, Diaper} | 2 |
| {Bread, Diaper, Milk} | 2 |
| {Beer, Bread, Milk} | 1 |

---

## Apriori Algorithm

- **F_k**: frequent k-itemsets
- **L_k**: candidate k-itemsets

**Algorithm:**
- Let k = 1
- Generate F₁ = {frequent 1-itemsets}
- Repeat until F_k is empty:
  - **Candidate Generation**: Generate L_{k+1} from F_k
  - **Candidate Pruning**: Prune candidate itemsets in L_{k+1} containing subsets of length k that are infrequent
  - **Support Counting**: Count the support of each candidate in L_{k+1} by scanning the DB
  - **Candidate Elimination**: Eliminate candidates in L_{k+1} that are infrequent, leaving only those that are frequent ⇒ F_{k+1}

---

## Candidate Generation: Brute-force method

Starting from the frequent items {Beer, Bread, Cola, Diapers, Eggs, Milk}, brute-force candidate 3-itemset generation produces all ⁶C₃ = 20 triplets, which are then pruned by support and by the Apriori property, leaving only {Bread, Diapers, Milk} as the frequent 3-itemset.

---

## Candidate Generation: Merge F_{k-1} and F₁ itemsets

Merging frequent 2-itemsets {{Beer, Diapers}, {Bread, Diapers}, {Bread, Milk}, {Diapers, Milk}} with frequent 1-items {Beer, Bread, Diapers, Milk} yields the candidate 3-itemsets {{Beer, Bread, Diapers}, {Beer, Diapers, Milk}, {Bread, Diapers, Milk}, {Beer, Bread, Milk}}. After candidate pruning, only {Bread, Diapers, Milk} remains.

Some candidates are unnecessary because their subsets are infrequent.

---

## Candidate Generation: F_{k-1} × F_{k-1} Method

Merge two frequent (k-1)-itemsets if their **first (k-2) items are identical**.

F₃ = {ABC, ABD, ABE, ACD, BCD, BDE, CDE}

- Merge(**AB**C, **AB**D) = **AB**CD
- Merge(**AB**C, **AB**E) = **AB**CE
- Merge(**AB**D, **AB**E) = **AB**DE

Do not merge(**A**BD, **A**CD) because they share only a prefix of length 1 instead of length 2.

---

## Candidate Pruning

Let F₃ = {ABC, ABD, ABE, ACD, BCD, BDE, CDE} be the set of frequent 3-itemsets.

L₄ = {ABCD, ABCE, ABDE} is the set of candidate 4-itemsets generated (from previous slide).

**Candidate pruning:**
- Prune ABCE because ACE and BCE are infrequent
- Prune ABDE because ADE is infrequent

**After candidate pruning:** L₄ = {ABCD}

---

## Candidate Generation: F_{k-1} × F_{k-1} Method — Worked Example

Merging pairs of frequent 2-itemsets {{Beer, Diapers}, {Bread, Diapers}, {Bread, Milk}, {Diapers, Milk}} produces the single candidate 3-itemset {Bread, Diapers, Milk}. After candidate pruning, it remains as {Bread, Diapers, Milk}.

Use of the F_{k-1} × F_{k-1} method for candidate generation results in only one 3-itemset. Surviving sets are eliminated by the support counting step if they fail minsup.

Using this method with the earlier example: Total candidates considered = 6 + 6 + 1 = 13 (vs. 16 for the F_{k-1} × F₁ method, and 41 for brute-force).

---

## Alternate F_{k-1} × F_{k-1} Method

Merge two frequent (k-1)-itemsets if the **last (k-2) items of the first one is identical to the first (k-2) items of the second**.

F₃ = {ABC, ABD, ABE, ACD, BCD, BDE, CDE}

- Merge(A**BC**, **BC**D) = A**BC**D
- Merge(A**BD**, **BD**E) = A**BD**E
- Merge(A**CD**, **CD**E) = A**CD**E
- Merge(B**CD**, **CD**E) = B**CD**E

---

## Candidate Pruning for Alternate F_{k-1} × F_{k-1} Method

Let F₃ = {ABC, ABD, ABE, ACD, BCD, BDE, CDE} be the set of frequent 3-itemsets.

L₄ = {ABCD, ABDE, ACDE, BCDE} is the set of candidate 4-itemsets generated (from previous slide).

**Candidate pruning:**
- Prune ABDE because ADE is infrequent
- Prune ACDE because ACE and ADE are infrequent
- Prune BCDE because BCE is infrequent

**After candidate pruning:** L₄ = {ABCD}

---

## Support Counting of Candidate Itemsets

Scan the database of transactions to determine the support of each candidate itemset.
- Must match every candidate itemset against every transaction, which is an **expensive operation**.

**Transactions:**

| TID | Items |
|-----|-------|
| 1 | Bread, Milk |
| 2 | Beer, Bread, Diaper, Eggs |
| 3 | Beer, Coke, Diaper, Milk |
| 4 | Beer, Bread, Diaper, Milk |
| 5 | Bread, Coke, Diaper, Milk |

**Candidate itemsets:**
- {Beer, Diaper, Milk}
- {Beer, Bread, Diaper}
- {Bread, Diaper, Milk}
- {Beer, Bread, Milk}

---

## Rule Generation

Given a frequent itemset L, find all non-empty subsets f ⊂ L such that f → L − f satisfies the minimum confidence requirement.

If {A, B, C, D} is a frequent itemset, candidate rules:

| | | | |
|---|---|---|---|
| ABC → D | ABD → C | ACD → B | BCD → A |
| A → BCD | B → ACD | C → ABD | D → ABC |
| AB → CD | AC → BD | AD → BC | BC → AD |
| BD → AC | CD → AB | | |

If |L| = k, then there are **2^k − 2** candidate association rules (ignoring L → ∅ and ∅ → L).

---

## Questions?
