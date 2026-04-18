# Lecture 11: Boosting, AdaBoost, Association Rule Mining, and Hyperparameter Tuning (Preview)

## 1. Recap: The Explainability Problem with Ensemble Methods

### Why Ask About Explanation?

When we combine ten different classifiers and take the majority vote to produce a decision, an obvious question arises. Is that decision correct, and more importantly, **can we explain it?**

### How Majority Voting and Bagging Produce Decisions

The majority voting, bagging, or ensemble approach simply takes the outputs of different experts (different classifiers) and combines the results. If someone asks you why you made a particular decision, answering "I received these decisions and I am combining them" is **not** an explanation.

The end user is not looking for explanations from ten different classifiers. They want a **short and useful explanation**. Ideally:

- Why did the first classifier make its decision?
- Why did the second classifier make its decision?
- Why did the third classifier make its decision?

Stacking ten such explanations together does not serve the user.

### Why the Problem Gets Worse with More Features

In a very simple example with only one feature, explaining a single classifier is manageable. However:

- With hundreds of features, explaining even a single classifier is hard.
- With millions of features, you cannot explain a single classifier at all.
- Combining many such classifiers produces an explanation that is far too long and too complex.

> **Key takeaway**: Ensemble methods are **weak in explanation**. Saying "we take the majority voting on the outputs of several classifiers" is not transparent, because you do not know anything about why C1, C2, C4, or C5 made their decisions.

**Transparency** is the goal we cannot easily reach with bagging-style ensembles.

---

## 2. Bagging Revisited: Where Bagging Falls Short

In bagging we sub-sample the training data to create different training sets:

- Sampling is done **with replacement**.
- Samples are treated **equally**, with no importance considered.
- Classifiers are also treated equally, with no importance weighting.

This is not perfect. In reality:

- Some samples are **difficult** to predict.
- Some samples are **very easy** to get right.
- That is normal.

### Analogy: Students in Different Courses

Just because students are studying machine learning, advanced machine learning, NLP, or reinforcement learning does not mean all students have the same ability across all topics. For example, a given student might be good at SVM, good at NLP, or strong in some other topic.

**Different classifiers can have different knowledge on different regions of the input space**, exactly like different students have different strengths across topics.

> **Insight**: A student who knows they are weak in Q-learning from the start should not spend equal time on every topic. They should focus on weak areas. Reviewing material you already know 100 times does not improve your Q-learning.

---

## 3. Boosting: Core Idea

Boosting is an **iterative procedure that adapts to the changing distribution of training data**.

The recipe:

1. Train a classifier on a sample of the training data.
2. Apply that classifier on the **full original training data**, not on the test set.
3. Inspect which samples the classifier missed.
4. **Increase** the weight of misclassified samples so they are more likely to be selected next round.
5. **Decrease** the weight of correctly classified samples.
6. Repeat. At each step the distribution of training data changes.

> **Important**: We test the newly trained classifier on the **training** data, not on the test set. The purpose is to find out which training samples our classifier is weak on so we can focus on them next.

### The Reinforcement Learning Study Analogy

Suppose you are studying reinforcement learning and you have time before the final exam. Today you start reviewing.

- **Planned schedule**: Review all material ten times.
- **First review (Round 1)**: You think you are good in all topics, so you review everything equally. Then you try some sample questions.
- **Self assessment**: You realize you do not know Q-learning, you do not know deep Q-learning, you do not know SARSA, you do not know TD. But you know the definition of reinforcement learning.
- **Second review (Round 2)**: Since your brain is already good on the definition parts, there is no need to review them. Focus more on your weak points: TD, SARSA, Q-learning. Then test yourself again on all topics.
- **After Round 2**: If you realize that your Q-learning improved, you can take a day off on it. There is no need to review it again, because your brain can degrade something if you focus on it too much. Continue to improve your ability on all topics.

Each round changes the **distribution of topics you study**. Round 1 treats every topic equally. Round 2 downweights definitions and upweights TD, SARSA, and Q-learning. Round 3 adjusts again based on what improved and what did not.

### What Is the Difference Between Bagging and Boosting? Weights.

In bagging, in all review steps, you choose topics at random. It is entirely possible that even if you are weak in Q-learning, you review ten times **without ever touching Q-learning**. Weights fix this.

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Sampling | Random with replacement, uniform probability | Sampling with replacement, probabilities driven by weights |
| Sample weights | All equal, never change | Updated each round based on miss or hit |
| Classifier weights | All equal in majority vote | Weighted by classifier accuracy (α_i) |
| Cross-round adaptation | None | Distribution changes each round to focus on difficult samples |
| Risk | May miss important hard cases | Focused, but sensitive to noisy labels |

---

## 4. Changing the Weights: How and Why

Initially, all N records (the original samples) are assigned equal weights. There is no difference between the topics.

**Unlike bagging**, in boosting the weights may change at the end of each round:

- Records that are **misclassified** have their weights **increased**.
- Records that are correctly classified have their weights **decreased**.

If we miss Q-learning, its weight must be increased. If we remember the definition of reinforcement learning, its weight decreases.

### Example Progression Across Three Rounds

Suppose the original set has samples indexed 1 through 5.

- **Round 1**: Everything is sampled uniformly at random with replacement. Every sample has an equal chance to appear.
- **Round 2**: The lecturer notes that the probability of sample 4 effectively appears **three times** because Round 1 misclassified it. A concrete round-2 sample from the lecturer's example is `{4, 5, 5, 4}`. Sample 4 is drawn more often now because its weight was increased after Round 1.
- **Round 3**: In the lecturer's example the model now correctly classifies sample 4, so the weight of 4 is **decreased** for Round 3. The probability of selecting sample 5 is **increased** (because it is now the difficult one), and for the other samples the probability is decreased.

### From Weights to Actual Sampling: The Bag Analogy

Imagine samples 1, 2, 3, 4, 5 inside a bag:

- When all weights are equal, putting your hand into the bag gives each sample an equal chance of being drawn. Return the sample after each draw, and repeat.
- Once the weights change, probabilities are no longer equal. Some samples are now more likely to be drawn than others.

**Implementation question**: how do we reflect a higher probability physically?

**Simplest answer**: duplicate the entries. If one sample should be drawn with probability 25% and another with probability 35%, duplicate the entries in that ratio inside the bag. Assigning the weight is conceptually important. The implementation question is how we use the weights during selection.

*(added example)* If we have four items A, B, C, D with weights 0.1, 0.2, 0.3, 0.4, one simple implementation is to put 1 copy of A, 2 of B, 3 of C, and 4 of D into the bag and draw uniformly. In code, we usually just call a weighted sampler:

```python
import numpy as np
samples = ["A", "B", "C", "D"]
weights = [0.1, 0.2, 0.3, 0.4]
draw = np.random.choice(samples, size=1, p=weights)
```

---

## 5. AdaBoost (AdaBoost.M1)

In bagging, we do not consider any importance for any sample, and we do not consider any importance weight for the classifiers either. In **AdaBoost (or AdaBoost.M1)** we consider the weight **both** for the samples to be selected **and** for the classifiers themselves.

### Notation

- **N**: number of samples in the original training set D.
- **K**: number of boosting rounds.
- **C_i**: the base classifier trained in round i.
- **D_i**: the training set sampled for round i.
- **w_j^i**: the weight of sample j entering round i.
- **α_i**: the importance of classifier C_i.
- **Z_i**: the normalization factor (sum of all current weights) for round i.

### Error Rate for Each Base Classifier

For each classifier C_i, we compute its error rate on the **original training set** D:

$$ \varepsilon_i = \sum_{j=1}^{N} w_j^i \cdot \mathbb{1}[C_i(x_j) \neq y_j] $$

Reading this formula:

- `w_j^i` is the weight for sample j under classifier i.
- The indicator $\mathbb{1}[C_i(x_j) \neq y_j]$ equals **0** when classifier i predicts the correct label for x_j (that is, $C_i(x_j) = y_j$) and **1** otherwise.
- The error rate is therefore the weighted average of these indicators across all N samples.

### Classifier Importance (α_i)

Based on the error rate, we define the importance weight for classifier C_i:

$$ \alpha_i = \frac{1}{2} \ln\left(\frac{1 - \varepsilon_i}{\varepsilon_i}\right) $$

Behavior of α_i:

- Small ε_i (accurate classifier) leads to large positive α_i.
- ε_i close to 0.5 leads to α_i close to 0.
- ε_i greater than 0.5 would produce a **negative** α_i, which is why the safeguard below kicks in.

### Updating Sample Weights

After round i, we update each sample's weight for the next round:

$$ w_j^{i+1} = \frac{w_j^i}{Z_i} \times \begin{cases} e^{-\alpha_i} & \text{if } C_i(x_j) = y_j \text{ (correct)} \\ e^{+\alpha_i} & \text{if } C_i(x_j) \neq y_j \text{ (misclassified)} \end{cases} $$

- Correct prediction: `w_j` in the next round is **decreased** by the factor $e^{-\alpha_i}$.
- Misclassification: `w_j` in the next round is **increased** by the factor $e^{+\alpha_i}$.

`Z_i` is the sum of all current weights and is used so that the updated weights still form a valid probability distribution.

### Safeguard: Reject Useless Classifiers

If any intermediate round produces an error rate **greater than 0.5**, the classifier is worse than random guessing. At that step:

- We do **not** update the weights based on this rubbish classifier.
- We **reset** all weights to 1/N.
- We go back to resampling and continue.

> **Reason**: a classifier worse than random is not trustworthy enough to drive weight updates. Using its decisions to change the distribution would move the ensemble in the wrong direction.

### Full AdaBoost Algorithm

1. Initialize weights: $w_j^1 = 1/N$ for all $j = 1, \dots, N$.
2. Let K be the number of boosting rounds. For $i = 1$ to $K$:
   1. Create training set $D_i$ by sampling with replacement from D according to the weights $w^i$.
   2. Train the base classifier $C_i$ on $D_i$.
   3. Apply $C_i$ to **all** of the original training set D (not just $D_i$).
   4. Compute the error rate $\varepsilon_i$ as the weighted average of misclassifications on the samples.
   5. If $\varepsilon_i > 0.5$: reset weights to 1/N, discard this classifier, go back to step 2.1.
   6. Otherwise, compute $\alpha_i = \frac{1}{2}\ln\frac{1-\varepsilon_i}{\varepsilon_i}$.
   7. Update the weights $w_j^{i+1}$ according to the rule above, and normalize by $Z_i$.
3. Return the weighted ensemble $\{(C_i, \alpha_i)\}_{i=1}^{K}$.

### Pseudocode *(reconstructed)*

```python
import numpy as np

def adaboost_m1(X, y, base_classifier_cls, K):
    N = len(X)
    w = np.ones(N) / N                      # initial equal weights
    classifiers, alphas = [], []

    i = 0
    while i < K:
        # Step 2.1: sample D_i with replacement using weights w
        idx = np.random.choice(N, size=N, replace=True, p=w)
        X_i, y_i = X[idx], y[idx]

        # Step 2.2: train base classifier
        C_i = base_classifier_cls().fit(X_i, y_i)

        # Step 2.3, 2.4: error on the original D, weighted by w
        pred = C_i.predict(X)
        miss = (pred != y).astype(float)
        eps = np.sum(w * miss)

        # Step 2.5: safeguard against weak classifier
        if eps > 0.5:
            w = np.ones(N) / N
            continue

        # Step 2.6: classifier importance
        alpha = 0.5 * np.log((1 - eps) / eps)

        # Step 2.7: weight update and normalization
        factor = np.where(miss == 1.0, np.exp(alpha), np.exp(-alpha))
        w = w * factor
        w = w / np.sum(w)                   # divide by Z_i

        classifiers.append(C_i)
        alphas.append(alpha)
        i += 1

    return classifiers, alphas
```

---

## 6. Weighted Majority Voting

In **simple majority voting**, we treat all classifiers equally. Each gets one vote.

In **weighted majority voting**, we consider the importance of each classifier. A stronger classifier counts more than a weaker classifier.

### Everyday Analogy

If one of the experts is the best student ever, graduated from the best university in the world, and has many scientific papers, while the others do not have such credentials, then we should put more weight on that expert. This matches how we weigh opinions in life. When you ask different people a question, you give more weight to the one you are sure is experienced in that area. AdaBoost does the same by plugging each classifier's rate into the majority-voting process.

### Final Prediction

For a binary problem with labels in $\{-1, +1\}$:

$$ H(x) = \mathrm{sign}\left(\sum_{i=1}^{K} \alpha_i \, C_i(x)\right) $$

For a multi-class problem with label set $\mathcal{Y}$ *(reconstructed formula based on the lecturer's verbal description)*:

$$ H(x) = \arg\max_{y \in \mathcal{Y}} \sum_{i=1}^{K} \alpha_i \cdot \mathbb{1}[C_i(x) = y] $$

*(additional example)* Suppose K = 3, classifier alphas are $\alpha_1 = 0.42$, $\alpha_2 = 0.65$, $\alpha_3 = 0.92$, and for a test point x we get $C_1(x) = +1$, $C_2(x) = -1$, $C_3(x) = +1$. Then:

$$ H(x) = \mathrm{sign}(0.42 - 0.65 + 0.92) = \mathrm{sign}(0.69) = +1 $$

---

## 7. Worked Example: One-Dimensional AdaBoost

We have **one-dimensional data** with one feature and one target label. Classification is based on a **threshold**. We run **three rounds**.

### Setup

- For each round, we train a **decision stump** (a single threshold classifier).
- For each classifier we compute an error rate and, from it, an importance rate $\alpha_i$.

### Round 1

- All weights equal $1/N$.
- **Classifier C_1**: threshold 0.75. All samples with feature value less than 0.75 are labeled as $-1$, all others as $+1$.
- Apply $C_1$ on the original data, compare to the true labels, compute $\varepsilon_1$, then compute $\alpha_1$.

### Round 2

- Weights updated based on Round 1. Missed samples get higher weights, correctly classified samples get lower weights.
- **Classifier C_2**: a different threshold. In the lecturer's example, the threshold labels everything as $+1$ (meaning the chosen threshold placed all observed samples on one side).
- Compute $\varepsilon_2$ and $\alpha_2$.

### Round 3

- Weights updated based on Round 2.
- **Classifier C_3**: yet another threshold, giving a different decision boundary.
- Compute $\varepsilon_3$ and $\alpha_3$.

### Final Decision

$$ H(x) = \mathrm{sign}(\alpha_1 C_1(x) + \alpha_2 C_2(x) + \alpha_3 C_3(x)) $$

*(reconstructed numeric example)* Given 10 samples with feature values in $[0, 1]$, suppose the true labels are mixed. At Round 1 with threshold 0.75, the first classifier mispredicts 3 of the 10 equally weighted samples, giving $\varepsilon_1 = 0.3$ and $\alpha_1 = \frac{1}{2}\ln\frac{0.7}{0.3} \approx 0.42$. The three missed samples have their weights multiplied by $e^{0.42} \approx 1.52$ before normalization, making them substantially more likely to be sampled in Round 2.

---

## 8. Association Rule Mining

### Motivation and Formal Goal

The arrangement of items in a store, and the way they are placed in different locations, is one kind of information we can study. Association rule mining asks a precise question.

> **Goal**: given a set of transactions, find rules that predict the occurrence of an item based on the occurrences of other items in the transaction.

> **Classic use case**: market basket analysis. If customers who buy milk and diapers also tend to buy beer, a store can organize product placement to support that pattern.

### Market-Basket Transaction Example

| TID | Items |
|-----|-------|
| 1 | Bread, Milk |
| 2 | Bread, Diaper, Beer, Eggs |
| 3 | Milk, Diaper, Beer, Coke |
| 4 | Bread, Milk, Diaper, Beer |
| 5 | Bread, Milk, Diaper, Coke |

This dataset has 5 transactions and 6 unique items: Bread, Milk, Diaper, Beer, Eggs, Coke.

### Example Association Rules

- `{Diaper} → {Beer}`
- `{Milk, Bread} → {Eggs, Coke}`
- `{Beer, Bread} → {Milk}`
- `{Milk, Diaper} → {Beer}`

> **Important caveat**: Implication means **co-occurrence**, not causality. An association rule says items tend to appear together, not that buying X causes buying Y.

### Core Concepts

- **Itemset**: a collection of one or more items.
  - Example: `{Milk, Bread, Diaper}`.
- **k-itemset**: an itemset that contains exactly k items.
  - Example: `{Milk, Bread, Diaper}` is a **3-itemset**.
- **Support count** ($\sigma$): the frequency of occurrence of an itemset across all transactions.
  - Example: $\sigma(\{\text{Milk, Bread, Diaper}\}) = 2$ (transactions 4 and 5).
- **Support (s)**: the fraction of transactions that contain an itemset.
  $$ s(X) = \frac{\sigma(X)}{|T|} $$
  where $|T|$ is the total number of transactions.
  - Example: $s(\{\text{Milk, Bread, Diaper}\}) = 2/5 = 0.4$.
- **Frequent itemset**: an itemset whose support is **greater than or equal to** a user-defined **minsup** threshold.
- **Association rule**: an implication of the form $X \Rightarrow Y$, where X and Y are itemsets.
  - Example: `{Milk, Diaper} → {Beer}`.

### Rule Evaluation: Support and Confidence

Association rules are evaluated with two metrics.

- **Support (s)** of the rule: the fraction of transactions that contain both X and Y.
  $$ s(X \Rightarrow Y) = \frac{\sigma(X \cup Y)}{|T|} $$
- **Confidence (c)**: how often items in Y appear in transactions that also contain X.
  $$ c(X \Rightarrow Y) = \frac{\sigma(X \cup Y)}{\sigma(X)} $$

### Worked Example: {Milk, Diaper} ⇒ {Beer}

Using the transactions above:

$$ s = \frac{\sigma(\text{Milk, Diaper, Beer})}{|T|} = \frac{2}{5} = 0.4 $$

$$ c = \frac{\sigma(\text{Milk, Diaper, Beer})}{\sigma(\text{Milk, Diaper})} = \frac{2}{3} \approx 0.67 $$

### Association Rule Mining Task

Given a set of transactions T, the goal is to find **all rules** having:

- support $\geq$ **minsup** threshold, **and**
- confidence $\geq$ **minconf** threshold.

### Brute-Force Approach (and Why It Fails)

The naive algorithm:

1. List all possible association rules.
2. Compute the support and confidence for each rule.
3. Prune rules that fail the minsup and minconf thresholds.

This is **computationally prohibitive**.

### Computational Complexity

Let **d** be the number of unique items.

- Total number of itemsets: $2^d$.
- Total number of possible association rules:

$$ R = \sum_{k=1}^{d-1}\left[\binom{d}{k} \times \sum_{j=1}^{d-k}\binom{d-k}{j}\right] = 3^d - 2^{d+1} + 1 $$

**For d = 6, R = 602 rules.** Already too many for any real system. Now consider how many items appear in a real retail catalog.

*(added worked growth example)*

| d (items) | Itemsets $2^d$ | Association rules $3^d - 2^{d+1} + 1$ |
|-----------|----------------|----------------------------------------|
| 3 | 8 | 12 |
| 6 | 64 | 602 |
| 10 | 1,024 | 57,002 |
| 20 | ~1.05 million | ~3.48 billion |

### Observation: Binary Partitions of a Frequent Itemset Share Support

Consider the frequent itemset $\{\text{Milk, Diaper, Beer}\}$. All binary partitions of this itemset produce rules with **identical support** but potentially **different confidence**:

- `{Milk, Diaper} → {Beer}` (s = 0.4, c = 0.67)
- `{Milk, Beer} → {Diaper}` (s = 0.4, c = 1.0)
- `{Diaper, Beer} → {Milk}` (s = 0.4, c = 0.67)
- `{Beer} → {Milk, Diaper}` (s = 0.4, c = 0.67)
- `{Diaper} → {Milk, Beer}` (s = 0.4, c = 0.5)
- `{Milk} → {Diaper, Beer}` (s = 0.4, c = 0.5)

**Observations:**

- All six rules are binary partitions of the same itemset `{Milk, Diaper, Beer}`.
- Rules originating from the same itemset have **identical support** but can have **different confidence**.
- Therefore, we can **decouple** the support and confidence requirements. Find frequent itemsets first (which fixes support), then worry about confidence during rule generation.

### Two-Step Approach to Mining Association Rules

The decoupling observation leads directly to a two-step algorithm.

1. **Frequent Itemset Generation**: generate all itemsets whose support $\geq$ minsup.
2. **Rule Generation**: generate high confidence rules from each frequent itemset, where each rule is a binary partition of a frequent itemset.

> **Caveat**: Frequent itemset generation is still computationally expensive if done naively. The Apriori principle makes it tractable.

---

## 9. Frequent Itemset Generation

### The Itemset Lattice

For a universe of d items, the space of itemsets forms a **lattice**. At the top is `null` (the empty set). Below it sit the d singleton 1-itemsets, then the $\binom{d}{2}$ 2-itemsets, and so on, down to the single d-itemset containing all items.

*(added example)* For items `{A, B, C, D, E}` the lattice has:

- 1 empty set at the top.
- 5 one-itemsets: {A}, {B}, {C}, {D}, {E}.
- 10 two-itemsets: {AB}, {AC}, {AD}, {AE}, {BC}, {BD}, {BE}, {CD}, {CE}, {DE}.
- 10 three-itemsets.
- 5 four-itemsets.
- 1 five-itemset: {ABCDE}.

Total candidates: $2^5 = 32$, or $2^d$ in general.

### Brute-Force Frequent Itemset Generation

1. Treat each itemset in the lattice as a **candidate** frequent itemset.
2. Scan the database of N transactions to count the support of every candidate.
3. Match each transaction against every candidate.
4. Eliminate any candidate whose support is below minsup.

**Complexity**: $O(NMw)$, where:

- **N** = number of transactions.
- **M** = number of candidate itemsets, equal to $2^d$.
- **w** = average transaction width (number of items per transaction).

> Expensive because $M = 2^d$ dominates.

### Strategies to Speed Up Frequent Itemset Generation

1. **Reduce the number of candidates (M)**.
   - Complete search enumerates all $2^d$ candidates.
   - Use pruning techniques (Apriori) to reduce M dramatically.
2. **Reduce the number of transactions scanned (N)**.
   - Shrink the effective N as itemset size grows. A transaction that lacks any frequent item cannot contain any frequent k-itemset.
3. **Reduce the number of comparisons (NM)**.
   - Use efficient data structures (hash trees, tries) to store candidates and transactions so that we do not match every candidate against every transaction.

---

## 10. Apriori Algorithm

### The Apriori Principle

> **If an itemset is frequent, then all of its subsets must also be frequent.**

Equivalently, if a subset is **infrequent**, then every **superset** that contains it is also infrequent and can be pruned from the search without ever counting its support.

### Anti-Monotone Property of Support

The Apriori principle holds because support is **anti-monotone** with respect to set inclusion:

$$ \forall X, Y : (X \subseteq Y) \Rightarrow s(X) \geq s(Y) $$

- Adding an item to an itemset can only keep the support the same or make it smaller, never larger.
- This is known as the **anti-monotone** property of support.

### Illustrating the Principle

Suppose we have items $\{A, B, C, D, E\}$ and we discover that itemset $\{A, B\}$ is **infrequent**. Then every superset that contains $\{A, B\}$ must also be infrequent and can be pruned:

- 3-itemsets: $\{A, B, C\}$, $\{A, B, D\}$, $\{A, B, E\}$.
- 4-itemsets: $\{A, B, C, D\}$, $\{A, B, C, E\}$, $\{A, B, D, E\}$.
- 5-itemset: $\{A, B, C, D, E\}$.

This single pruning removes 7 candidates from consideration.

### Worked Example with minsup = 3

Transactions:

| TID | Items |
|-----|-------|
| 1 | Bread, Milk |
| 2 | Beer, Bread, Diaper, Eggs |
| 3 | Beer, Coke, Diaper, Milk |
| 4 | Beer, Bread, Diaper, Milk |
| 5 | Bread, Coke, Diaper, Milk |

**Step 1: 1-itemset counts**

| Item | Count |
|------|-------|
| Bread | 4 |
| Coke | 2 |
| Milk | 4 |
| Beer | 3 |
| Diaper | 4 |
| Eggs | 1 |

With **minsup = 3**, **Coke** (count = 2) and **Eggs** (count = 1) are pruned. Frequent 1-itemsets: $\{\text{Bread, Milk, Beer, Diaper}\}$.

**Step 2: Candidate count comparison**

- Without pruning (enumerate subsets of size 1, 2, 3): $\binom{6}{1} + \binom{6}{2} + \binom{6}{3} = 6 + 15 + 20 = 41$.
- With support-based pruning (exclude Coke and Eggs from all higher-level candidates): $6 + 6 + 4 = 16$.

**Step 3: 2-itemset counts** (no candidates involve Coke or Eggs):

| Itemset | Count |
|---------|-------|
| {Bread, Milk} | 3 |
| {Bread, Beer} | 2 |
| {Bread, Diaper} | 3 |
| {Milk, Beer} | 2 |
| {Milk, Diaper} | 3 |
| {Beer, Diaper} | 3 |

Frequent 2-itemsets (count $\geq$ 3): $\{\text{Bread, Milk}\}$, $\{\text{Bread, Diaper}\}$, $\{\text{Milk, Diaper}\}$, $\{\text{Beer, Diaper}\}$.

**Step 4: 3-itemset counts**:

| Itemset | Count |
|---------|-------|
| {Beer, Diaper, Milk} | 2 |
| {Beer, Bread, Diaper} | 2 |
| {Bread, Diaper, Milk} | 2 |
| {Beer, Bread, Milk} | 1 |

None meet minsup = 3, so there are no frequent 3-itemsets. The algorithm halts at k = 2 for this example.

### The Apriori Algorithm

**Notation (per the slide convention)**:

- $F_k$: set of **frequent** k-itemsets (those that meet minsup).
- $L_k$: set of **candidate** k-itemsets (those still to be evaluated).

> **Notation note**: These slides use $L_k$ for candidates and $F_k$ for frequent itemsets. Some textbooks (Agrawal and Srikant) invert this, using $C_k$ for candidates and $L_k$ for large (frequent) itemsets. Read carefully whenever you switch sources.

**Algorithm:**

1. Let $k = 1$.
2. Generate $F_1$, the set of frequent 1-itemsets, by counting each item's support and keeping those with support $\geq$ minsup.
3. Repeat until $F_k$ is empty:
   1. **Candidate Generation**: Generate $L_{k+1}$ from $F_k$.
   2. **Candidate Pruning**: Remove any candidate in $L_{k+1}$ whose k-subsets are not all in $F_k$ (some subset is infrequent, so by the Apriori principle the candidate cannot be frequent).
   3. **Support Counting**: Scan the database to count the support of each surviving candidate in $L_{k+1}$.
   4. **Candidate Elimination**: Drop candidates whose support is below minsup. What remains becomes $F_{k+1}$.
4. Return $\bigcup_k F_k$.

### Pseudocode *(reconstructed)*

```python
def apriori(transactions, minsup_count):
    items = sorted({item for t in transactions for item in t})

    F = {}                                   # F[k] = frequent k-itemsets
    F[1] = [frozenset([i]) for i in items
            if sum(1 for t in transactions if i in t) >= minsup_count]

    k = 1
    while F[k]:
        L_next = generate_candidates(F[k])                    # step 3.1
        L_next = [c for c in L_next if all_subsets_frequent(c, F[k])]  # step 3.2

        counts = {c: sum(1 for t in transactions if c.issubset(t))
                  for c in L_next}                             # step 3.3
        F[k + 1] = [c for c, cnt in counts.items()
                    if cnt >= minsup_count]                    # step 3.4
        k += 1

    return [fs for level in F.values() for fs in level]
```

---

## 11. Candidate Generation Methods

Several schemes exist to generate $L_{k+1}$ from $F_k$. They differ in how many unnecessary candidates they produce.

### Method 1: Brute-Force Method

Enumerate every k-itemset over the frequent 1-items.

*(example)* Given frequent 1-items $\{\text{Beer, Bread, Cola, Diapers, Eggs, Milk}\}$ (hypothetically, with Cola and Eggs kept frequent), brute-force candidate 3-itemset generation produces all $\binom{6}{3} = 20$ triplets. Each is then run through Apriori pruning and support counting. In the running example only $\{\text{Bread, Diapers, Milk}\}$ survives. Most of the work is wasted.

### Method 2: $F_{k-1} \times F_1$ Method

Merge frequent $(k-1)$-itemsets with frequent 1-items. More targeted than brute-force, but still generates some bad candidates.

*(example)* With $F_2 = \{\{\text{Beer, Diapers}\}, \{\text{Bread, Diapers}\}, \{\text{Bread, Milk}\}, \{\text{Diapers, Milk}\}\}$ and $F_1 = \{\text{Beer, Bread, Diapers, Milk}\}$, candidate 3-itemsets include:

- $\{\text{Beer, Bread, Diapers}\}$
- $\{\text{Beer, Diapers, Milk}\}$
- $\{\text{Bread, Diapers, Milk}\}$
- $\{\text{Beer, Bread, Milk}\}$

After candidate pruning (check that each size-2 subset lies in $F_2$), only $\{\text{Bread, Diapers, Milk}\}$ remains. Some candidates (for instance $\{\text{Beer, Bread, Milk}\}$) are unnecessary because their subsets are infrequent.

Total candidates considered over all levels: **6 + 6 + 4 = 16** (versus 41 for brute-force).

### Method 3: $F_{k-1} \times F_{k-1}$ Method (Shared Prefix)

Merge two frequent $(k-1)$-itemsets only when their **first $(k-2)$ items are identical**. Itemsets must be kept in a consistent lexicographic order for this to work.

*(example)* With $F_3 = \{ABC, ABD, ABE, ACD, BCD, BDE, CDE\}$:

- Merge(**AB**C, **AB**D) = **AB**CD. Valid.
- Merge(**AB**C, **AB**E) = **AB**CE. Valid.
- Merge(**AB**D, **AB**E) = **AB**DE. Valid.
- Do **not** Merge(**A**BD, **A**CD) because they share only a prefix of length 1 instead of the required length 2.

Candidate set: $L_4 = \{ABCD, ABCE, ABDE\}$.

### Candidate Pruning for Method 3

Given $F_3 = \{ABC, ABD, ABE, ACD, BCD, BDE, CDE\}$ and $L_4 = \{ABCD, ABCE, ABDE\}$:

- **Prune ABCE** because ACE and BCE are not in $F_3$ (infrequent).
- **Prune ABDE** because ADE is not in $F_3$ (infrequent).
- Keep **ABCD** because all its 3-subsets (ABC, ABD, ACD, BCD) are in $F_3$.

**After candidate pruning**: $L_4 = \{ABCD\}$.

### Method 3 Applied to the Running Example

Merging pairs of frequent 2-itemsets $\{\{\text{Beer, Diapers}\}, \{\text{Bread, Diapers}\}, \{\text{Bread, Milk}\}, \{\text{Diapers, Milk}\}\}$ (sorted lexicographically) produces the single candidate 3-itemset $\{\text{Bread, Diapers, Milk}\}$ (from Merge(Bread-Diapers, Bread-Milk)). After pruning it remains, and support counting then determines whether it is frequent.

**Total candidates considered across methods**:

| Method | Candidates across all levels |
|--------|------------------------------|
| Brute-force | 41 |
| $F_{k-1} \times F_1$ | 16 |
| $F_{k-1} \times F_{k-1}$ (shared prefix) | 13 (6 + 6 + 1) |

### Method 4: Alternate $F_{k-1} \times F_{k-1}$ Method (Suffix-Prefix Match)

Merge two frequent $(k-1)$-itemsets when the **last $(k-2)$ items of the first** are identical to the **first $(k-2)$ items of the second**.

*(example)* With $F_3 = \{ABC, ABD, ABE, ACD, BCD, BDE, CDE\}$:

- Merge(A**BC**, **BC**D) = A**BC**D
- Merge(A**BD**, **BD**E) = A**BD**E
- Merge(A**CD**, **CD**E) = A**CD**E
- Merge(B**CD**, **CD**E) = B**CD**E

Candidate set: $L_4 = \{ABCD, ABDE, ACDE, BCDE\}$.

### Candidate Pruning for Method 4

- **Prune ABDE** because ADE is infrequent.
- **Prune ACDE** because ACE and ADE are infrequent.
- **Prune BCDE** because BCE is infrequent.
- Keep **ABCD**.

**After candidate pruning**: $L_4 = \{ABCD\}$.

### Support Counting of Candidate Itemsets

Once candidates survive generation and pruning, we still need to count their support in the transaction database.

- Scan the database of transactions.
- For each candidate, check every transaction and increment its support count whenever the candidate is a subset of the transaction.
- Matching every candidate against every transaction is an **expensive** operation, which is why efficient data structures (hash trees, tries) are used in practice.

**Running example**:

Transactions:

| TID | Items |
|-----|-------|
| 1 | Bread, Milk |
| 2 | Beer, Bread, Diaper, Eggs |
| 3 | Beer, Coke, Diaper, Milk |
| 4 | Beer, Bread, Diaper, Milk |
| 5 | Bread, Coke, Diaper, Milk |

Candidate itemsets:

- $\{\text{Beer, Diaper, Milk}\}$
- $\{\text{Beer, Bread, Diaper}\}$
- $\{\text{Bread, Diaper, Milk}\}$
- $\{\text{Beer, Bread, Milk}\}$

Each candidate is checked against each of the five transactions to get its count.

---

## 12. Rule Generation

Given a frequent itemset $L$, find all non-empty subsets $f \subset L$ such that the rule $f \Rightarrow L \setminus f$ satisfies the minimum confidence requirement.

### Example: Candidate Rules from $\{A, B, C, D\}$

If $\{A, B, C, D\}$ is a frequent itemset, the candidate rules are all binary partitions where both antecedent and consequent are non-empty:

| | | | |
|---|---|---|---|
| ABC → D | ABD → C | ACD → B | BCD → A |
| A → BCD | B → ACD | C → ABD | D → ABC |
| AB → CD | AC → BD | AD → BC | BC → AD |
| BD → AC | CD → AB | | |

### Number of Candidate Rules per Frequent Itemset

If $|L| = k$, then there are $2^k - 2$ candidate association rules from $L$. The subtracted 2 excludes the trivial rules $L \Rightarrow \emptyset$ and $\emptyset \Rightarrow L$.

*(added example)* For $|L| = 4$, there are $2^4 - 2 = 14$ candidate rules, matching the table above. For $|L| = 5$, there are $2^5 - 2 = 30$ candidate rules from a single frequent itemset.

> **Recall**: all rules from the same frequent itemset share the same support, so only confidence needs to be computed for each candidate rule. Rules with confidence $\geq$ minconf are retained.

---

## 13. Hyperparameter Tuning (Preview for the Next Two Lectures)

> **Course note**: For the next two lectures, you need to cover hyperparameter tuning.

### Definition

**Hyperparameter tuning**: finding the best possible values for the hyperparameters of any algorithm. Hyperparameters must be set **before** training.

### Hyperparameters vs. Parameters

**Hyperparameters** are values **we give** to the algorithm. **Parameters** are learned **during training**. Different algorithms have different parameters and different hyperparameters.

| Algorithm | Hyperparameter (set by us) | Parameter (learned) |
|-----------|----------------------------|---------------------|
| K-means | Number of clusters K | Final cluster centroids (cluster representatives) |
| Linear regression | Regularization strength | Weights and bias |
| Neural network | Learning rate, batch size, layer sizes | Weights and biases of every layer |
| SVM | Kernel choice, C, $\gamma$ | Support vectors and their dual coefficients |

> **Why this distinction matters**: If you confuse them, you might try to "learn" the number of clusters during K-means training or "set" the weights of a neural network by hand. Both are wrong.

### Why Hyperparameter Tuning Matters

Finding good values for the hyperparameters **ensures optimal model performance**. A well-designed model with poorly chosen hyperparameters can easily underperform a simpler model with well-chosen ones.

### Four Approaches

#### 1. Manual Search

Try values one by one. Based on intuition and experience, set a hyperparameter, train, observe, adjust, repeat. Slow and labor-intensive but useful when you have strong intuition about the model.

#### 2. Grid Search

1. Define the set of possible values for **each** hyperparameter.
2. Train the model with **every combination** of those values.
3. Choose the combination with the best validation score.

#### 3. Random Search

Similar to grid search in that you define candidate values, but instead of trying every combination you pick **random combinations** from the predefined values. Often more efficient in high-dimensional hyperparameter spaces.

#### 4. Literature and Best Practices

Commonly used in scientific or academic work. Refer to the literature, or use best practices reported by other researchers.

> **Why reviewers ask**: When you publish a paper, reviewers often ask why you chose particular hyperparameter values and why you did not use other values. A valid response is that those values are popular among other researchers. For example, for batch size you can in principle use any value, but most papers use **32 or 64**, because most people use **32 or 64**.

### Minimal Code Examples *(added)*

```python
# Grid search with scikit-learn
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": [0.001, 0.01, 0.1, 1],
}

grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
```

```python
# Random search with scikit-learn
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {
    "C": loguniform(1e-2, 1e2),
    "gamma": loguniform(1e-4, 1e0),
    "kernel": ["linear", "rbf"],
}

rand = RandomizedSearchCV(SVC(), param_dist, n_iter=50, cv=5, random_state=0)
rand.fit(X_train, y_train)
print("Best params:", rand.best_params_)
```

---

## 14. Summary

- **Ensemble methods** combine several classifiers but are **weak in explanation**. The weakness grows with the number of features and the number of classifiers.
- **Bagging** samples uniformly with replacement and votes uniformly, ignoring which samples are hard and which classifiers are good.
- **Boosting** adapts the training distribution across rounds. Misclassified samples get higher weight, correctly classified samples get lower weight.
- **AdaBoost (AdaBoost.M1)** is a weighted boosting scheme. Weights apply to **both** samples (for selection) and classifiers (for voting).
  - Error rate: $\varepsilon_i = \sum_j w_j^i \cdot \mathbb{1}[C_i(x_j) \neq y_j]$.
  - Classifier importance: $\alpha_i = \frac{1}{2}\ln\frac{1-\varepsilon_i}{\varepsilon_i}$.
  - Weight update: multiply by $e^{-\alpha_i}$ if correct and by $e^{+\alpha_i}$ if wrong, then divide by $Z_i$.
  - If $\varepsilon_i > 0.5$, reset weights to $1/N$ and resample.
- **Weighted majority voting** generalizes the final ensemble decision so that stronger classifiers have more influence.
- **Association rule mining** discovers co-occurrence patterns, not causal relationships.
  - Core quantities: **itemset**, **k-itemset**, **support count** ($\sigma$), **support** ($s$), **frequent itemset**, **confidence** ($c$).
  - Brute-force rule enumeration is infeasible. For d items, there are $3^d - 2^{d+1} + 1$ possible rules (602 for d = 6).
  - **Observation**: rules from the same frequent itemset share support but differ in confidence, so support and confidence can be **decoupled**.
- **Two-step mining**: find frequent itemsets first, then generate rules from each frequent itemset.
- **Frequent itemset generation** has a brute-force cost of $O(NMw)$ with $M = 2^d$. Speed-ups come from reducing M, N, or NM.
- **Apriori principle**: if an itemset is frequent, all its subsets are frequent. Support is **anti-monotone**: $X \subseteq Y \Rightarrow s(X) \geq s(Y)$. Therefore, if any subset is infrequent, every superset can be pruned.
- **Apriori algorithm**: for each k, generate candidates $L_{k+1}$ from $F_k$, prune by subset-frequency, count support, keep the frequent ones as $F_{k+1}$, and iterate until $F_k$ is empty.
- **Candidate generation methods**: brute-force (all $\binom{d}{k}$), $F_{k-1} \times F_1$, $F_{k-1} \times F_{k-1}$ with shared prefix, and alternate $F_{k-1} \times F_{k-1}$ with suffix-prefix match. Efficiency ranking in the running example: 41 (brute-force) → 16 → 13.
- **Rule generation**: from each frequent itemset $L$ of size $k$, there are $2^k - 2$ candidate rules (all binary partitions excluding the trivial two). Keep rules with confidence $\geq$ minconf.
- **Hyperparameter tuning** is previewed for upcoming lectures. Hyperparameters are set before training. Parameters are learned during training. Four practical approaches: **manual**, **grid**, **random**, and **literature / best practices** (for example, batch sizes of 32 or 64 because they are widely used).
