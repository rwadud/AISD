# Lecture 10: Autoencoders for Anomaly Detection, One-Class SVM, and Ensemble Learning

*Adapted from course slides by Dr. Abbas Akkasi (CST8506, Winter 2026), originally based on Pang-Ning Tan's Data Mining course.*

## Autoencoders Recap

**Autoencoder**: a neural network architecture where the input is given to the network and the goal is to reconstruct it at the output. The architecture can be any type: MLP, RNN, or any other neural network.

> **Key property**: in an autoencoder, the input and the target output are exactly the same. This is different from translation or summarization, where input and output differ.

### Structure of an Autoencoder

```
Input x ──► [Encoder] ──► Hidden state h ──► [Decoder] ──► Reconstructed x̂
                          (lower dim)
```

- The **encoder** compresses the input into a lower dimensional representation called the hidden state `h`.
- The **decoder**, after training, reconstructs the input from this compressed representation.

### Why Use an Autoencoder?

#### 1. Data Compression / Dimensionality Reduction

The encoder decreases the dimension of the data.

- Example: input dimension = 10 million, hidden layer dimension = 100.
- Instead of storing data at 10 million dimensions, we only need to store it at 100 dimensions.
- After training, whenever we want to access a specific input, we only need the lower dimensional representation.

#### 2. Noise Reduction (Denoising Autoencoder)

**Training procedure for a standard autoencoder**:
- Input to model: the input sample.
- Target: the same input sample.

**Training procedure for a denoising autoencoder**:
- Suppose training data contains noise (noisy images, noisy voice, noisy text).
- You know what the clean data (without noise) looks like.
- Input to model: the **noisy** data.
- Target: the **clean** data (without noise).
- This trains the model to denoise the data, to remove the noise.

#### 3. Anomaly Detection

- Try to regenerate the input using the autoencoder.
- If the generated output is **different** from the real input:
  - That means the input contains noise, and our model denoised it.
  - Therefore the input sample is **anomalous**.

### Types of Autoencoders

- The most popular type has a hidden layer **smaller** than the input dimension.
- It is also possible to **increase** the size of the hidden layer.

### What Kind of Noise?

Any kind of noise: in images, voice, or text. Since we work with numbers, an image has pixel values; if for any reason some pixel values change (something added to different pixels), the sample is considered noisy.

> **Important application**: use autoencoders both to **detect anomalies** and to **denoise data**.

---

## One-Class SVM (OCSVM)

### Motivation

We already know SVM is a classifier that separates data from different classes. But for anomaly detection:
- We don't have labels, OR
- We have only **one label**, and having only one label is effectively the same as having no label.

**Question**: How can you use an SVM to separate data when there is no label?
**Answer**: That is what One-Class SVM does.

### Main Idea

Given some data, try to find a **boundary** around the normal data:
- Samples **inside** the boundary → treated as **normal**.
- Samples **outside** the boundary → considered **outliers**.

The kernel choice and the way we set the boundary all aim to enclose the majority of normal data within some boundary. That means we are separating this majority from something else, but we don't actually have that something, which is the problem.

### The Origin Trick

To solve the "no other class" problem, OCSVM uses the **origin trick**:
- Assume that the data point at the origin `(0, 0, 0, ..., 0)` is an outlier (not normal data).
- Try to separate the real samples from the origin.

That is how OCSVM works.

### Decision Function

$$f(x) = w \cdot \varphi(x) - \rho$$

Where:
- $\varphi(x)$ is the kernel.
- $\rho$ is a parameter that is learned.

For any data point $x$:
- If $f(x) > 0$ → **normal**.
- Otherwise → **outlier**.

This is an optimization problem similar to the others we've seen.

### Hyperparameters

You will probably work with these in one of your assignments or labs.

| Hyperparameter | Role |
|---|---|
| `nu` (ν) | Controls what portion of the data is considered normal. |
| `gamma` (γ) | Changes the shape/curvature of the boundary. |

**Examples of `nu`**:
- `nu = 0.05` → at most about 5% of the data is considered outliers (e.g., about 4 samples flagged as outliers).
- `nu = 0.2` → the number of flagged outliers is much higher.

### Is OCSVM Confined to SVM?

No. Everything in scikit-learn can be used similarly; you just apply it the same way.

```python
# (added) Minimal usage example in scikit-learn
from sklearn.svm import OneClassSVM

ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
ocsvm.fit(X_train)

preds = ocsvm.predict(X_test)         # +1 = normal, -1 = outlier
scores = ocsvm.decision_function(X_test)  # f(x) values
```

---

## Part 3 of the Course: Ensemble Learning (Classifier Fusion)

### The Core Idea

Instead of relying on one classifier, take the opinions/decisions of **multiple classifiers** and combine them.

### Formal Definition

**Ensemble learning**: construct a set of base classifiers learned from the training data, then predict the class label of test records by combining the predictions made by multiple classifiers (e.g., by taking a majority vote).

### Real-World Analogy

- If you have a health problem, you go to a physician → that is a single classifier prediction.
- If the problem is serious, you don't rely on one prediction. You consult others.
- Physician 1 says disease A, physician 2 says B, physician 3 says A, physician 4 says you are fine.
- You take a **majority vote** to see which decision most experts agree on.

### Voting Approaches

- Simple majority voting.
- Weighted majority voting (weighted according to accuracy or relevance).
- Heuristic weights for majority.

### General Approach

```
              D (Training Data)
                    │
       ┌────┬───────┴───────┬────┐
       ▼    ▼               ▼    ▼
      C₁   C₂    ...     C_{t-1}  C_t     (Step 1: Build Multiple Classifiers)
       │    │               │    │
       └────┴───────┬───────┴────┘
                    ▼
                   C*                       (Step 2: Combine Classifier Responses)
                    │
                    ▼
                Prediction
```

---

## Why Do Ensemble Methods Work?

### Setup

- 25 base classifiers (analogous to 25 physicians).
- Each has an error rate of $\varepsilon = 0.35$ (each misclassifies about 35% of the data).
- Majority vote of classifiers used for classification.

### Case 1: All Classifiers Identical

Instead of going to 100 different physicians, you go to the **same** physician 100 times; he or she will not change their opinion.

→ Error rate of the ensemble = $\varepsilon$ (0.35).

### Case 2: Classifiers Independent (Errors Uncorrelated)

They behave differently from each other. That means we have a **diverse** set of classifiers.

> **Key principle**: in ensemble learning, we should have diversity.

### Ensemble Error Rate Under Majority Voting

The ensemble is wrong when **more than half** of the base classifiers are wrong.

- With 25 classifiers: half is 12.5, so at least **13** must be wrong for the ensemble to be wrong.
- In that case, the probability of individual terms is on the order of $1.1 \times 10^{-6}$.

$$e_{\text{ensemble}} = \sum_{i=13}^{25} \binom{25}{i} \varepsilon^{i} (1-\varepsilon)^{25-i} = 0.06$$

Where:
- $\binom{25}{i}$ is the number of ways to select $i$ classifiers from 25.
- $\varepsilon^{i}$ is the probability of $i$ classifiers being wrong.
- $(1-\varepsilon)^{25-i}$ is the probability the rest are correct.
- We sum over all possibilities where the ensemble is wrong ($i = 13, 14, ..., 25$).

> **Takeaway**: compared to a single classifier error rate of 35%, the ensemble error rate drops to 6%. Even under the assumption that most classifiers in the pool could be wrong, the ensemble error is only 6%. That is why ensembles work. In theory, ensemble learning works, but it is **not guaranteed** that you always get improvement when you apply it.

### Clarification Questions From Class

**Q: For the first 12 classifiers, do we consider a threshold or error rate and separate them?**
A: No. For the first 12, assume they are correct (they predict the class label correctly). We want to know the probability that the ensemble predicts the wrong class.

**Q: What if we have more than two classes and the vote is split three ways?**
A: In this setup we only use majority voting. In the worst case, the ensemble error rate can be **much higher** than the error rate of a single classifier.

---

## Necessary Conditions for Ensemble Methods

Ensemble methods work better than a single base classifier if:

### 1. Base Classifiers Must Be Independent (Diverse)

You cannot go to one physician 100 times and claim you got 100 different opinions.

**Measuring diversity**:
- There are **more than 10 different metrics** in the classifier combination literature for evaluating diversity.
- **Simplest metric**: look at the performance of the classifiers. If performance differs, they are diverse.
- **Better**: regardless of the performance metric, if the **predictions** of classifiers on different samples differ, the classifiers are diverse.

### 2. Base Classifiers Must Perform Better Than Random Guessing

If you go to 100 physicians whose knowledge level is low, the ensemble may still fail. Having a specialty doesn't mean being an expert.

- Each individual classifier must have error rate **< 0.5** (better than random guessing for binary classification).

---

## Rationale for Ensemble Learning

Ensemble methods work best with **unstable base classifiers**.

**Unstable classifier**: sensitive to minor perturbations in the training set, due to *high model complexity*. If you change something during training, the results change a lot.

| Classifier | Stability | Reason |
|---|---|---|
| Unpruned Decision Tree | Unstable | If you change the data, the tree changes. |
| Artificial Neural Network (ANN) | Unstable | Everything depends on initial weight values. |
| KNN | Unstable | If you remove the nearest neighbor, the label can change. |
| Logistic Regression | Stable | |
| SVM | Stable | |

> Ensemble learning is **designed for unstable classifiers**.

---

## Bias-Variance Decomposition

### Intuition

Analogous to the problem of reaching a target $y$ by firing projectiles from $x$ (a regression problem). The error decomposes into three components:

- **Variance**: spread of $\hat{y}$ around $\hat{y}_{\text{avg}}$.
- **Bias**: distance between $\hat{y}_{\text{avg}}$ and $y_{\text{avg}}$.
- **Noise**: spread of $y$ around $y_{\text{avg}}$ (irreducible).

### Variance

If you change the projection (change the parameters of this device), it predicts different values. Examples:
- Change the initial weights of the network.
- Remove some samples.

The **average spread** of this difference, the spread of the prediction, is the **variance**.

### Bias

The difference between the **average of the true targets** and the **average of the predictions** is what we call **bias**.

### Ideal Case

- **No bias**: $\hat{y} = y$.
- **Low variance**: stability across runs.

> If every time you run the model it generates different results, after two runs you would never use it.

We want **low bias** AND **low variance**, but usually this is not possible, so we need a **trade-off**.

### Generalization Error

For classification, the generalization error of model $m$ is:

$$\text{gen.error}(m) = c_1 + \text{bias}(m) + c_2 \times \text{variance}(m)$$

Where:
- $c_1$ is an **irreducible error** we cannot remove. This exists because most classification algorithms use random values in some steps. For example, for a neural network, we never know the ideal initial weights.
- $c_2$ is a constant.

This is a bit different from test error; it is the overall error we expect on unseen data.

### Bias-Variance Trade-off and Overfitting

| Situation | Bias | Variance |
|---|---|---|
| Overfitting | Low | High |
| Underfitting | High | Low |
| Ideal | Low | Low |

**Overfitting**: the model fits too closely to the training data; training data is over-represented, so it predicts poorly on new data.
- The model **memorized** rather than learned.
- It cannot be applied to new data.

**Underfitting**: the model does not get enough training to understand the pattern.
- Does not work for training data OR unseen data.

We can rectify overfitting and underfitting by adjusting bias and variance.

### Relation to Ensemble Methods

Ensemble methods try to **reduce the variance** of complex models (with low bias) by **aggregating** the responses of multiple base classifiers. This helps deal with **overfitting**. There are, of course, other approaches to solve overfitting.

---

## Causes and Remedies for Underfitting / Overfitting

### Reasons for Underfitting

- Model is not complex enough / too simple.
- Not enough samples from different classes.
- Model was not trained for enough epochs.
- Low data quality.

### Reasons for Overfitting

- Training too much.
- Too little training data.
- Using a very complex model.

### How to Deal With Overfitting

1. **Data augmentation** / collect more data.
   - We don't always collect perfect data, so extra augmented data usually contains some noise (though we wouldn't directly call it noisy data). So augmentation implicitly introduces some noise into the training data.
2. **Regularization**:
   - Introduce dropout at different stages.
   - The **learning rate** is also related to overfitting.
3. **Simplify the model**:
   - Pruning.
   - Dropout in neural networks.
4. **Proper activation function** (for neural networks).
5. **Ensemble methods**.

---

## Constructing a Diverse Set of Classifiers

Four main strategies, based on what part of the learning process is manipulated.

### 1. By Manipulating the Training Set

Example: **bagging**, **boosting**, **random forests**.

- Train, for example, one SVM on 10 different bootstrap subsets.
- Remember: ensemble methods work well with **unstable** classifiers; if you change the training data, the results change.

### 2. By Manipulating Input Features

Example: **random forests**.

- Not applicable to ANN, but works for traditional models.
- Use different subsets of features (including feature reduction).

### 3. By Manipulating Class Labels

Example: **error-correcting output coding**.

- Remove some samples from the majority class.
- Change some labels to add noise.

### 4. By Manipulating the Learning Algorithm

Example: **injecting randomness in the initial weights of an ANN**, or using different algorithms entirely.

- SVM results ≠ logistic regression results ≠ neural network results.
- Or keep the algorithm and change hyperparameters (C, gamma, etc.).

### Class Discussion: Are Three Classifiers With the Same Accuracy Diverse?

- Suppose SVM, KNN, and ANN each achieve 60%.
- Are they diverse? **Yes**.
- **Why?** They got the same percentage, but **how** they got there matters. One may have classified a sample as "no" and another as "yes"; the overall percentage is the same, but the predictions differ.
- To assess diversity, you must **look at predictions per sample**. It is possible to get the same overall performance with different true positives, false positives, and true negatives.
- Because different algorithms use different mechanisms:
  - For some samples, SVM will work fine.
  - For other samples, KNN will work.
  - For others, ANN.
- **Main idea of ensemble learning**: use different classifiers so they cover each other's weaknesses. If one classifier works well in some region of the data, use that one; if another works well in another region, use that one.

### Practical Note From the Instructor

If statistically one model (e.g., SVM) has a 30% chance to work well and another has 60%, it is not necessary to combine them. Just use the better model. It is more efficient, especially if the better one is a complicated tree model.

But if the F1 scores are the same, it makes sense to combine them. Usually F1 scores are quite different. This example emphasizes that **performance alone cannot tell you about diversity**. Even if all classifiers score 85%, combining them may still yield 90%.

### Combination of Approaches

We can use a mix: different algorithms, different settings, different training subsets, and different features. You can also organize multiple stages (lower-level and higher-level models).

---

## Famous Ensemble Approaches

Three main families:
- **Bagging**
- **Stacking**
- **Boosting** (has several variants: AdaBoost, Gradient Boosting, etc.)

---

## Stacking

Stacking is **simpler** than bagging and boosting.

### Idea

The ensemble is arranged in **layers**, with one level feeding into the next. You stack classifiers one on top of the other.

### Physician Analogy

1. You have a problem. You go to physician 1, then physician 2, then physician 3.
2. Instead of taking the majority vote yourself, you give the three decisions to a **final decision-maker** (another expert).
3. That final expert analyzes the decisions without talking to you and makes the final decision.

### Comparison With Majority Voting

| Step | Single Classifier | Normal Ensemble (Majority Vote) | Stacking |
|---|---|---|---|
| Represent yourself via features | ✓ | ✓ | ✓ |
| Ask one expert | ✓ | | |
| Ask multiple experts | | ✓ | ✓ |
| Take majority of their decisions | | ✓ | |
| Feed decisions into another expert (meta-classifier) | | | ✓ |

> In a previous assignment, you used **logistic regression** as the meta-classifier and gave it the outputs from different base classifiers. That is stacking.

```python
# (reconstructed example) Stacking with scikit-learn
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

base_learners = [
    ('svm', SVC(probability=True)),
    ('knn', KNeighborsClassifier()),
    ('dt',  DecisionTreeClassifier()),
]

stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression()  # meta-classifier
)
stack.fit(X_train, y_train)
```

---

## Bagging (Bootstrap AGGregatING)

### Bootstrap Sampling

**Bootstrap sampling**: sampling **with replacement**.

Example of three bootstrap rounds from 10 original samples:

| Original Data     | 1 | 2 | 3  | 4  | 5 | 6 | 7  | 8  | 9 | 10 |
|-------------------|---|---|----|----|---|---|----|----|---|----|
| Bagging (Round 1) | 7 | 8 | 10 | 8  | 2 | 5 | 10 | 10 | 5 | 9  |
| Bagging (Round 2) | 1 | 4 | 9  | 1  | 2 | 3 | 2  | 7  | 3 | 2  |
| Bagging (Round 3) | 1 | 8 | 5  | 10 | 5 | 5 | 9  | 6  | 3 | 7  |

- Since we sample with replacement:
  - Duplicates are possible.
  - Some original samples may **not** appear in a given bootstrap sample.
- Build a classifier on each bootstrap sample.

### Probability of Being Selected

The probability of a training instance being selected in a bootstrap sample is:

$$P(\text{selected}) = 1 - \left(1 - \frac{1}{n}\right)^n \approx 0.632 \text{ when } n \text{ is large}$$

Where $n$ is the number of training instances.

*(Reconstructed derivation)*: The probability of a specific sample NOT being drawn in one of $n$ draws is $(1 - 1/n)^n \to 1/e \approx 0.368$. So the probability of being selected at least once is $1 - 1/e \approx 0.632$.

That means the probability of each individual sample from the original data being selected in one of the bootstrap samples is acceptable; it is high enough.

### Bagging Algorithm (Algorithm 4.5)

1. Let $k$ be the number of bootstrap samples.
2. **for** $i = 1$ to $k$ **do**
3. &nbsp;&nbsp;&nbsp;&nbsp;Create a bootstrap sample of size $N$, $D_i$.
4. &nbsp;&nbsp;&nbsp;&nbsp;Train a base classifier $C_i$ on the bootstrap sample $D_i$.
5. **end for**
6. $C^*(x) = \arg\max_y \sum_i \delta(C_i(x) = y)$.

Where $\delta(\cdot) = 1$ if its argument is true and $0$ otherwise.

**Are classifiers diverse here?** Yes, because we use different training sets, so even with the same algorithm the results differ across rounds.

**Can we have the same training set?** Yes, duplicates are allowed.

---

## Bagging With Decision Stumps: Worked Example

### Dataset

Consider a 1-dimensional data set with 10 samples:

**Original Data:**

| x | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| y | 1   | 1   | 1   | -1  | -1  | -1  | -1  | 1   | 1   | 1   |

### Classifier: Decision Stump

A decision stump is a decision tree of size 1:
- Decision rule: $x \leq k$ versus $x > k$.
- Split point $k$ is chosen based on entropy.

```
      (x ≤ k)
      /     \
   True    False
    │        │
  y_left  y_right
```

### Bagging Rounds 1–5

**Round 1** (bootstrap sample):

| x | 0.1 | 0.2 | 0.2 | 0.3 | 0.4 | 0.4 | 0.5 | 0.6 | 0.9 | 0.9 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| y | 1   | 1   | 1   | 1   | -1  | -1  | -1  | -1  | 1   | 1   |

Rule: $x \leq 0.35 \rightarrow y = 1$; $x > 0.35 \rightarrow y = -1$.

**Round 2:**

| x | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.5 | 0.9 | 1.0 | 1.0 | 1.0 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| y | 1   | 1   | 1   | -1  | -1  | -1  | 1   | 1   | 1   | 1   |

Rule: $x \leq 0.7 \rightarrow y = 1$; $x > 0.7 \rightarrow y = 1$ (both sides predict 1).

**Round 3:**

| x | 0.1 | 0.2 | 0.3 | 0.4 | 0.4 | 0.5 | 0.7 | 0.7 | 0.8 | 0.9 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| y | 1   | 1   | 1   | -1  | -1  | -1  | -1  | -1  | 1   | 1   |

Rule: $x \leq 0.35 \rightarrow y = 1$; $x > 0.35 \rightarrow y = -1$.

**Round 4:**

| x | 0.1 | 0.1 | 0.2 | 0.4 | 0.4 | 0.5 | 0.5 | 0.7 | 0.8 | 0.9 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| y | 1   | 1   | 1   | -1  | -1  | -1  | -1  | -1  | 1   | 1   |

Rule: $x \leq 0.3 \rightarrow y = 1$; $x > 0.3 \rightarrow y = -1$.

**Round 5:**

| x | 0.1 | 0.1 | 0.2 | 0.5 | 0.6 | 0.6 | 0.6 | 1.0 | 1.0 | 1.0 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| y | 1   | 1   | 1   | -1  | -1  | -1  | -1  | 1   | 1   | 1   |

Rule: $x \leq 0.35 \rightarrow y = 1$; $x > 0.35 \rightarrow y = -1$.

### Bagging Rounds 6–10

**Round 6:**

| x | 0.2 | 0.4 | 0.5 | 0.6 | 0.7 | 0.7 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| y | 1   | -1  | -1  | -1  | -1  | -1  | -1  | 1   | 1   | 1   |

Rule: $x \leq 0.75 \rightarrow y = -1$; $x > 0.75 \rightarrow y = 1$.

**Round 7:**

| x | 0.1 | 0.4 | 0.4 | 0.6 | 0.7 | 0.8 | 0.9 | 0.9 | 0.9 | 1.0 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| y | 1   | -1  | -1  | -1  | -1  | 1   | 1   | 1   | 1   | 1   |

Rule: $x \leq 0.75 \rightarrow y = -1$; $x > 0.75 \rightarrow y = 1$.

**Round 8:**

| x | 0.1 | 0.2 | 0.5 | 0.5 | 0.5 | 0.7 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| y | 1   | 1   | -1  | -1  | -1  | -1  | -1  | 1   | 1   | 1   |

Rule: $x \leq 0.75 \rightarrow y = -1$; $x > 0.75 \rightarrow y = 1$.

**Round 9:**

| x | 0.1 | 0.3 | 0.4 | 0.4 | 0.6 | 0.7 | 0.7 | 0.8 | 1.0 | 1.0 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| y | 1   | 1   | -1  | -1  | -1  | -1  | -1  | 1   | 1   | 1   |

Rule: $x \leq 0.75 \rightarrow y = -1$; $x > 0.75 \rightarrow y = 1$.

**Round 10:**

| x | 0.1 | 0.1 | 0.1 | 0.1 | 0.3 | 0.3 | 0.8 | 0.8 | 0.9 | 0.9 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| y | 1   | 1   | 1   | 1   | 1   | 1   | 1   | 1   | 1   | 1   |

Rule: $x \leq 0.05 \rightarrow y = 1$; $x > 0.05 \rightarrow y = 1$ (always 1).

### Summary of Trained Decision Stumps

| Round | Split Point | Left Class | Right Class |
|-------|-------------|------------|-------------|
| 1     | 0.35        | 1          | -1          |
| 2     | 0.7         | 1          | 1           |
| 3     | 0.35        | 1          | -1          |
| 4     | 0.3         | 1          | -1          |
| 5     | 0.35        | 1          | -1          |
| 6     | 0.75        | -1         | 1           |
| 7     | 0.75        | -1         | 1           |
| 8     | 0.75        | -1         | 1           |
| 9     | 0.75        | -1         | 1           |
| 10    | 0.05        | 1          | 1           |

Thresholds can repeat across rounds because bootstrap samples can produce similar splits.

### Classification (Majority Vote)

For each test $x$, take the sign of the sum of predictions across all 10 stumps.

| Round | x=0.1 | x=0.2 | x=0.3 | x=0.4 | x=0.5 | x=0.6 | x=0.7 | x=0.8 | x=0.9 | x=1.0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1     | 1     | 1     | 1     | -1    | -1    | -1    | -1    | -1    | -1    | -1    |
| 2     | 1     | 1     | 1     | 1     | 1     | 1     | 1     | 1     | 1     | 1     |
| 3     | 1     | 1     | 1     | -1    | -1    | -1    | -1    | -1    | -1    | -1    |
| 4     | 1     | 1     | 1     | -1    | -1    | -1    | -1    | -1    | -1    | -1    |
| 5     | 1     | 1     | 1     | -1    | -1    | -1    | -1    | -1    | -1    | -1    |
| 6     | -1    | -1    | -1    | -1    | -1    | -1    | -1    | 1     | 1     | 1     |
| 7     | -1    | -1    | -1    | -1    | -1    | -1    | -1    | 1     | 1     | 1     |
| 8     | -1    | -1    | -1    | -1    | -1    | -1    | -1    | 1     | 1     | 1     |
| 9     | -1    | -1    | -1    | -1    | -1    | -1    | -1    | 1     | 1     | 1     |
| 10    | 1     | 1     | 1     | 1     | 1     | 1     | 1     | 1     | 1     | 1     |
| **Sum** | **2** | **2** | **2** | **-6** | **-6** | **-6** | **-6** | **2** | **2** | **2** |
| **Predicted** | **1** | **1** | **1** | **-1** | **-1** | **-1** | **-1** | **1** | **1** | **1** |

This matches the true labels perfectly, whereas a single decision stump could not achieve this separation.

### Observations

> Most of the time you end up training classifiers that classify everything into one class, **but the ensemble still works**.

> **Bagging can also increase the complexity (representation capacity) of simple classifiers** such as decision stumps.

---

## Strengths and Weaknesses of Bagging

### Strengths

- Based on the different samples used in each round, the stumps divide positives and negatives differently, **including cases where a single classifier would get it wrong**.
- The labels produced differ across rounds because of the sub-sampling.
- **Performance is better than a single decision stump**.

### Weakness: Explainability

For an **individual classifier**, we can explain the decision:
- "Because $x$ was less than 0.35, I predicted 1."

For an **ensemble**, we cannot explain the decision well:
- "Because I trained 10 classifiers and took the majority vote" is **not** a satisfying reason.
- It does not explain **why each classifier** predicted what it did.

> Just because a model or system rejected something, it should **explain why**. You should explain to the customer, but first the model should explain to you. This is **Explainable AI**. Explainable AI is not a new domain, but it is a hard domain.

### Scaling Up the Problem

- With 10 classifiers, each stump is very simple (e.g., "if $x < 0.35$ predict 1").
- But what if we have **millions of experts**, like **MoE (Mixture of Experts)**? We still need to explain the decision; we cannot just say "because these millions of experts each said this, we arrived at this decision."

> **Course note**: the most important weakness of the ensemble is **lack of explainability**. This is a key concept to remember.

---

## Boosting

**Boosting** is an iterative procedure to adaptively change the distribution of training data by focusing more on **previously misclassified records**.

- Initially, all $N$ records are assigned equal weights (for being selected for training).
- Unlike bagging, weights **may change** at the end of each boosting round.
- Records that are **wrongly classified** have their weights **increased** in the next round.
- Records that are **classified correctly** have their weights **decreased** in the next round.

### Illustration

| Original Data      | 1 | 2 | 3 | 4  | 5 | 6 | 7 | 8  | 9 | 10 |
|--------------------|---|---|---|----|---|---|---|----|---|----|
| Boosting (Round 1) | 7 | 3 | 2 | 8  | 7 | 9 | 4 | 10 | 6 | 3  |
| Boosting (Round 2) | 5 | 4 | 9 | 4  | 2 | 5 | 1 | 7  | 4 | 2  |
| Boosting (Round 3) | 4 | 4 | 8 | 10 | 4 | 5 | 4 | 6  | 3 | 4  |

- Example **4** is hard to classify.
- Its weight is increased, so it is more likely to be chosen again in subsequent rounds.

---

## AdaBoost

### Notation

- Base classifiers: $C_1, C_2, \ldots, C_T$.
- Training set of size $N$.
- $w_j^{(i)}$: weight of example $j$ at round $i$.

### Error Rate of a Base Classifier

$$\varepsilon_i = \frac{1}{N} \sum_{j=1}^{N} w_j^{(i)} \, \delta\bigl(C_i(x_j) \neq y_j\bigr)$$

### Importance of a Classifier

$$\alpha_i = \frac{1}{2} \ln\left(\frac{1 - \varepsilon_i}{\varepsilon_i}\right)$$

- Lower error → higher $\alpha_i$ → more influence in the final vote.

### Weight Update Rule

$$w_j^{(i+1)} = \frac{w_j^{(i)}}{Z_i} \times \begin{cases} e^{-\alpha_i} & \text{if } C_i(x_j) = y_j \quad (\text{correct: decrease weight}) \\ e^{\alpha_i} & \text{if } C_i(x_j) \neq y_j \quad (\text{wrong: increase weight}) \end{cases}$$

Where $Z_i$ is a normalization factor so weights sum to 1.

> **Reset rule**: if any intermediate round produces an error rate higher than 50%, the weights are reverted back to $1/N$ and the resampling procedure is repeated.

### Classification (Weighted Vote)

$$C^*(x) = \arg\max_y \sum_{i=1}^{T} \alpha_i \, \delta\bigl(C_i(x) = y\bigr)$$

### AdaBoost Algorithm (Algorithm 4.6)

1. $\mathbf{w} = \{w_j = 1/N \mid j = 1, 2, \ldots, N\}$. *Initialize weights.*
2. Let $k$ be the number of boosting rounds.
3. **for** $i = 1$ to $k$ **do**
4. &nbsp;&nbsp;&nbsp;&nbsp;Create training set $D_i$ by sampling (with replacement) from $D$ according to $\mathbf{w}$.
5. &nbsp;&nbsp;&nbsp;&nbsp;Train a base classifier $C_i$ on $D_i$.
6. &nbsp;&nbsp;&nbsp;&nbsp;Apply $C_i$ to all examples in the original training set $D$.
7. &nbsp;&nbsp;&nbsp;&nbsp;$\varepsilon_i = \frac{1}{N}\left[\sum_j w_j \, \delta(C_i(x_j) \neq y_j)\right]$. *Calculate weighted error.*
8. &nbsp;&nbsp;&nbsp;&nbsp;**if** $\varepsilon_i > 0.5$ **then**
9. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Reset $\mathbf{w} = \{w_j = 1/N\}$.
10. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Go back to Step 4.
11. &nbsp;&nbsp;&nbsp;&nbsp;**end if**
12. &nbsp;&nbsp;&nbsp;&nbsp;$\alpha_i = \frac{1}{2} \ln \frac{1-\varepsilon_i}{\varepsilon_i}$.
13. &nbsp;&nbsp;&nbsp;&nbsp;Update the weight of each example according to the weight update rule.
14. **end for**
15. $C^*(\mathbf{x}) = \arg\max_y \sum_{j=1}^{T} \alpha_j \, \delta(C_j(\mathbf{x}) = y)$.

### AdaBoost Example

Same 1-D dataset as the bagging example:

| x | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| y | 1   | 1   | 1   | -1  | -1  | -1  | -1  | 1   | 1   | 1   |

Decision stumps used with entropy-based split.

**Training sets for rounds 1–3:**

**Round 1:**

| x | 0.1 | 0.4 | 0.5 | 0.6 | 0.6 | 0.7 | 0.7 | 0.7 | 0.8 | 1.0 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| y | 1   | -1  | -1  | -1  | -1  | -1  | -1  | -1  | 1   | 1   |

**Round 2:**

| x | 0.1 | 0.1 | 0.2 | 0.2 | 0.2 | 0.2 | 0.3 | 0.3 | 0.3 | 0.3 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| y | 1   | 1   | 1   | 1   | 1   | 1   | 1   | 1   | 1   | 1   |

**Round 3:**

| x | 0.2 | 0.2 | 0.4 | 0.4 | 0.4 | 0.4 | 0.5 | 0.6 | 0.6 | 0.7 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| y | 1   | 1   | -1  | -1  | -1  | -1  | -1  | -1  | -1  | -1  |

**Summary of trained stumps:**

| Round | Split Point | Left Class | Right Class | $\alpha$ |
|-------|-------------|------------|-------------|----------|
| 1     | 0.75        | -1         | 1           | 1.738    |
| 2     | 0.05        | 1          | 1           | 2.7784   |
| 3     | 0.3         | 1          | -1          | 4.1195   |

**Weights per round:**

| Round | x=0.1 | x=0.2 | x=0.3 | x=0.4 | x=0.5 | x=0.6 | x=0.7 | x=0.8 | x=0.9 | x=1.0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1     | 0.1   | 0.1   | 0.1   | 0.1   | 0.1   | 0.1   | 0.1   | 0.1   | 0.1   | 0.1   |
| 2     | 0.311 | 0.311 | 0.311 | 0.01  | 0.01  | 0.01  | 0.01  | 0.01  | 0.01  | 0.01  |
| 3     | 0.029 | 0.029 | 0.029 | 0.228 | 0.228 | 0.228 | 0.228 | 0.009 | 0.009 | 0.009 |

Notice how misclassified samples (e.g., x = 0.1, 0.2, 0.3 after round 1) get much larger weights in round 2, and then the ones that round 2 gets wrong (x = 0.4–0.7) get boosted for round 3.

**Classification (weighted vote):**

| Round | x=0.1 | x=0.2 | x=0.3 | x=0.4 | x=0.5 | x=0.6 | x=0.7 | x=0.8 | x=0.9 | x=1.0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1     | -1    | -1    | -1    | -1    | -1    | -1    | -1    | 1     | 1     | 1     |
| 2     | 1     | 1     | 1     | 1     | 1     | 1     | 1     | 1     | 1     | 1     |
| 3     | 1     | 1     | 1     | -1    | -1    | -1    | -1    | -1    | -1    | -1    |
| **Weighted sum** | **5.16** | **5.16** | **5.16** | **-3.08** | **-3.08** | **-3.08** | **-3.08** | **0.397** | **0.397** | **0.397** |
| **Predicted** | **1** | **1** | **1** | **-1** | **-1** | **-1** | **-1** | **1** | **1** | **1** |

The weighted vote correctly recovers all 10 labels, even though each individual stump makes mistakes.

---

## Random Forest

### Algorithm

Construct an ensemble of decision trees by manipulating the **training set** *and* the **features**:

1. Use a bootstrap sample to train every decision tree (similar to bagging).
2. Tree induction:
   - At every internal node of the decision tree, **randomly sample $p$ attributes** for selecting the split criterion.
   - Repeat this procedure until all leaves are pure (an **unpruned** tree).

### Characteristics

- Base classifiers are **unpruned trees**, hence **unstable classifiers** (which is what we want for ensembles).
- Base classifiers are **decorrelated** (due to randomization in both training set and features).
- Random forests **reduce variance** of unstable classifiers **without negatively impacting the bias**.

### Hyperparameter $p$ (number of features sampled at each split)

| Choice | Effect |
|---|---|
| Small $p$ | Ensures lack of correlation between trees. |
| High $p$ | Promotes strong base classifiers. |
| $\sqrt{d}$ | Common default choice. |
| $\log_2(d + 1)$ | Another common default. |

Where $d$ is the total number of features.

```python
# (added) Random Forest in scikit-learn
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',   # p = sqrt(d)
    bootstrap=True,
    random_state=42
)
rf.fit(X_train, y_train)
```

---

## Gradient Boosting

- Constructs a **series of models** iteratively.
- Models can be any predictive model that has a **differentiable loss function**.
- Commonly, **trees** are the chosen model.
  - **XGBoost** (extreme gradient boosting) is a popular package because of its impressive performance.
- Boosting can be viewed as optimizing the loss function by **iterative functional gradient descent**.
- Implementations are available in Python, R, MATLAB, and more.

```python
# (added) Gradient Boosting / XGBoost sketch
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)

# Or with XGBoost
# import xgboost as xgb
# model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
# model.fit(X_train, y_train)
```

---

## Summary: Bagging vs. Boosting vs. Stacking

| Aspect | Bagging | Boosting | Stacking |
|---|---|---|---|
| Sampling | Bootstrap (with replacement), equal weights | Bootstrap with **adaptive weights** | Typically uses the full data |
| Rounds | Independent / parallel | Sequential (each depends on previous) | Two levels: base + meta |
| Combination | Majority vote (unweighted) | Weighted vote (by $\alpha_i$) | Meta-classifier decides |
| Focus | Reduce **variance** | Reduce **bias** and variance | Combine heterogeneous models |
| Example | Random Forest | AdaBoost, Gradient Boosting, XGBoost | Logistic regression on top of SVM/KNN/ANN |

---

## Summary Table: Single Classifier vs. Ensemble

| Aspect | Single Classifier | Ensemble |
|---|---|---|
| Error rate (example) | 35% | 6% (with 25 independent classifiers) |
| Explainability | Good | Poor |
| Stability needed | N/A | Works best with unstable bases |
| Diversity needed | N/A | Required |
| Variance reduction | No | Yes (helps with overfitting) |
| Complexity | Lower | Higher |

---

## Key Takeaways

> **Autoencoders** reconstruct their input and can be used for compression, denoising, and anomaly detection.

> **One-Class SVM** uses the origin trick to separate normal data from the origin, which is treated as the "other" class.

> **Ensemble learning** combines multiple diverse base classifiers to reduce variance and improve performance, provided the classifiers are independent, better than random, and ideally unstable.

> **Bagging** uses bootstrap samples and averages (majority vote). **Boosting** (AdaBoost, Gradient Boosting, XGBoost) adaptively reweights hard examples and uses a weighted vote. **Stacking** feeds base outputs into a meta-classifier. **Random Forest** is bagging + feature subsampling on unpruned decision trees.

> The big trade-off of ensembles: **higher accuracy** but **lower explainability**, a concern amplified in modern architectures like Mixture of Experts.
