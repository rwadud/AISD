# Lecture 9: Evaluation Metrics for Imbalanced Data, ROC Curves, Sampling Techniques, and Outlier Detection

> **Course**: CST8506 Advanced Machine Learning. Dr. Abbas Akkasi, Winter 2026.
> Slide materials originally developed by Pang-Ning Tan for his Data Mining course.

## 1. Evaluation on Imbalanced Class Problems

### The Class Imbalance Problem

Many real-world classification problems have skewed class distributions, with far more records from one class than another. Typical examples:

- Credit card fraud.
- Intrusion detection.
- Defective products on a manufacturing assembly line.
- COVID-19 test results on a random sample.

> **Key challenge**: evaluation measures such as **accuracy** are not well suited for imbalanced classes.

### Why Accuracy Fails

When we have an imbalanced class problem, such as credit card fraud detection, we evaluate based on the **confusion matrix** rather than on accuracy. The reason is that most samples belong to the negative class, so a trivial model that predicts everything as negative would still achieve high accuracy without being useful on the minority class.

### Confusion Matrix Layout

The binary confusion matrix uses the letters `a`, `b`, `c`, `d`:

| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | `a` (TP) | `b` (FN) |
| **Actual No** | `c` (FP) | `d` (TN) |

### Accuracy Formula

$$\text{Accuracy} = \frac{a + d}{a + b + c + d} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Error Rate} = 1 - \text{Accuracy}$$

### Problem With Accuracy: the 99 Percent Trivial Model

Consider a 2-class problem:

- Number of Class **No** examples: 990.
- Number of Class **Yes** examples: 10.

If a model predicts **everything to be class No**, its confusion matrix is:

| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | 0 | 10 |
| **Actual No** | 0 | 990 |

Accuracy = 990 / 1000 = **99 percent**. This is misleading because this trivial model does not detect any class Yes example. Detecting the rare class is usually the interesting task (frauds, intrusions, defects).

### Which Model Is Better?

**Model A**:

| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | 0 | 10 |
| **Actual No** | 0 | 990 |

Accuracy = 99 percent. Catches no positives at all.

**Model B**:

| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | 10 | 0 |
| **Actual No** | 500 | 490 |

Accuracy = 50 percent. Catches every positive but issues 500 false alarms.

Looking at accuracy alone, Model A looks far superior. Looking at positive-class recall, only Model B does anything useful.

A second pair from the slides:

**Model A** (partial positive coverage, zero false alarms):

| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | 5 | 5 |
| **Actual No** | 0 | 990 |

**Model B** (full recall, many false alarms):

| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | 10 | 0 |
| **Actual No** | 500 | 490 |

Again, accuracy alone cannot discriminate these two behaviors cleanly.

### Precision

**Precision**: out of all samples predicted as positive, how many were actually positive.

$$\text{Precision}\ (p) = \frac{a}{a + c} = \frac{TP}{TP + FP}$$

Here, `a + c` is the number of samples predicted as positive, and `a` is the number of correctly predicted positives. Also known as **Positive Predictive Value**.

### Recall

**Recall**: among all the real positive samples, how many were predicted correctly.

$$\text{Recall}\ (r) = \frac{a}{a + b} = \frac{TP}{TP + FN}$$

Here, `a + b` is the total number of actual positive samples. Also called **sensitivity** or **TP Rate**.

### F-Measure

**F-measure** (harmonic mean of precision and recall):

$$F = \frac{2rp}{r + p} = \frac{2a}{2a + b + c}$$

### Why We Need an F-Measure

If you compare two models with only one metric, comparison is easy. Based on the value of that metric, one is better than the other. But once the number of metrics grows, comparison becomes difficult.

Suppose you have two models:
- Model A has high recall and low precision.
- Model B has low recall and high precision.

How can we compare them? To solve this, we define another metric like the **F-measure**. Specifically, the F1 score combines precision and recall into a single number.

### The F-Beta Family

In general, we have the **F-beta score**. Beta represents the importance of recall with respect to precision. It expresses how many times recall is more important than precision.

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}} \quad \text{\textit{(reconstructed)}}$$

When we say **F1 score**, beta equals 1, so we treat their importance as equal.

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

But most of the time their importance is not equal.

### Example: Disease Detection

In disease detection, both precision and recall are important. For COVID detection, which is more important?

> **Recall**, because you need to find most of the people who are actually infected by COVID. Precision is also important, but just a little.

### Worked Example 1 (from the slides)

| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | 10 | 0 |
| **Actual No** | 10 | 980 |

$$\text{Precision} = \frac{10}{10 + 10} = 0.5$$

$$\text{Recall} = \frac{10}{10 + 0} = 1.0$$

$$F = \frac{2 \cdot 1 \cdot 0.5}{1 + 0.5} = 0.62$$

$$\text{Accuracy} = \frac{990}{1000} = 0.99$$

This is the same "precision is 10 over 10 plus 10" computation the instructor did at the board.

### Worked Example 2 (from the slides)

| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | 1 | 9 |
| **Actual No** | 0 | 990 |

$$\text{Precision} = \frac{1}{1 + 0} = 1.0$$

$$\text{Recall} = \frac{1}{1 + 9} = 0.1$$

$$F = \frac{2 \cdot 0.1 \cdot 1}{1 + 0.1} = 0.18$$

$$\text{Accuracy} = \frac{991}{1000} = 0.991$$

Near-perfect accuracy and perfect precision, yet the F-measure is only 0.18 because recall collapses.

### Same Accuracy, Very Different F

**Classifier A** (balanced data):

| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | 40 | 10 |
| **Actual No** | 10 | 40 |

Precision = 0.8, Recall = 0.8, F = 0.8, Accuracy = 0.8.

**Classifier B** (severely imbalanced data):

| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | 40 | 10 |
| **Actual No** | 1000 | 4000 |

Precision ≈ 0.04, Recall = 0.8, F ≈ 0.08, Accuracy ≈ 0.8.

Both classifiers show the same accuracy. By accuracy alone they look identical. By precision and F-measure, Classifier B is clearly unusable.

### The Full Metric Set (from the slides)

Using the TP, FN, FP, TN layout:

$$\text{Accuracy} = \frac{TP + TN}{TP + FN + FP + TN}$$

$$\text{Error Rate} = 1 - \text{Accuracy}$$

$$\text{Precision} = \text{Positive Predictive Value} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \text{Sensitivity} = \text{TP Rate} = \frac{TP}{TP + FN}$$

$$\text{Specificity} = \text{TN Rate} = \frac{TN}{TN + FP}$$

$$\text{FP Rate} = \alpha = \frac{FP}{TN + FP} = 1 - \text{Specificity}$$

$$\text{FN Rate} = \beta = \frac{FN}{FN + TP} = 1 - \text{Sensitivity}$$

$$\text{Power} = \text{Sensitivity} = 1 - \beta$$

**Terminology from the transcript**: FPR is the quantity sometimes denoted alpha, and FNR maps to beta in the statistical sense (probability of a Type II error).

### Accuracy Can Hide Problems

If you now have two models, the first with an accuracy of 99 percent, and the second whose accuracy looks similar but whose underlying precision, recall, and F-measure differ, just by looking at accuracy you cannot tell which model handles the minority class better.

### TPR over FPR at a Glance

Reinterpreting the two Classifiers A and B above via TPR and FPR:

**Classifier A** (50 positives, 50 negatives):
- TPR = Recall = 0.8, FPR = 0.2.
- TPR / FPR = 4.

**Classifier B** (50 positives, 5000 negatives):
- TPR = Recall = 0.8, FPR = 0.2.
- TPR / FPR = 4.

The TPR / FPR ratio is identical, but the absolute count of false positives is radically different because the class sizes are different. This is another lens on why precision matters alongside recall.

### Three Classifiers at the Same Precision (from the slides)

**Classifier A**:

| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | 10 | 40 |
| **Actual No** | 10 | 40 |

Precision = 0.5, TPR = 0.2, FPR = 0.2, F = 0.28.

**Classifier B**:

| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | 25 | 25 |
| **Actual No** | 25 | 25 |

Precision = 0.5, TPR = 0.5, FPR = 0.5, F = 0.5.

**Classifier C**:

| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | 40 | 10 |
| **Actual No** | 40 | 10 |

Precision = 0.5, TPR = 0.8, FPR = 0.8, F = 0.61.

All three share precision = 0.5, yet their operating points on the ROC plane differ dramatically. F-measure prefers Classifier C because it catches more real positives, even though it also triggers more false alarms.

### Task-Specific Measurements

These metrics are for classification tasks. For different machine learning tasks we have different evaluations:

- **Machine translation**: **BLEU**.
- **Text generation**: **perplexity**.
- **Model comparison in classification**: **precision-recall curves**.

### Score Aggregation Variants

You will also hear about different versions of these metrics such as the **micro score**, **macro score**, and **weighted scores**. Not all are used everywhere. You should know which metric fits each situation.

*(added clarification)*

| Variant | How It Aggregates |
|---|---|
| **Macro** | Compute the metric for each class independently, then take the unweighted average. Treats all classes equally. |
| **Micro** | Pool TP, FP, FN across all classes, then compute the metric once. Dominated by the majority class. |
| **Weighted** | Per-class metric averaged, weighted by the number of true instances in each class. |

### Interpreting Combinations of Accuracy and F-Measure

If you have an F-measure of 80 percent and accuracy of 80 percent, the model looks strong. If F-measure is very low while accuracy is similar, performance is poor on the minority class, and we judge the model based on F-measure, which is more significant here.

### Deployment Decision

Suppose you want to decide about deployment of a model. You trained a model for a task and have these evaluation values, and you want to decide whether to deploy the model or not.

> **Course note**: In college or university we train models for grades, but in reality we should train models for use. Not every model we train should be deployed. There must be some criteria for that decision.

**The three-model scenario from class**: imagine you train a model and obtain a sequence of results. Suppose we see three candidates, and the last one is better compared to the previous two. The question is still, are you going to deploy it?

Decision logic:

1. If the model is worse than a random guess (random gives 50 percent, and this one is only 28 percent), forget about it.
2. If it is better than random, you then decide based on the problem.
3. As mentioned, for some problems recall is much more important than precision, and in those cases we prefer the model with higher recall.

### Detecting Imbalance From the Metrics Themselves

By looking at the difference between precision and recall, you can also tell whether your training dataset was balanced or imbalanced.

- With an **imbalanced** dataset, the difference between precision and recall is usually large.
- With a **balanced** dataset they are similar.

For example, if you have 90 positive samples but only 10 are predicted correctly, recall on the positive class is very low.

---

## 2. ROC Curve (Receiver Operating Characteristic)

### What It Is

The **ROC curve** is a graphical approach showing the trade-off between **detection rate** (TPR) and **false alarm rate** (FPR). We use it to display model performance and to compare different models.

> **Historical note**: ROC was developed in the 1950s for signal detection theory to analyze noisy signals.

The curve plots **TPR against FPR**. The performance of a model at one threshold is represented as a single point on the ROC curve.

### Corner Points and the Diagonal

- **(0, 0)**: declare everything to be the **negative** class. TPR is zero and FPR is zero.
- **(1, 1)**: declare everything to be the **positive** class. All positives and all negatives are predicted positive.
- **(1, 0)**: the **ideal** corner (catch all positives, zero false alarms).
- **Diagonal line** from (0, 0) to (1, 1): **random guessing**.
- **Below the diagonal**: predictions are **opposite** of the true class (flip the labels to get something useful).

### Why We Need Continuous Scores

To draw an ROC curve, the classifier must produce **continuous-valued output**. Outputs are used to **rank test records**, from the most likely positive to the least likely. By using **different thresholds** on this value, we can create different variations of the classifier with different TPR / FPR trade-offs.

Many classifiers produce only discrete outputs (the predicted class). To get continuous-valued outputs we can use: **decision trees**, rule-based classifiers, neural networks, Bayesian classifiers, k-nearest neighbors, and SVM.

### Decision Tree Case

At a leaf we have a set of positive and negative samples. Instead of assigning a discrete label, we take the fraction of positive samples over all samples at that leaf:

$$\text{score} = \frac{\text{number of positive samples at the leaf}}{\text{total samples at the leaf}}$$

*(slide example)* A decision tree that splits on features like `x2 < 12.63`, `x1 < 13.29`, `x1 < 6.56`, and so on produces leaves with probabilities such as 0.107, 0.059, 0.220, 0.071, 0.164, 0.143, 0.669, 0.727, 0.271, 0.654, 0. These probabilities are the continuous-valued outputs used to build the ROC curve.

This converts the classifier into a ranking from the most likely positive to the least likely. By sweeping the threshold, we produce different variations of the classifier with different TPR and FPR.

### Converting Discrete Classifiers

Many classifiers produce only discrete values, so we convert them by dividing the number of positive samples by all the samples in the node. Then, based on different threshold values, we make decisions on class labels. Any sample with a value higher than the threshold is labeled positive. Each threshold gives a different confusion matrix.

### ROC Example on a 2-D Training Set (from the slides)

Using the decision tree above on a 2-D training set:

**Threshold α = 0.3**:

| | Predicted Class `o` | Predicted Class `+` |
|---|---|---|
| **Actual `o`** | 645 | 209 |
| **Actual `+`** | 298 | 948 |

**Threshold α = 0.7**:

| | Predicted Class `o` | Predicted Class `+` |
|---|---|---|
| **Actual `o`** | 181 | 673 |
| **Actual `+`** | 78 | 1168 |

Changing α shifts TP, FP, TN, FN in opposite directions, tracing out a curve.

### One-Dimensional Intuition

Given a one-dimensional feature with two classes, we have only one variable. Any sample located at `x > t` (the threshold, also called alpha) is classified as positive.

If we place `t` somewhere, all samples greater than `t` on one side are labeled positive, and the others negative. If we change `t`, everything changes. Sweeping `t` across all values, we get different TPR and FPR values.

*(slide example)* For one particular threshold `t` in a 1-D example:

$$\text{TPR} = 0.5, \quad \text{FNR} = 0.5, \quad \text{FPR} = 0.12, \quad \text{TNR} = 0.88$$

### Procedure to Build an ROC Curve

1. Use a classifier that produces a continuous-valued score for each instance. The higher the score, the more likely the instance is positive.
2. Sort the instances in decreasing order by score.
3. Apply a threshold at each unique value of the score.
4. For each threshold, count TP, FP, TN, FN, then compute TPR = TP / (TP + FN) and FPR = FP / (FP + TN).

### Ten-Instance Construction Example (from the slides)

Ten scored instances and their true classes:

| Instance | Score | True Class |
|---|---|---|
| 1 | 0.95 | + |
| 2 | 0.93 | + |
| 3 | 0.87 | - |
| 4 | 0.85 | - |
| 5 | 0.85 | - |
| 6 | 0.85 | + |
| 7 | 0.76 | - |
| 8 | 0.53 | + |
| 9 | 0.43 | - |
| 10 | 0.25 | + |

Sweeping the threshold across every unique score produces the following counts:

| Class at that score | + | - | + | - | - | - | + | - | + | + | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Threshold ≥** | 0.25 | 0.43 | 0.53 | 0.76 | 0.85 | 0.85 | 0.85 | 0.87 | 0.93 | 0.95 | 1.00 |
| **TP** | 5 | 4 | 4 | 3 | 3 | 3 | 3 | 2 | 2 | 1 | 0 |
| **FP** | 5 | 5 | 4 | 4 | 3 | 2 | 1 | 1 | 0 | 0 | 0 |
| **TN** | 0 | 0 | 1 | 1 | 2 | 3 | 4 | 4 | 5 | 5 | 5 |
| **FN** | 0 | 1 | 1 | 2 | 2 | 2 | 2 | 3 | 3 | 4 | 5 |
| **TPR** | 1.0 | 0.8 | 0.8 | 0.6 | 0.6 | 0.6 | 0.6 | 0.4 | 0.4 | 0.2 | 0 |
| **FPR** | 1.0 | 1.0 | 0.8 | 0.8 | 0.6 | 0.4 | 0.2 | 0.2 | 0 | 0 | 0 |

Plotting TPR against FPR at each column yields the ROC curve.

### Walk-Through of the First Two Columns (narrative from class)

**Threshold t = 0.25**:
- All samples with score below 0.25 are predicted as negative.
- All samples with score greater than or equal to 0.25 are predicted as positive.
- With threshold 0.25, every sample's score meets the threshold, so every prediction is positive.
- How many are correct positives? **Five**.
- So TP = 5 and FP = 5, with TN = 0 and FN = 0.

**Threshold t = 0.43**:
- All samples with score below 0.43 are predicted as negative.
- All samples with score greater than or equal to 0.43 are predicted as positive.
- Among the positive predictions, how many were truly positive? **Four**.
- So TP = 4, and we read the other counts from the same column of the table.

Essentially, the classifier is better than random for small threshold values.

### Area Under the Curve

In general, if you have two classifiers with two ROC curves and you want to compare them, look at the **area under the curve (AUC)**. The classifier with the **maximum AUC** has the better ROC.

Reference values:

- **Ideal classifier**: AUC = **1**.
- **Random guess**: AUC = **0.5** (the diagonal).

### Region-Specific Comparison (M1 vs M2)

Suppose we have two models M1 and M2, with the diagonal representing random guessing.

- In the region of **small FPR**, M1 is better. The area under M1 is much higher than that under M2.
- For **large FPR**, M2 is better.
- If most of the area is before FPR equal to 0.5, the curve for M2 is lower in that region.

> No model consistently outperforms the other. Pick M1 or M2 based on the operating point you can tolerate.

### What Low FPR and High TPR Mean in Practice

After training, you should analyze your results: where your model works fine, where it does not, and on which kinds of samples.

- If the FPR for a model is **high**, the model predicts most negative samples as positive.
- When FPR is **low**, the model correctly classifies negative samples.

So if you have different models:
- When your false positive rate is low, you can use **M1**.
- When it is higher, you can use **M2**.

*(added code example)*

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_score = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)
auc = roc_auc_score(y_test, y_score)

plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
```

---

## 3. Building Classifiers with Imbalanced Training Sets

Resampling is the common technique to deal with the imbalanced class problem. The idea is to modify the distribution of the **training data** so that the rare class is well represented:

- **Undersample** the majority class.
- **Oversample** the rare class.
- **SMOTE** (Synthetic Minority Oversampling Technique).

Based on how we apply undersampling or oversampling, different sampling algorithms have been proposed. If you search, you will find many. At its simplest, we have **random undersampling** and **random oversampling**. We also have **SMOTE**, a specific oversampling technique that artificially creates new samples.

### Random Undersampling

Starting configuration:

- Class 1: **9000** instances.
- Class 2: **1000** instances.

We want balanced classes. Select 1000 instances **randomly** from class 1. Result: 1000 + 1000 instances. That is random undersampling. We reduce the majority class by randomly removing some samples.

### Random Oversampling

Same starting configuration. Duplicate instances of the minority class until it matches the majority. Result: 9000 + 9000 instances. We simply sample randomly with replacement, which means after random oversampling we have many duplicates of our data.

> Neither oversampling nor undersampling is guaranteed to solve the imbalance problem.

### SMOTE (Synthetic Minority Oversampling Technique)

The logic behind SMOTE is simple. Random oversampling replicates data, which just duplicates it. With SMOTE we oversample without replicating the exact same data. We generate new data.

**Procedure** (from the slides):

1. Take the difference between a minority instance and one of its nearest neighbors.
2. Multiply the difference by a random number in the interval [0, 1].
3. Add this scaled difference to the instance to generate a new instance along the line segment.
4. Continue to the next nearest neighbor, up to the k-th.
5. Repeat until enough new instances have been created.

*(reconstructed formula)* A synthetic sample is generated as:

$$x_{new} = x_i + \lambda \cdot (x_{nn} - x_i), \quad \lambda \in [0, 1]$$

where `x_i` is a minority instance and `x_nn` is one of its k-nearest neighbors in the minority class.

*(added code example)*

```python
from imblearn.over_sampling import SMOTE
from collections import Counter

smote = SMOTE(k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(Counter(y_train))
print(Counter(y_resampled))
```

### Imbalance in Text Classification

**Question from class**: SMOTE works because all samples are numeric. If we are working in text classification where samples are text and we also have a class imbalance problem, how do we handle it?

One student suggested translating the text into a vector representation, so we have a vector and can apply the method normally. **The issue**: when we convert text to a numeric representation, interpolating these vectors may not map back to a valid token.

**Alternatives**:

1. **Text perturbation**, such as replacing one word with its synonym.
2. **Generative augmentation**. With today's text generation ability, we can give a piece of text and generate thousands of similar texts.

### Comparison Table *(additional example)*

| Approach | What It Does | Pros | Cons |
|---|---|---|---|
| Random Undersampling | Drop majority samples | Simple, fast, balances quickly | Loses information |
| Random Oversampling | Duplicate minority samples with replacement | Simple, keeps all info | Many duplicates, overfitting risk |
| SMOTE | Synthesize along k-nearest-neighbor lines | New data, less overfitting | Numeric features only |
| Text perturbation (synonyms) | Swap words | Works on text | Limited variety |
| LLM-based augmentation | Generate similar texts | Rich variety | Quality varies, cost |

---

## 4. Outliers and Anomalies

### Definition

**Anomalies or outliers**: the set of data points that are considered different from the rest of the data.

### Outlier vs Noise

**Noise** is not a valid data point. An **outlier** is a valid data point but different from the norm of the data.

**Height example**: say the average height of people is 1.80 meters. If you find someone 3 meters tall, the value is valid, but it is different from the norm. Usually we want to find these outliers because they affect the mean, since most people are not 3 meters tall.

### Context Matters

Outliers may occur one in a thousand, and context is important. For example, freezing temperatures in July matter in context. One-in-a-thousand events still occur often if you have lots of data, so raw frequency is not enough.

An outlier can be important or a nuisance:

- **Important**: a very high blood pressure is important and we need to detect it.
- **Data error**: 200 pounds for a two-year-old might be too much.

### Causes of Anomalies

Anomalies or outliers come from:

1. **Data from different classes**. *(slide example)* Measuring the weights of oranges, with a few grapefruit mixed in.
2. **Natural variation**. Unusually tall people.
3. **Data errors**. The 200 pound two-year-old.

---

## 5. Approaches to Outlier Detection

There are different approaches to outlier detection.

### 5.1 Statistical Approach

Statistically, an outlier is an object that has a **low probability** with respect to the probability distribution model of the data. If you find the probability distribution for the data, the data points with the lowest probability are most likely outliers.

Methods like **Grubbs' test** can be used to check whether a dataset contains outliers.

*(added formula)* Grubbs' test statistic for the maximum deviation from the mean:

$$G = \frac{\max_i | x_i - \bar{x} |}{s}$$

where `x-bar` is the sample mean and `s` is the sample standard deviation.

### 5.2 Proximity-Based or Distance-Based

Anomalies are points far from other points. The **outlier score** of an object is the distance to its `k`-th nearest neighbor.

We give this outlier score to each sample and judge whether the sample is an outlier.

### 5.3 Density-Based

**Density-based outlier**: the outlier score of an object is the **inverse of the density** around the object.

Density can be defined in terms of the `k` nearest neighbors in two common ways.

**Definition 1: inverse of distance to the k-th neighbor.**

$$\text{density}(x, k) = \frac{1}{d_k(x)} = \frac{1}{\text{dist}(x, y_k)}$$

Here the inverse of distance defines density.

**Definition 2: inverse of the average distance to the `k` neighbors.**

$$\text{density}(x) = \frac{1}{\frac{1}{k} \sum_{i=1}^{k} d(x, y_i)}$$

> A point is an outlier **not just because it is far from others**, but because its density is **much less than that of its neighbors**.

**Limitation**: if there are regions of different density, this approach can have problems, since it works based on the region's density. If our dataset has some regions with low density, this simple method struggles.

### 5.4 Local Outlier Factor (LOF)

For the case where different regions have different densities, we use **LOF**, the **Local Outlier Factor**. For each point, compute the density of its local neighborhood. Then compute LOF as the **average of the ratios** of the density of sample `p` and the density of its nearest neighbors. Outliers are the points with the **largest LOF value**.

**Relative density** (from the slides):

Let `y_1, ..., y_k` be the `k` nearest neighbors of `x`.

$$\text{density}(x, k) = \frac{1}{\text{dist}(x, y_k)}$$

$$\text{relative density}(x, k) = \frac{\frac{1}{k} \sum_{i=1}^{k} \text{density}(y_i, k)}{\text{density}(x, k)} = \frac{\text{dist}(x, k)}{\frac{1}{k} \sum_{i=1}^{k} \text{dist}(y_i, k)}$$

Rule of thumb:

- If relative density `(x, k) >> 1`, then `x` is a **strong outlier**.
- If relative density `(x, k) < 1`, then `x` is **not** an outlier.

> *(slide remark)* In the plain nearest-neighbor approach, a point `p2` in a dense cluster next to a sparse cluster may not be flagged. The LOF approach, because it is relative to local density, flags both `p1` and `p2` when appropriate.

### 5.5 Clustering-Based

An object is a **cluster-based outlier** if it does **not strongly** belong to any cluster. We also have clustering-based rules: points far from cluster centers are outliers. First we have to find the cluster centers. And small clusters can also be considered outliers.

Formally, an object is an outlier if:

- It is **not close enough** to a **prototype-based cluster** center. Outliers themselves can impact the clustering that is produced.
- Its density is **too low** for a **density-based cluster**.

### 5.6 Reconstruction-Based Methods

We have another technique, **reconstruction-based methods**, which usually use **autoencoders**.

The underlying assumption is that there are patterns in the distribution of the **normal** class that can be captured using lower-dimensional representations. We reduce data to a lower-dimensional representation (for example with **PCA** or **autoencoders**), then project back and measure the **reconstruction error** for each object. Large reconstruction errors indicate anomalies.

**Formal reconstruction error**: we bring the object back to the original space and call it `x-hat`. The reconstruction error is the difference between the original object and `x-hat`, measured in the original space.

$$\text{Reconstruction Error}(x) = \| x - \hat{x} \|$$

- If this error is **low**, the object is not an outlier.
- If it is **high**, the object is an outlier.

**How to compute it**:

1. Use a dimensionality reduction model like **PCA**, or whatever dimensionality reduction methods you have.
2. Apply an **autoencoder**. The output tries to reproduce the input, and the hidden layer serves as a dimensionality reduction technique.

> **Bonus use**: autoencoders are also used for noise reduction.

*(added pipeline representation)*

```
Input x  -->  [Encoder]  -->  bottleneck z  -->  [Decoder]  -->  x_hat
                                                                  |
                                            reconstruction error = ||x - x_hat||
```

*(added code example)*

```python
import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

model.eval()
with torch.no_grad():
    x_hat = model(x)
    scores = ((x - x_hat) ** 2).sum(dim=1)  # high score = likely outlier
```

### 5.7 One-Class SVM (OCSVM)

**Problem setup**: we apply **one-class SVM** on our unlabeled data to find the **boundary of the normal data**, and all samples outside this boundary are labeled as outliers. The data may contain some outliers, and it does not contain class labels. The question is how to build a classifier given only one class.

#### Geometry of Standard SVM vs OCSVM (from the slides)

A standard **SVM** separates two classes with a maximum-margin hyperplane:

$$w \cdot x - b = 0$$

with parallel margin planes `w . x - b = 1` and `w . x - b = -1`.

An **OCSVM** has only one class. It separates the data from the **origin** with a hyperplane:

$$\vec{w} \cdot \vec{x} + b = 0$$

and maximizes the gap from the origin. Slack variables `ξ_i`, `ξ_j` allow some points to fall on the wrong side of the boundary.

#### How OCSVM Works

The method uses the **origin trick** together with a **Gaussian (RBF) kernel**:

$$\kappa(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$$

Properties of this kernel that make the origin trick work:

1. **Every point is mapped to a unit hypersphere**:

$$\kappa(x, x) = \langle \phi(x), \phi(x) \rangle = \|\phi(x)\|^2 = 1$$

2. **Every pair of points lies in the same orthant (quadrant)**:

$$\kappa(x, y) = \langle \phi(x), \phi(y) \rangle \geq 0$$

3. We aim to **maximize the distance** of the separating plane from the origin.

#### OCSVM Equations

Hyperplane equation in feature space:

$$f(x) = w \cdot \phi(x) - \rho$$

Decision rule:

- If `f(x) >= 0`, classify as **normal**.
- If `f(x) < 0`, classify as an **outlier**.

Here `φ` is the mapping to the high-dimensional kernel space, and the weight vector expands as:

$$w = \sum_{i=1}^{n} \alpha_i \, \phi(x_i)$$

The parameter `ν` (nu) is the assumed **fraction of outliers**. The optimization problem is:

$$\min_{w, \rho, \xi} \ \frac{1}{2} \|w\|^2 - \rho + \frac{1}{n\nu} \sum_{i=1}^{n} \xi_i$$

subject to

$$\langle w, \phi(x_i) \rangle \geq \rho - \xi_i, \quad \xi_i \geq 0$$

#### Effect of the `ν` Parameter (from the slides)

- With **`ν = 0.05`**, the decision region is **large and irregular**, with extending arms that reach out to capture most points as normal, so only a few far points are flagged as outliers.
- With **`ν = 0.2`**, the region is a **much smaller, tighter ellipse** around the dense core, so many more points fall outside and are flagged as outliers.

*(added code example)*

```python
from sklearn.svm import OneClassSVM

oc_svm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
oc_svm.fit(X_normal)
preds = oc_svm.predict(X_test)   # +1 for normal, -1 for outlier
```

---

## 6. Summary Table of Outlier Detection Methods *(added)*

| Method | Core Idea | Best Suited For |
|---|---|---|
| Statistical (Grubbs') | Lowest probability under assumed distribution | Univariate, known distribution |
| Distance-based (k-NN) | Far from `k`-th nearest neighbor | Uniform density regions |
| Density-based | Low local density | Reasonably uniform density |
| LOF | Density relative to neighbors | Regions of varying density |
| Clustering-based | Far from cluster center, or forms a very small cluster | Naturally clustered data |
| Reconstruction (PCA or autoencoder) | High reconstruction error | High-dimensional data, noise robust |
| One-class SVM | Outside the learned boundary of normal data | Unlabeled data with a single normal class |

---

## 7. Key Takeaways

> On imbalanced data, **accuracy is misleading**. Evaluate using precision, recall, F-measure, and task-specific metrics.

> **F-beta** weights recall against precision. Use larger beta when missing positives is costly (COVID detection, fraud).

> **ROC and AUC** let us compare models across all thresholds. Ideal AUC is 1, random AUC is 0.5. For small FPR, one model may win. For large FPR, another may win.

> **Sampling techniques**, random undersampling, random oversampling, and SMOTE, balance the training set. In text, prefer perturbation or LLM-based augmentation.

> **Outlier detection** has many paradigms: statistical, distance, density, LOF, clustering, reconstruction, and one-class SVM. Pick based on data density, dimensionality, and label availability.

> **Course note**: not every model you train should be deployed. Compare against random guessing, weigh the domain cost of false negatives versus false positives, and decide accordingly.
