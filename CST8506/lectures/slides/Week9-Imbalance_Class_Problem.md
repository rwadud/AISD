# CST8506 -- Advanced Machine Learning

## Week 9: Imbalanced Class Problem

**Dr. Abbas Akkasi -- Winter 2026**

> *These slides are adapted from materials originally developed by Pang-Ning Tan on his Data Mining Course.*

---

## Agenda

- Imbalance Class Problem
- Sampling Methods
- Anomaly Detection

---

## Class Imbalance Problem

- Lots of classification problems where the classes are skewed (more records from one class than another)
    - Credit card fraud
    - Intrusion detection
    - Defective products in manufacturing assembly line
    - COVID-19 test results on a random sample

- **Key Challenge:**
    - Evaluation measures such as **accuracy** are not well-suited for imbalanced class

---

## Confusion Matrix

|              | PREDICTED CLASS |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL CLASS — Class=Yes** | a | b |
| **ACTUAL CLASS — Class=No**  | c | d |

- **a:** TP (true positive)
- **b:** FN (false negative)
- **c:** FP (false positive)
- **d:** TN (true negative)

---

## Accuracy

|              | PREDICTED CLASS |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL CLASS — Class=Yes** | a (TP) | b (FN) |
| **ACTUAL CLASS — Class=No**  | c (FP) | d (TN) |

**Most widely-used metric:**

$$\text{Accuracy} = \frac{a + d}{a + b + c + d} = \frac{TP + TN}{TP + TN + FP + FN}$$

---

## Problem with Accuracy

- Consider a 2-class problem
    - Number of Class NO examples = 990
    - Number of Class YES examples = 10

- If a model predicts everything to be class NO, accuracy is 990/1000 = 99 %
    - This is misleading because this trivial model does not detect any class YES example
    - Detecting the rare class is usually more interesting (e.g., frauds, intrusions, defects, etc)

|              | PREDICTED CLASS |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL CLASS — Class=Yes** | 0 | 10 |
| **ACTUAL CLASS — Class=No**  | 0 | 990 |

---

## Which model is better?

**Model A:**

|              | PREDICTED       |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL — Class=Yes** | 0 | 10 |
| **ACTUAL — Class=No**  | 0 | 990 |

Accuracy: 99%

**Model B:**

|              | PREDICTED       |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL — Class=Yes** | 10 | 0 |
| **ACTUAL — Class=No**  | 500 | 490 |

Accuracy: 50%

---

## Which model is better?

**Model A:**

|              | PREDICTED       |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL — Class=Yes** | 5 | 5 |
| **ACTUAL — Class=No**  | 0 | 990 |

**Model B:**

|              | PREDICTED       |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL — Class=Yes** | 10 | 0 |
| **ACTUAL — Class=No**  | 500 | 490 |

---

## Alternative Measures

|              | PREDICTED CLASS |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL CLASS — Class=Yes** | a | b |
| **ACTUAL CLASS — Class=No**  | c | d |

$$\text{Precision (p)} = \frac{a}{a + c}$$

$$\text{Recall (r)} = \frac{a}{a + b}$$

$$\text{F-measure (F)} = \frac{2rp}{r + p} = \frac{2a}{2a + b + c}$$

---

## Alternative Measures (Example 1)

|              | PREDICTED CLASS |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL CLASS — Class=Yes** | 10 | 0 |
| **ACTUAL CLASS — Class=No**  | 10 | 980 |

$$\text{Precision (p)} = \frac{10}{10 + 10} = 0.5$$

$$\text{Recall (r)} = \frac{10}{10 + 0} = 1$$

$$\text{F-measure (F)} = \frac{2 \cdot 1 \cdot 0.5}{1 + 0.5} = 0.62$$

$$\text{Accuracy} = \frac{990}{1000} = 0.99$$

---

## Alternative Measures (Example 2)

|              | PREDICTED CLASS |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL CLASS — Class=Yes** | 1 | 9 |
| **ACTUAL CLASS — Class=No**  | 0 | 990 |

$$\text{Precision (p)} = \frac{1}{1 + 0} = 1$$

$$\text{Recall (r)} = \frac{1}{1 + 9} = 0.1$$

$$\text{F-measure (F)} = \frac{2 \cdot 0.1 \cdot 1}{1 + 0.1} = 0.18$$

$$\text{Accuracy} = \frac{991}{1000} = 0.991$$

---

## Which of these classifiers is better?

**Classifier A:**

|              | PREDICTED CLASS |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL — Class=Yes** | 40 | 10 |
| **ACTUAL — Class=No**  | 10 | 40 |

Precision (p) = 0.8, Recall (r) = 0.8, F-measure (F) = 0.8, Accuracy = 0.8

**Classifier B:**

|              | PREDICTED CLASS |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL — Class=Yes** | 40 | 10 |
| **ACTUAL — Class=No**  | 1000 | 4000 |

Precision (p) ≈ 0.04, Recall (r) = 0.8, F-measure (F) ≈ 0.08, Accuracy ≈ 0.8

---

## Measures of Classification Performance

|              | PREDICTED CLASS |              |
|--------------|:---------------:|:------------:|
|              | **Yes**         | **No**       |
| **ACTUAL — Yes** | TP | FN |
| **ACTUAL — No**  | FP | TN |

$$Accuracy = \frac{TP + TN}{TP + FN + FP + TN}$$

$$ErrorRate = 1 - accuracy$$

$$Precision = \text{Positive Predictive Value} = \frac{TP}{TP + FP}$$

$$Recall = Sensitivity = TP\ Rate = \frac{TP}{TP + FN}$$

$$Specificity = TN\ Rate = \frac{TN}{TN + FP}$$

$$FP\ Rate = \alpha = \frac{FP}{TN + FP} = 1 - specificity$$

$$FN\ Rate = \beta = \frac{FN}{FN + TP} = 1 - sensitivity$$

$$Power = sensitivity = 1 - \beta$$

---

## Alternative Measures (TPR/FPR)

**Classifier A:**

|              | PREDICTED CLASS |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL — Class=Yes** | 40 | 10 |
| **ACTUAL — Class=No**  | 10 | 40 |

Precision (p) = 0.8, TPR = Recall (r) = 0.8, FPR = 0.2, F-measure (F) = 0.8, Accuracy = 0.8

$$\frac{TPR}{FPR} = 4$$

**Classifier B:**

|              | PREDICTED CLASS |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL — Class=Yes** | 40 | 10 |
| **ACTUAL — Class=No**  | 1000 | 4000 |

Precision (p) = 0.038, TPR = Recall (r) = 0.8, FPR = 0.2, F-measure (F) = 0.07, Accuracy = 0.8

$$\frac{TPR}{FPR} = 4$$

---

## Which of these classifiers is better?

**Classifier A:**

|              | PREDICTED CLASS |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL — Class=Yes** | 10 | 40 |
| **ACTUAL — Class=No**  | 10 | 40 |

Precision (p) = 0.5, TPR = Recall (r) = 0.2, FPR = 0.2, F-measure = 0.28

**Classifier B:**

|              | PREDICTED CLASS |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL — Class=Yes** | 25 | 25 |
| **ACTUAL — Class=No**  | 25 | 25 |

Precision (p) = 0.5, TPR = Recall (r) = 0.5, FPR = 0.5, F-measure = 0.5

**Classifier C:**

|              | PREDICTED CLASS |              |
|--------------|:---------------:|:------------:|
|              | **Class=Yes**   | **Class=No** |
| **ACTUAL — Class=Yes** | 40 | 10 |
| **ACTUAL — Class=No**  | 40 | 10 |

Precision (p) = 0.5, TPR = Recall (r) = 0.8, FPR = 0.8, F-measure = 0.61

---

## ROC (Receiver Operating Characteristic)

- A graphical approach for displaying trade-off between detection rate (TPR) and false alarm rate (FPR)
- Developed in 1950s for signal detection theory to analyze noisy signals
- ROC curve plots TPR against FPR
    - Performance of a model represented as a point in an ROC curve

---

## ROC Curve

(TPR, FPR):

- (0,0): declare everything to be negative class
- (1,1): declare everything to be positive class
- (1,0): ideal

- Diagonal line:
    - Random guessing
    - Below diagonal line:
        - prediction is opposite of the true class

---

## ROC (Receiver Operating Characteristic)

- To draw ROC curve, classifier must produce **continuous-valued** output
    - Outputs are used to **rank test records**, from the most likely positive class record to the least likely positive class record
    - By using **different thresholds** on this value, we can create different variations of the classifier with TPR/FPR tradeoffs
- Many classifiers produce only discrete outputs (i.e., predicted class)
    - How to get continuous-valued outputs?
        - Decision trees, rule-based classifiers, neural networks, Bayesian classifiers, k-nearest neighbors, SVM

---

## Example: Decision Trees

A decision tree splits on features (e.g., x2 < 12.63, x1 < 13.29, x1 < 6.56, x2 < 17.35, x1 < 2.15, x1 < 7.24, x2 < 8.64, x1 < 12.11, x2 < 1.38, x1 < 18.88) and produces leaf labels. By assigning each leaf a probability of being positive (e.g., 0.107, 0.059, 0.220, 0.071, 0.164, 0.143, 0.669, 0.727, 0.271, 0.654, 0), the tree gives **continuous-valued outputs** rather than just discrete class labels.

---

## ROC Curve Example

Using the same decision tree with continuous-valued outputs on a 2D training set:

**At α = 0.3:**

|              | Predicted Class |              |
|--------------|:---------------:|:------------:|
|              | **Class o**     | **Class +**  |
| **Actual — Class o** | 645 | 209 |
| **Actual — Class +** | 298 | 948 |

**At α = 0.7:**

|              | Predicted Class |              |
|--------------|:---------------:|:------------:|
|              | **Class o**     | **Class +**  |
| **Actual — Class o** | 181 | 673 |
| **Actual — Class +** | 78 | 1168 |

---

## ROC Curve Example (1-D)

- 1-dimensional data set containing 2 classes (positive and negative)
- Any points located at x > t is classified as positive

At threshold t:

TPR = 0.5, FNR = 0.5, FPR = 0.12, TNR = 0.88

---

## How to Construct an ROC curve

| Instance | Score | True Class |
|:--------:|:-----:|:----------:|
| 1  | 0.95 | + |
| 2  | 0.93 | + |
| 3  | 0.87 | - |
| 4  | 0.85 | - |
| 5  | 0.85 | - |
| 6  | 0.85 | + |
| 7  | 0.76 | - |
| 8  | 0.53 | + |
| 9  | 0.43 | - |
| 10 | 0.25 | + |

- Use a classifier that produces a continuous-valued score for each instance
    - The more likely it is for the instance to be in the + class, the higher the score
- Sort the instances in decreasing order according to the score
- Apply a threshold at each unique value of the score
- Count the number of TP, FP, TN, FN at each threshold
    - TPR = TP / (TP + FN)
    - FPR = FP / (FP + TN)

---

## How to construct an ROC curve

| Class       | + | - | + | - | - | - | + | - | + | + |     |
|-------------|---|---|---|---|---|---|---|---|---|---|-----|
| **Threshold ≥** | 0.25 | 0.43 | 0.53 | 0.76 | 0.85 | 0.85 | 0.85 | 0.87 | 0.93 | 0.95 | 1.00 |
| **TP**  | 5 | 4 | 4 | 3 | 3 | 3 | 3 | 2 | 2 | 1 | 0 |
| **FP**  | 5 | 5 | 4 | 4 | 3 | 2 | 1 | 1 | 0 | 0 | 0 |
| **TN**  | 0 | 0 | 1 | 1 | 2 | 3 | 4 | 4 | 5 | 5 | 5 |
| **FN**  | 0 | 1 | 1 | 2 | 2 | 2 | 2 | 3 | 3 | 4 | 5 |
| **TPR** | 1 | 0.8 | 0.8 | 0.6 | 0.6 | 0.6 | 0.6 | 0.4 | 0.4 | 0.2 | 0 |
| **FPR** | 1 | 1 | 0.8 | 0.8 | 0.6 | 0.4 | 0.2 | 0.2 | 0 | 0 | 0 |

**ROC Curve:** plot TPR vs FPR using the above values.

---

## Using ROC for Model Comparison

- No model consistently outperforms the other
    - M₁ is better for small FPR
    - M₂ is better for large FPR

- Area Under the ROC curve (AUC)
    - Ideal:
        - Area = 1
    - Random guess:
        - Area = 0.5

---

## Building Classifiers with Imbalanced Training Set

- Resampling is the common technique to deal with the imbalanced class problem
- Modify the distribution of **training data** so that rare class is well-represented in training set
    - Undersample the majority class
    - Oversample the rare class
    - SMOTE (Synthetic Minority Oversampling Technique)

---

## Undersampling

- Class 1 -- 9000 instances
- Class 2 -- 1000 instances

- Solution: make the classes balanced (equal size)
- How?
- Select 1000 instances randomly from class 1
- Good approach?

Result: 1000 instances from class 1 + 1000 instances from class 2.

---

## Oversampling

- Class 1 -- 9000 instances
- Class 2 -- 1000 instances

- Solution: make the classes balanced (equal size)
- How?
- Duplicate instances of minority class

Result: 9000 instances from class 1 + 9000 instances from class 2.

---

## SMOTE - Logic

For each minority class instance, add new synthetic instances along the line segments joining k minority nearest neighbors.

1. Take difference between an instance and the nearest neighbor
2. Multiply by a random number in [0, 1]
3. Add this difference to the instance to generate new instance along the line segment
4. Continue on with next NN up to kNN
5. Repeat until enough number of instances are created

---

## Anomaly/Outlier Detection

- What are anomalies/outliers?
    - The set of data points that are considerably **different** than the remainder of the data
- Natural implication is that anomalies are **relatively rare**
    - One in a thousand occurs often if you have lots of data
    - Context is important, e.g., freezing temps in July
- Can be important or a nuisance
    - Unusually high blood pressure
    - 200 pound, 2 year old

---

## Causes of Anomalies

- Data from different classes
    - Measuring the weights of oranges, but a few grapefruit are mixed in
- Natural variation
    - Unusually tall people
- Data errors
    - 200 pound 2 year old

---

## Anomaly Detection Techniques

- Statistical Approaches
    - An outlier is an object that has a low probability with respect to a probability distribution model of the data. E.g., Grubbs' Test
- Proximity-based
    - Anomalies are points far away from other points
    - The outlier score of an object is the distance to its kth nearest neighbor
- Density-Based (e.g., Local Outlier Factor (LOF) method)
- Clustering-based
    - Points far away from cluster centers are outliers
    - Small clusters are outliers
- Reconstruction Based
- One class SVM

---

## Density-Based Approaches

- **Density-based Outlier:** The outlier score of an object is the inverse of the density around the object.
    - Can be defined in terms of the k nearest neighbors
    - One definition: Inverse of distance to kth neighbor
    - Another definition: Inverse of the average distance to k neighbors

- A point is an outlier **not just because it is far from others**, but because it is **much less dense than its neighbors**.
- If there are regions of different density, this approach can have problems

---

## Relative Density

- Consider the density of a point relative to that of its k nearest neighbors
- Let $y_1, \ldots, y_k$ be the $k$ nearest neighbors of $x$

$$density(x, k) = \frac{1}{dist(x, k)} = \frac{1}{dist(x, y_k)}$$

$$relative\ density(x, k) = \frac{\sum_{i=1}^{k} density(y_i, k) / k}{density(x, k)} = \frac{dist(x, k)}{\sum_{i=1}^{k} dist(y_i, k) / k}$$

- Can use average distance instead

- If relative density(x, k) ≫ 1, then x is strong outlier
- If relative density(x, k) < 1, then x is not an outlier

---

## Relative Density-based: LOF Approach

- For each point, compute the density of its local neighborhood
- Compute **local outlier factor (LOF)** of a sample *p* as the average of the ratios of the density of sample *p* and the density of its nearest neighbors
- Outliers are points with largest LOF value

In the NN approach, p₂ is not considered as outlier, while LOF approach finds both p₁ and p₂ as outliers.

---

## Clustering-Based Approaches

An object is a cluster-based outlier if it **does not strongly** belong to any cluster.

- For prototype-based clusters, an object is an outlier if it is not close enough to a cluster center.
    - Outliers can impact the clustering produced
- For density-based clusters, an object is an outlier if its **density is too low**

---

## Reconstruction-Based Approaches

- Based on assumptions **there are patterns in the distribution of the normal class** that can be captured using lower-dimensional representations
- Reduce data to lower dimensional data
    - E.g. Use Principal Components Analysis (PCA) or Auto-encoders
- Measure the **reconstruction error** for each object
    - The difference between original and reduced dimensionality version

---

## Reconstruction Error

- Let **x** be the original data object
- Find the representation of the object in a lower dimensional space
- Project the object back to the original space
- Call this object $\hat{x}$

$$\text{Reconstruction Error}(x) = \|x - \hat{x}\|$$

- Objects with large reconstruction errors are anomalies

---

## One Class SVM - OCSVM

- Uses an SVM approach to classify normal objects
- Uses the given data to construct such a model
- This data may contain outliers
- But the data does not contain class labels
- How to build a classifier given one class?

---

## One Class SVM - OCSVM (Geometry)

Standard **SVM** separates two classes (e.g., blue vs green) with a maximum-margin hyperplane $w \cdot x - b = 0$, with parallel margins $w \cdot x - b = 1$ and $w \cdot x - b = -1$.

**OCSVM** has only one class. It separates the data from the **origin** with a hyperplane $\vec{w} \cdot \vec{x} + b = 0$, maximizing the gap from the origin. Slack variables $\xi_i, \xi_j$ allow some points to lie on the wrong side of the boundary.

---

## How Does OCSVM Work?

- Uses the **"origin" trick**
- Use a Gaussian kernel

$$\kappa(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$$

- Every point mapped to a unit hypersphere

$$\kappa(x, x) = \langle \phi(x), \phi(x) \rangle = \|\phi(x)\|^2 = 1$$

- Every point in the same orthant (quadrant)

$$\kappa(x, y) = \langle \phi(x), \phi(y) \rangle \geq 0$$

- Aim to maximize the distance of the separating plane from the origin

---

## Equations for OCSVM

- Equation of hyperplane

$$f(x) = w \cdot \phi(x) - \rho$$

  - **If** $f(x) \geq 0 \rightarrow$ normal
  - **If** $f(x) < 0 \rightarrow$ outlier

- $\phi$ is the mapping to high dimensional space
- Weight vector is (direction of the separating surface)

$$w = \sum_{i=1}^{n} \alpha_i \phi(x_i)$$

- $\nu$ is fraction of outliers
- Optimization condition is the following

$$\min_{w, \rho, \xi} \frac{1}{2} \|w\|^2 - \rho + \frac{1}{n\nu} \sum_{i=1}^{n} \xi_i,$$

$$\text{subject to: } \langle w, \phi(x_i) \rangle \geq \rho - \xi_i, \quad \xi_i \geq 0$$

---

## Finding Outliers with a One-Class SVM

- Decision boundary with $\nu = 0.05$ and $\nu = 0.2$

With $\nu = 0.05$, the decision region is large and irregular, with extending arms that reach out to capture most points as normal, leaving only a few far points flagged as outliers. With $\nu = 0.2$, the region is a much smaller, tighter ellipse around the dense core, so many more points fall outside and are flagged as outliers.
