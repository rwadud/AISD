# Lecture 2: Support Vector Machines (SVM)

## Recap: PCA vs. LDA Component Limits

Before introducing SVMs, the lecturer reviewed the key difference between PCA and LDA and clarified the rules for choosing the number of components in each method.

### LDA Recap

- LDA takes the class label into account when computing components. PCA does not consider class labels at all. In PCA, you are just taking the components, not considering the class. In LDA, everything is class based.
- **LDA objective** (two goals simultaneously):
  1. Maximize the distance **between** classes.
  2. Minimize the distance **within** each class (points in the same class should be closer together).
- LDA considers all points in each class when computing the class means. For example, if class one has five points, the class mean is computed from those five points. The same is done for class two.

### Maximum Number of Components

| Method | Maximum `n_components` | Rule |
|--------|----------------------|------|
| PCA | $\min(\text{n}\_\text{samples},\; \text{n}\_\text{features})$ | The lesser of the number of samples or the number of features |
| LDA | $\min(\text{n}\_\text{classes} - 1,\; \text{n}\_\text{features})$ | The lesser of (number of classes minus one) or the number of features |

**Why classes minus one for LDA?** To separate two classes you need one line. To separate three classes you need two lines. In general, you need $k - 1$ separators for $k$ classes. Therefore the maximum number of LDA components is $k - 1$ (or the number of features, whichever is less).

**Worked example:** 2 classes, 8 features.
- $\text{n}\_\text{classes} - 1 = 2 - 1 = 1$
- $\text{n}\_\text{features} = 8$
- $\min(1, 8) = 1$ component

> **Course note:** In the exam, you will be given a specific number of features, classes, and samples, and asked how many PCA components and how many LDA components you can create. For example: "We have 5 classes, 20 features, and some number of samples. How many PCA components can you make, and how many LDA components can you make?" Without knowing these concepts, you cannot get the marks.

> **Course note:** Finishing the labs is easy. The objective is for you to at least try to document the parameters, because you will learn from it. Go to the scikit-learn documentation and make sure you know what each parameter means.

> **Course note:** In Lab 2, the PCA math must be done by hand with a specific set of numbers. The PCA math was covered last week. Try to do it by yourself. That will be an exam question.

---

## Introduction to Support Vector Machines

### Classification Methods Covered So Far

1. **Decision Trees**
2. **Logistic Regression**: originally used to predict a continuous value (e.g., house price). The predicted value is sent through a **sigmoid function**. If the output is $> 0.5$, the point is assigned to one class. If $< 0.5$, it is assigned to the other class. This makes logistic regression usable for classification.
3. **Support Vector Machines (SVM)**: the topic of this lecture.

> SVMs are normally considered an advanced technique. The math is not trivial, but the intuition is straightforward.

### The Core Idea: Separating Classes with a Line

Given two classes (blue points and red points) in two dimensions, we want to draw a line (the **decision boundary**) that separates them. Many different lines can separate the same two classes, and all of them are "valid" separators. The challenge is to find the **best** one.

Always keep in mind: we work with 2D pictures for illustration, but in practice data can have three, a hundred, or more dimensions. Drawing and finding the separator in high dimensions is not easy.

### Why Not Just Any Line? The Overfitting Problem

- A line drawn very close to the red class keeps all blue points safe, but red points are at risk because a new red point could appear on the wrong side. This line is **overfitted to the red class**.
- Similarly, a line drawn very close to the blue class is overfitted to the blue class.
- We need a line that is **not overfitted** and **not biased** toward either class.

---

## The Margin Concept

### What Is a Margin?

The SVM approach is to create a **margin**, a band of empty space between the two classes. Multiple margins can separate the classes, but SVMs aim to find the one that is as **wide as possible**.

- A narrow margin can still separate the classes, but a wider margin gives more room, making the classifier more robust to new, unseen points.
- The **objective** of SVM is to **maximize this margin**.

### Support Vectors

- **Support vectors**: the data points that are closest to the decision boundary (the hyperplane). These are the reference points used to define the margin.
- Once the model is trained, only the support vectors matter. All other points are irrelevant to the position of the boundary.
- Support vectors **determine the position and orientation** of the hyperplane.
- The maximum possible margin is defined by the support vectors.

> In simple terms: find the widest possible gap between the two classes. The points that sit right on the edge of that gap are the support vectors.

---

## Separators Across Dimensions

The type of separator depends on the number of dimensions:

| Dimensions | Separator | Name |
|-----------|-----------|------|
| 1 | A point (dot) | Threshold |
| 2 | A line | Line |
| 3 | A flat surface | Plane |
| >3 | A generalized flat surface | **Hyperplane** |

The **hyperplane** is the general term for the separator in SVM. In higher dimensional spaces the separator is a hyperplane, not a line.

### What Are Support Vectors?

- The points **closest to the hyperplane**.
- They **influence the position and orientation** of the hyperplane.
- Using the support vectors, we **maximize the classification margin**.

### Can We Check Separability Before Training?

In real life, data points will not be neatly clustered. They will be scattered. There is no way to determine whether a separating hyperplane exists before actually building the model. However:

- We can use dimensionality reduction techniques (PCA, LDA from last week) to reduce to 2D and **visualize** the data.
- One key reason for dimensionality reduction is visualization: unless we can see the data, we have no idea whether a linear separator will work.
- After fitting, we evaluate by checking classification **accuracy**.

### SVM as a Supervised Method

**Support Vector Machine** is a **supervised learning** method. It requires class labels. It can be used for both classification and regression tasks, but is most commonly used for **classification**. SVM works best on **small, complex datasets**.

---

## Mathematical Foundation of SVM

> This is not as simple as KNN or naive Bayes. There is math in machine learning, and we cannot avoid it. Do not get discouraged. If you do not get it in one stretch, review it and come back with questions.

### The Prediction Problem

Given:
- A trained model (the green line / hyperplane).
- A new, unseen point (the black point).

When we see the picture, we visually know the black point belongs to the blue class. But we do not have the picture. We need a mathematical way to determine which side of the hyperplane the new point falls on.

**Task:** Determine which side of the hyperplane the new point falls on without visual inspection.

The problem with "find the distance to the line" is that a line has infinitely many points on it. We do not know which specific point on the line to measure the distance to.

### Step 1: Create a Perpendicular Vector $\mathbf{w}$

We need a criterion that works for any point anywhere in the feature space, not just for one specific location. The next time we get a new point, it could be anywhere. The criterion should be applicable for any point anywhere in this dimensional space.

The standard mathematical approach: create a vector $\mathbf{w}$ that is **perpendicular** to the decision boundary (the green line). This vector $\mathbf{w}$ has a direction (because a vector represents a direction in space) and serves as the normal to the hyperplane.

### Step 2: Represent the New Point as a Vector $\mathbf{x}$

The new point $\mathbf{x}$ is also treated as a vector from the origin.

### Step 3: Project $\mathbf{x}$ onto $\mathbf{w}$ Using the Dot Product

To measure which side of the line $\mathbf{x}$ falls on, we project $\mathbf{x}$ onto $\mathbf{w}$:

$$\text{projection} = \mathbf{x} \cdot \mathbf{w}$$

**Assumption:** The perpendicular distance from the green line (hyperplane) to the origin is some constant $C$.

### Step 4: Decision Rule Based on the Dot Product

| Condition | Meaning |
|----------|---------|
| $\mathbf{x} \cdot \mathbf{w} = C$ | $\mathbf{x}$ is exactly **on** the green line |
| $\mathbf{x} \cdot \mathbf{w} > C$ | $\mathbf{x}$ is on the **red** (positive) side |
| $\mathbf{x} \cdot \mathbf{w} < C$ | $\mathbf{x}$ is on the **blue** (negative) side |

### Step 5: Rearranging into Standard Form

Starting from the positive class condition:

$$\mathbf{x} \cdot \mathbf{w} > C$$

Move $C$ to the left side:

$$\mathbf{x} \cdot \mathbf{w} - C > 0$$

Now define $b = -C$ (we rename $-C$ as $b$ to match the standard line equation $y = mx + b$):

$$\mathbf{x} \cdot \mathbf{w} + b > 0$$

### The SVM Decision Function

$$
y = \begin{cases} +1 & \text{if } \mathbf{x} \cdot \mathbf{w} + b \geq 0 \\\ -1 & \text{if } \mathbf{x} \cdot \mathbf{w} + b < 0 \end{cases}
$$

This function classifies any new point into one of two classes. We do not have a third class here, so one or the other, we have to assign either to the red or to the blue. This works similarly to cluster assignment in k-means (assign to the nearest centroid), except here we use the sign of the dot product plus bias.

> **Q: Is the separator always a line?**
> No. In two dimensions it looks like a line, but mathematically it is a **plane** (or hyperplane in higher dimensions). The explanation uses 2D for simplicity, but the real separator is a hyperplane in the full feature space.

### Training vs. Prediction

- The green line (hyperplane) is created during **training**. Training is a separate phase.
- Once the model is finalized, it is used for **prediction**. The hyperplane does not change with each new prediction.
- The correct model is found through a trade-off between **accuracy** and **complexity**.

> **Course note:** Grid search and random search for hyperparameter tuning will be covered in a later class.

---

## Maximizing the Margin: The Optimization Problem

### Setting Up the Parallel Boundary Lines

The core concept: find the separator such that two parallel lines drawn on either side of it produce the **maximum margin** $D$.

Two parallel boundary lines are defined:

$$L_1: \quad \mathbf{w} \cdot \mathbf{x} + b = -1$$
$$L_2: \quad \mathbf{w} \cdot \mathbf{x} + b = +1$$

**Why use the values +1 and -1?** For convenience. Any constant would work (it is just a scaling factor), but using 1 simplifies the math. This is similar to how we normalized eigenvalues to 1 in PCA for convenience.

### Class Constraints

- **Red (positive) points** must satisfy: $\mathbf{w} \cdot \mathbf{x}_i + b \geq +1$
- **Blue (negative) points** must satisfy: $\mathbf{w} \cdot \mathbf{x}_i + b \leq -1$

These two conditions are combined into a single constraint using the label $y_i \in \{+1, -1\}$:

$$y_i \cdot (\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1$$

When $y_i = +1$ (red), this reduces to $\mathbf{w} \cdot \mathbf{x}_i + b \geq 1$. When $y_i = -1$ (blue), it reduces to $\mathbf{w} \cdot \mathbf{x}_i + b \leq -1$.

### Distance Between Two Parallel Planes

For two parallel planes $\mathbf{w} \cdot \mathbf{x} + b = C_1$ and $\mathbf{w} \cdot \mathbf{x} + b = C_2$, the distance formula is:

$$D = \frac{|C_2 - C_1|}{\|\mathbf{w}\|}$$

Substituting $C_1 = -1$ and $C_2 = +1$:

$$D = \frac{|1 - (-1)|}{\|\mathbf{w}\|} = \frac{2}{\|\mathbf{w}\|}$$

where $\|\mathbf{w}\| = \sqrt{w_1^2 + w_2^2 + \cdots + w_n^2}$ is the Euclidean norm of $\mathbf{w}$.

> You do not need to prove this formula. In an advanced math course you would derive it, but for this course, just know the formula.

### The SVM Optimization (Hard Margin)

$$\text{Maximize} \quad D = \frac{2}{\|\mathbf{w}\|}$$

$$\text{subject to} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \quad \forall \; i$$

This is equivalent to **minimizing** $\|\mathbf{w}\|$ (or $\frac{1}{2}\|\mathbf{w}\|^2$ for mathematical convenience) *(added)*.

### Interpreting X and Y Axes in Plots

When we see a scatter plot with X and Y axes and colored dots:
- The **color** represents the **class** label.
- The **X and Y axes** represent the **features** (e.g., length and width), not the classes.

### Multiclass SVM

For more than two classes, SVM uses a **one versus one** strategy:
- 3 classes → 3 pairwise comparisons (class 1 vs. 2, class 1 vs. 3, class 2 vs. 3), each with its own separating line.
- In general, $k$ classes require $\binom{k}{2} = \frac{k(k-1)}{2}$ pairwise classifiers *(added)*.

---

## Maximum Margin Classifier (Hard Margin)

- Also called **hard margin SVM**.
- Does **not accept any misclassifications**. Every training point must be on the correct side of its corresponding boundary line.
- Requires the classes to be **perfectly linearly separable**.

---

## Support Vector Classifier (Soft Margin)

### The Problem with Hard Margins

In real life, classes are almost never perfectly linearly separable. There will be overlap. The hard margin requirement is too strict.

### Accepting Some Misclassifications

The **support vector classifier** relaxes the hard margin condition:
- We still want to maximize the margin.
- But we are willing to **accept some misclassifications**.
- We set a parameter that controls **how many misclassifications** we are willing to tolerate (e.g., 1%, 2%, 5%).

**Concrete example:** On one side, three blue points might be misclassified. On the other side, two red points might be misclassified. But if this is the best margin we can get, we go with it, as long as the number of misclassifications is within the tolerance we set.

The optimization becomes:
- Maximize the margin.
- Subject to: some points may be on the wrong side of the boundary, within a tolerance we define.
- The previous condition (maximize margin) is still there, but now we also have a constraint on how many misclassifications are acceptable. Based on that parameter, the margin will be created.

> The difference: the **maximum margin classifier** (hard margin) allows zero misclassifications, and the classes must be perfectly linearly separable. The **support vector classifier** (soft margin) uses a linear classifier but is willing to accept some misclassifications. How much misclassification we are willing to accept, we set that.

---

## Support Vector Machine (Nonlinear Classification)

### When No Linear Separator Works

In some cases, the classes are arranged so that **no linear boundary** can separate them (e.g., one class surrounds the other in a circular pattern).

### The Projection Idea

The approach: **project the data into a higher dimensional space** where a linear separator might exist.

- If 2 dimensions are not enough, try 3, 4, 5, 20, 200, or more.
- Keep projecting until the classes become linearly separable in the higher dimensional space.
- This transforms a non-separable problem into a separable one.

### The Computational Problem

Projecting 100 features into a potentially infinite dimensional space is **computationally very expensive**. It is not a simple operation like squaring or cubing values.

### The Kernel Trick

**Kernel trick**: a mathematical technique that computes the dot product in the higher dimensional space **without actually performing the projection**.

- The kernel function gives the **same results** as if the data were projected to higher dimensions, without the computational cost of actually doing it.
- For example, a kernel function might give the same answer as projecting 10 feature values into a 10,000 dimensional space, without ever creating that 10,000 dimensional representation.

> **Key understanding:** The kernel function does the dot product calculations and returns the results you would get if you projected into higher dimensions, without actually projecting. That is the kernel trick.

### Important Distinction: Projection Space vs. Feature Space

- The **projection space** (the higher dimensional space used by the kernel) is **not the same** as the original **feature space**.
- If you have 5 features, that does not mean you project into 5 dimensions. The projection space could be 20, 200, or infinite dimensions.
- The number of features and the dimensionality of the projection space are **two completely different things**.

---

## Kernel Types

### Available Kernels in scikit-learn

| Kernel | Description | Key Parameter |
|--------|-------------|---------------|
| `linear` | No kernel transformation, creates a linear boundary | None |
| `poly` | Polynomial boundary (e.g., quadratic curve) | `degree` |
| `rbf` | Radial Basis Function, projects to potentially infinite dimensions | `gamma` |
| `sigmoid` | Sigmoid shaped boundary | `gamma`, `coef0` |

```python
# Linear SVM (added)
from sklearn.svm import SVC
clf_linear = SVC(kernel='linear')

# Polynomial SVM (added)
clf_poly = SVC(kernel='poly', degree=3)

# RBF SVM (added)
clf_rbf = SVC(kernel='rbf', gamma='scale')
```

### Linear Kernel

- Does not apply any kernel transformation.
- Creates a straight line (or flat hyperplane) as the separator.
- Best for data that is already linearly separable.

### Polynomial Kernel

- Creates a curved (polynomial) decision boundary.
- You specify the **degree** of the polynomial.
- A degree 2 polynomial boundary looks like a quadratic curve.
- Generally creates more support vectors than the linear kernel.

### Radial Basis Function (RBF) Kernel

- The **most widely used** kernel.
- Keeps projecting (conceptually) to higher and higher dimensions **until the classes become separable**.
- Can theoretically project to **infinite** dimensions.
- Very powerful, but also **very expensive** computationally.
- Default kernel in scikit-learn's `SVC`.

> **RBF will keep projecting until the classes are separable.** It can go to infinity. That is why it can be very expensive.

### Comparing Kernels: Wine Dataset Example (After LDA with 3 Classes)

| Kernel | Support Vectors | Observation |
|--------|----------------|-------------|
| Linear | ~8 points | Few support vectors, good classification for linearly separable data |
| Polynomial | More points | More support vectors than linear |
| RBF | Points around class boundaries | Finds boundary points around each class |

- **Linear** uses the fewest support vectors. In a linearly separable case, this is the best choice. The rest of the points are not support vectors.
- **RBF** identifies points around the edges of each class. This is useful when classes are not linearly separable (e.g., circular arrangements).
- When data is linearly separable, using RBF is **unnecessary** and wasteful.

> For simple, linearly separable data, do not use SVM (or at least use a linear kernel). Simpler methods like naive Bayes or logistic regression might give equal or even better accuracy.

---

## SVM Hierarchy Summary

| Level | Name | Margin | Separator | Misclassifications |
|-------|------|--------|-----------|-------------------|
| 1 | Maximum Margin Classifier | Hard margin | Linear (hyperplane) | None allowed |
| 2 | Support Vector Classifier | Soft margin | Linear (hyperplane) | Some allowed (controlled by C) |
| 3 | Support Vector Machine | Soft margin | Nonlinear (kernel) | Some allowed (controlled by C) |

- **Maximum Margin Classifier** = hard margin, no misclassifications, classes must be linearly separable.
- **Support Vector Classifier** = soft margin, allows some misclassifications, still uses a linear separator.
- **Support Vector Machine** = support vector classifier + nonlinear kernel (RBF, polynomial, sigmoid, etc.).

There is **no shortcut** to decide which kernel is best. If you can visualize the data in 2D, you can get an idea. Otherwise, in a 50-dimensional space, you cannot see how the data looks, and you must experiment.

---

## Key Hyperparameters: C and Gamma

### The Balancing Act

When creating an SVM model, multiple factors must be balanced simultaneously:
1. The margin should be **maximized**.
2. Some misclassifications may be **accepted** (soft margin).
3. The model should **not overfit** (perform perfectly on training data but poorly on new data).
4. The model should **not underfit** (perform poorly even on training data).

### Parameter C (Regularization)

**C** controls how many misclassifications the model is willing to accept. C and the regularization parameter are inversely proportional to each other (one is one over the other).

| C Value | Effect |
|---------|--------|
| **Small C** | More misclassifications tolerated → wider margin → higher regularization → risk of **underfitting** |
| **Large C** | Fewer misclassifications tolerated → narrower margin → lower regularization → risk of **overfitting** |

- **Overfitting:** The boundary is very close to the training points. The model accommodates every single training point perfectly, but fails when new points arrive.
- **Underfitting:** The margin is too wide, and even the training data is not classified well. The model is useless.

### Parameter Gamma

**Gamma** controls how close the decision boundary should be to the data points of each class. It determines how well the model fits the training data.

| Gamma Value | Effect |
|-------------|--------|
| **Low gamma** | Boundary is far from data points → smoother, simpler decision boundary → risk of **underfitting** |
| **High gamma** | Boundary is very close to data points → complex, tightly fitted boundary → risk of **overfitting** |

```python
# Example: tuning C and gamma (added)
from sklearn.svm import SVC

# Risk of underfitting: small C, low gamma
clf_under = SVC(kernel='rbf', C=0.01, gamma=0.001)

# Risk of overfitting: large C, high gamma
clf_over = SVC(kernel='rbf', C=1000, gamma=10)

# Balanced (common starting point)
clf_balanced = SVC(kernel='rbf', C=1.0, gamma='scale')
```

> **Course note:** Grid search and random search will be covered in a future class for finding the optimal C and gamma values.

---

## Key Takeaways

1. SVM finds the **maximum margin** hyperplane to separate classes.
2. **Support vectors** are the closest points to the boundary and are the only points that matter for the model.
3. The separator is a **point** in 1D, a **line** in 2D, a **plane** in 3D, and a **hyperplane** in higher dimensions.
4. **Hard margin** (maximum margin classifier) requires perfect linear separability.
5. **Soft margin** (support vector classifier) allows some misclassifications, controlled by **C**.
6. For non-linearly separable data, use a **kernel** (polynomial, RBF, sigmoid) to implicitly project data to higher dimensions.
7. The **kernel trick** avoids the actual projection by computing dot products in the higher dimensional space directly.
8. **RBF** is the most widely used kernel but is also the most expensive.
9. **C** controls margin width vs. misclassification tolerance. **Gamma** controls how tightly the boundary wraps around data points.
10. SVM works best for **small, complex datasets**. For simple data, simpler methods (naive Bayes, logistic regression) may perform equally well or better.

---

## Advantages & Disadvantages

### Advantages

- High accuracy, faster prediction.
- Memory efficient (only support vectors are stored after training).
- Works well if the dataset is small and separable.
- Effective in high-dimensional spaces.
- Effective when the number of dimensions is greater than the number of instances.
- Variety of kernel functions available for different data distributions.

### Disadvantages

- Not suitable for larger datasets (computationally expensive).
- Poor performance on overlapping classes.
- Highly sensitive to the choice of kernel.

---

## References

- https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
- https://www.analyticsvidhya.com/blog/2021/10/support-vector-machinesvm-a-complete-guide-for-beginners/
- https://towardsdatascience.com/hyperparameter-tuning-for-support-vector-machines-c-and-gamma-parameters-6a5097416167/
- https://www.geeksforgeeks.org/machine-learning/gamma-parameter-in-svm/
