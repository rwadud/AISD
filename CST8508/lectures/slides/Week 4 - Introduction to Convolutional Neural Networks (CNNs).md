# Convolutional Neural Networks (CNNs) for Machine Vision

Instructor: Stephin Rachel Thomas | Feb 05, 2026

*Transforming visual recognition through deep learning.*

---

## Today's Topics

- Artificial Neural Networks
- Disadvantages of simple ANN for Image classification
- Introduction to CNN
- CNN architecture
- Deep dive into CNN layers
- Application of CNN
- Performance Evaluation Metrics

---

## What are Artificial Neural Networks?

**1. Biological Inspiration**
ANNs are inspired by the structure and function of the human brain, composed of interconnected nodes called neurons.

**2. Learning Through Data**
These networks learn by analyzing large datasets, adjusting the connections between neurons to improve their performance.

**3. Pattern Recognition**
ANNs are particularly effective at recognizing complex patterns in data, making them ideal for image classification.

---

## Classification using Traditional Methods

Traditional methods use hand-crafted features. Example: cat vs. not-cat classification using a decision tree.

| Ear shape | Face shape | Whiskers | Cat/No cat |
|-----------|------------|----------|------------|
| Pointy    | Round      | Present  | 1          |
| Floppy    | Nor Round  | Present  | 1          |
| Floppy    | Round      | Absent   | 0          |
| Pointy    | Nor Round  | Present  | 0          |
| Pointy    | Round      | Present  | 1          |
| Pointy    | Round      | Absent   | 1          |
| Floppy    | Nor Round  | Absent   | 0          |
| Pointy    | Round      | Absent   | 1          |
| Floppy    | Round      | Absent   | 0          |
| Floppy    | Round      | Absent   | 0          |

**Decision-tree method:**
```
          Ear shape
         /         \
      Pointy       Floppy
        |              |
   Face shape      Whiskers
   /       \       /      \
Round  Not round Present  Absent
  |        |       |        |
 Cat    Not cat   Cat    Not cat
```

---

## ANN for Image Classification

ANNs can classify images (e.g., cats vs. dogs) by feeding pixel data through:
- **Input Layer** → **Hidden Layers** → **Output Layer** → Output class (Cat / Dog)

The network receives raw image data and learns to distinguish between classes through training.

---

## Limitation of ANN for Image Classification

A 1000×1000px image (3 channels) has **3×10⁶** input values. With 10³ hidden neurons:

**No. of weights = 3×10⁶ × 10³ = 3×10⁹**

Problems:
- High computational cost
- Over-fitting problem
- Longer training time

---

## Convolutional Neural Network (CNN)

**1. Definition**
A deep learning model designed for processing images to identify patterns and make decisions.

**2. Objective**
Solve complex visual tasks with deep learning.

**3. Benefits**
- Handles high-dimensional, structured data like images, videos and audio.
- Hierarchical feature learning.
- Robust to translation of object.

---

## CNN Architecture

CNNs typically consist of an **input layer**, **multiple hidden layers**, and an **output layer**.

The hidden layers include a series of **convolutional layers**, **pooling layers** and **fully connected layers**.

Each layer performs distinct operations:
- **Convolutional** layers apply a convolution operation
- **Pooling** layers perform down-sampling
- **Fully connected** layers compute the class scores

```
Input → [Convolution → Pooling] × N → Flatten → Fully Connected → Output
         ←── Extracting Features ───→            ←─ Classification ─→
```

---

## Key Components of CNN

**1. Convolutional Layers**
Extract spatial features from input images.

**2. Pooling Layers**
Reduce spatial dimensions, simplify computation.

**3. Fully Connected Layers**
Integrate features for final classification.

---

## Deep Dive into Convolutional Layers

- In these layers, small, learnable filters slide over the input data (like images) to extract features such as edges, textures, and shapes.
- Each filter in a convolutional layer detects different features, and multiple layers work together to capture increasingly complex aspects of the data.
- The convolutional layers thus play a crucial role in feature detection and representation, enabling CNNs to effectively perform tasks like image recognition and classification.

A **convolution kernel** slides across the image and produces a **convolution output** (feature map).

---

## CNN Fundamentals

The basic principle of a CNN is to automatically learn and extract **hierarchical features** from input data, typically images, through the use of convolutional layers.

```
Image → Low-Level Feature → Mid-Level Feature → High-Level Feature → Trainable Classifier
```

- **Low-level**: edges, colors, gradients
- **Mid-level**: textures, parts
- **High-level**: object-level representations

---

## Convolutional Layers

Convolutional layers produce **feature maps** by applying learned filters to the input.

- Convolutional layers help the network focus on only the most important features
- Not all the pixel information in the image is relevant for training the model
- Improves performance and accuracy

---

## Convolution Operation

A filter (kernel) slides over the input image computing element-wise products and summing them.

**Example: 6×6 input \* 3×3 filter → 4×4 output**

Input image (6×6):
```
0  0  0  1  1  1
0  0  0  1  1  1
0  0  0  1  1  1
0  0  0  1  1  1
0  0  0  1  1  1
0  0  0  1  1  1
```

Filter (3×3) — Sobel vertical edge detector:
```
 1   0  -1
 2   0  -2
 1   0  -1
```

**Step 1** — top-left position:
```
0*1 + 0*0 + 0*(-1) +
0*2 + 0*0 + 0*(-2) +
0*1 + 0*0 + 0*(-1) = 0
```

**Step 2** — shifted one right:
```
0*1 + 0*0 + 1*(-1) +
0*2 + 0*0 + 1*(-2) +
0*1 + 0*0 + 1*(-1) = -4
```

**Full output (4×4):**
```
 0  -4  -4   0
 0  -4  -4   0
 0  -4  -4   0
 0  -4  -4   0
```

The vertical edge (column of 1s) is detected as negative values in the output.

---

## Convolutional Layers — Parameters

**Filter Size**
The filter size determines the extent of the input data that each filter covers, affecting the granularity of the features detected; smaller filters capture fine details, while larger filters identify broader patterns.

**Stride**
Stride, the step size with which filters move across the input, influences the overlap of receptive fields and the size of the output feature map; larger strides result in smaller, more abstract feature maps.

**Padding**
Padding, the addition of zeroes around the input border, allows control over the spatial dimensions of the output, preserving edge information and enabling deeper layers to build a spatial hierarchy of increasingly complex and abstract features.

---

## Convolutional Layer — Output Image Size

The image output size is given by:

$$\frac{N - F + 2P}{S} + 1$$

Where:
- **F** = size of filter
- **S** = stride
- **N** = size of image
- **P** = amount of padding

---

## Pooling Layers

- Responsible for reducing the spatial size of the feature maps generated by convolutional layers
- By performing operations such as **max** or **average pooling**, they down-sample the input features, which helps to decrease the computational load and the number of parameters in the network
- This reduction also contributes to making the network more tolerant to variations and distortions in the input data, enhancing its ability to generalize

---

## Pooling Layers — Visualization

The pooling layer reduces the spatial dimensionality of the input feature map.

Two common types using a 2×2 window:
- **2×2 Max Pooling** — takes the maximum value in each window
- **2×2 Avg Pooling** — takes the average value in each window

---

## Pooling Operation

**Example: 4×4 feature map → 2×2 pooled feature map**

Input (4×4):
```
6  2  7  5
5  5  3  1
8  1  4  2
6  3  2  5
```

**Max-pooling** (2×2 windows, stride 2):
```
6  7
8  5
```

**Average-pooling** (2×2 windows, stride 2):
```
4.5   4
4.5  3.25
```
