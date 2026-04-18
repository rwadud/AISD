# Lecture 6: PyTorch, Autograd, and Deep Learning Pipelines

**Instructor:** Stephin Rachel Thomas
**Date:** June 17, 2025

## Topics
1. What is PyTorch
2. History of PyTorch (Torch to PyTorch 2.0)
3. Key Features of PyTorch
4. PyTorch Capabilities (Array Math, Auto Diff, Cloud, Hardware)
5. Core Components of PyTorch
6. Training Loop Recap (Weights, Backpropagation, Gradients)
7. Learning Rate Schedulers
8. Deep Dive into Tensors (Ranks, Properties, CUDA)
9. PyTorch and Autograd
10. Computational Graphs
11. PyTorch vs TensorFlow (Full Comparison)
12. Tracking Gradients with `requires_grad`
13. Deep Learning Training Process (Prep, Dev, Deploy)
14. Basic Building Blocks of a PyTorch Pipeline
15. Custom Dataset Class and `ImageFolder`
16. DataLoader, Transforms, and Preprocessing
17. Neural Network Module (`nn.Module`)
18. Custom Layers
19. Optimizers and Loss Functions
20. Model Definition (Lego Block Analogy)
21. Training Loop and Zeroing Gradients
22. Convergence and Overfitting Monitoring
23. Evaluation and `torch.no_grad()`
24. Validation Strategies (K-fold, Leave-one-out)
25. Computer Vision with PyTorch (`torchvision`, Transfer Learning)
26. Advanced Features (CUDA, JIT, TorchScript, C++)
27. Multi-GPU Training (Data and Model Sharding)
28. Debugging Deep Learning Code
29. Best Practices in PyTorch
30. Community and Ecosystem
31. Industry Relevance and Tomorrow's Class

---

## What is PyTorch?

**PyTorch** is an **open-source machine learning library** for Python, known for its:

- **Flexibility**
- **Ease of use**
- **Dynamic computation graph**

It allows researchers to experiment quickly with deep neural networks, and it is extensively used in academia and industry for applications ranging from **computer vision** to **natural language processing**.

> **PyTorch is one of the most popular deep learning frameworks that allows us to implement neural networks more efficiently.**

---

## History of PyTorch

### Early Beginnings: Torch (2002)

- PyTorch evolved from **Torch**, a machine learning framework built on **Lua** in **2002**.
- Torch gained popularity for its **GPU acceleration** and was widely used in academia.
- However, Lua was not as popular as Python, limiting Torch's adoption.

### Birth of PyTorch (2016)

- Facebook's **AI Research Lab (FAIR)** developed PyTorch to provide a **Pythonic alternative** to Torch.
- Released in **October 2016**, PyTorch introduced:
  - **Dynamic computation graphs**
  - **Easy debugging**
- These features made it popular among researchers.

### Growth and Adoption (2017 to 2020)

- Quickly became the preferred framework for **AI research**, **deep learning**, and **NLP**.
- **Hugging Face** adopted PyTorch for **transformers** and **NLP models**.
- Facebook introduced **TorchScript** for model deployment.

### Competition with TensorFlow (2021 to Present)

- By **2021**, PyTorch had surpassed TensorFlow in research usage.
- **PyTorch 2.0 (2023)** introduced **faster performance with `torch.compile`**.
- In **September 2022**, PyTorch transitioned to the **Linux Foundation**, ensuring **open governance**.
- Today, it is widely used in academia, industry, and production AI models.

---

## Key Features of PyTorch

PyTorch's key features include:

- **Dynamic computation graph**, which allows changes to the network on the fly.
- **Strong GPU acceleration** for faster computations.
- **Deep integration with Python**.

This integration makes PyTorch not only powerful but also flexible and intuitive, offering **seamless compatibility with popular Python libraries** like **NumPy** and **SciPy**.

---

## PyTorch Capabilities at a Glance

PyTorch:

- Facilitates building **deep learning projects**.
- Easily runs **array-based calculations**.
- Builds **dynamic neural networks**.
- Performs **automatic differentiation** with strong **GPU acceleration**.
- Was developed to process **large-scale image analysis** tasks:
  - Object detection
  - Segmentation
  - Classification
- Is supported by all major **cloud platforms**:
  - Amazon Web Services (AWS)
  - Google Cloud Platform (GCP)
  - Microsoft Azure
- Supports **CPU**, **GPU**, **TPU**, and **parallel processing**.

---

## Core Components of PyTorch

PyTorch comprises several core components that work together to simplify creating and training complex models:

| Component | Role |
|-----------|------|
| **Tensors** | Similar to NumPy arrays but with GPU support. |
| **Autograd** | Automatic differentiation engine. |
| **Optimizers** | Abstract the optimization algorithms used to train neural networks. |
| **`nn.Module`** | Base class for layers, losses, and full models. |

---

## Training Loop Recap

Before diving further into PyTorch mechanics, we review what happens when you train a neural network.

### Weights as the Learning Element

If your data is represented by **X**, your **weights** are what is learning. You take your input and transform it through the weights to give you some output.

When you save a model to disk, you are saving the weights. A model file has two parts:

1. **Architecture**: the structure of the network.
2. **Weights**: a big matrix of learned values that slot into the right parts of your model.

### The Training Loop in Words

Given a supervised learning task (data plus labels):

1. **Random initialization**: start with a random initialization of weights.
2. **Forward pass**: feed input through the model. The input goes through the weights plus bias, then through the **activation function**. Say it is cats and dogs, at the output your model should produce some predictions.
3. **Loss calculation**: compare predictions to ground truth labels using a **loss function** (e.g., cross-entropy loss, which is your basic loss for binary classification).
4. **Backpropagation**: calculate gradients of the loss with respect to the weights, and propagate them backwards through the network.
5. **Weight update**: use the gradients to update the weights.
6. **Repeat**: send in a new batch of data and keep doing this until you converge.

### Why Calculate Gradients

The goal of optimization is to **increase accuracy** or **decrease loss**. You start at some initial value of your weights, and the goal is to reach the **global minimum** that minimizes your loss, which is the configuration of weights that minimizes the loss.

So how do you get there? You go in the direction of **maximum change**, because the direction of maximum change of your weights minimizes your loss.

> **Key insight**: The **gradient** (derivative) gives you the direction of maximum change. That is why you take gradients as part of backpropagation.

### Loss Landscape Diagram *(reconstructed)*

```
Loss
  ^
  |    *  <-- start (random init)
  |   / \
  |  /   \        *
  | /     \     /
  |/       \   /
  |         \ /
  |          *  <-- global minimum (goal)
  +------------------> Weights
```

You are searching for the weight configuration that lands at the lowest point on the loss surface.

---

## Learning Rate Scheduler

The **learning rate scheduler** controls how big a step you take in the direction of the gradient.

- If the step is too big, you will **overshoot** and move to the other side of the minimum.
- So you want to keep **decreasing** that step size as training progresses.

> **Course note**: This was asked in lab. The learning rate scheduler is part of your hyperparameter setup, and it decays the learning rate over time.

---

## Deep Dive into Tensors

Tensors are the **fundamental building blocks** in PyTorch, representing data like **images** or **text**.

> To handle and store the data in all stages of deep learning, PyTorch uses this essential data structure called a **tensor**.

**Inputs**, **intermediate representations**, and **outputs** are all stored as tensors.

### Mathematical View

In mathematics, tensors can be defined as **generalizations of scalars, vectors, and matrices** to any dimension.

- In PyTorch, tensors are **multidimensional arrays containing elements of a single data type**.
- A tensor is similar to the fundamental object in **NumPy** called `ndarray`.
- `ndarray` is an **n-dimensional homogeneous array** of fixed-size items.

### Tensor Ranks

| Rank | Name | Example |
|------|------|---------|
| **0-d tensor** | Scalar | `1` |
| **1-d tensor** | Vector | `[1, 5, 7]` |
| **2-d tensor** | Matrix | `[[1, 5, 7], [2, 9, 3], [4, 8, 6]]` |
| **3-d tensor** | Cube | (multidimensional array) |

### Examples of Tensor Shapes

- **rank 0** tensor, dimensions `[]` (scalar)
- **rank 1** tensor, dimensions `[5]` (vector)
- **rank 2** tensor, dimensions `[5, 3]` (matrix)
- **rank 3** tensor, dimensions `[4, 4, 2]`

### Properties of Tensors

- **Tensor operations are performed significantly faster using GPUs.**
- Tensors can be **stored and manipulated at scale** using distributed processing on multiple CPUs and GPUs, and across multiple servers.
- **Tensors keep track of the graph of computations that created them.** This is what makes autograd possible.

### Tensor Code Example

This code shows how tensors are **created**, **manipulated**, and **moved to a GPU** for accelerated computing.

```python
import torch

# Create a tensor
tensor_a = torch.tensor([2, 3, 4], dtype=torch.float32)

# Manipulating tensor (e.g., scaling and addition)
tensor_b = tensor_a * 2 + 1  # Element-wise scaling and addition

# Using tensor in a simple computation (e.g., element-wise multiplication)
result = tensor_a * tensor_b

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Move tensors to the GPU
    tensor_a = tensor_a.cuda()
    tensor_b = tensor_b.cuda()
    result = result.cuda()
    print("Tensors moved to GPU:", tensor_a, tensor_b, result)
else:
    print("CUDA is not available. Tensors are on CPU:", tensor_a, tensor_b, result)
```

**Output:**

```
Tensors moved to GPU: tensor([2., 3., 4.], device='cuda:0')
                      tensor([5., 7., 9.], device='cuda:0')
                      tensor([10., 21., 36.], device='cuda:0')
```

---

## PyTorch and Autograd

### Formal Definition

There are **two steps** in training neural networks:

1. **Forward propagation**
2. **Backward propagation**

After the loss function is calculated, the **derivative of the loss function** in terms of the parameters is calculated. You iteratively update the weight parameters so that the loss function returns the smallest possible loss.

- This is called **iterative optimization**, because we use an optimizer to perform the update of parameters.
- It is also called **gradient-based optimization**.

> **Autograd is a set of techniques that allows us to compute gradients for arbitrary complex loss functions efficiently.**

### Why Autograd Matters

The whole point of PyTorch is **autograd**, which stands for **automatic differentiation**. Once you understand autograd, then everything else follows, because with autograd you can basically write your own PyTorch from scratch.

> **Autograd** is a way of calculating gradients and keeping track of them. Once you understand autograd, you basically understand most of deep learning and any deep learning library.

### Starting with a Simple Function

Let us start with a simple function to see what autograd actually does.

$$f(x) = 3 (x + 2)^2$$

The first derivative of this function, using the chain rule:

$$\frac{df}{dx} = 3 \cdot 2 (x + 2) \cdot 1 = 6 (x + 2)$$

Whenever you have $x^2$, the derivative of $x^2$ is $2x$. So the first derivative of this is $2$ times this expression, scaled by the outer constant.

Evaluated at $x = 1$:

$$\frac{df}{dx}\bigg|_{x=1} = 6 (1 + 2) = 18$$

---

## Computational Graphs

### What Autograd Builds Behind the Scenes

When you give PyTorch a function, it creates a **computational graph** that keeps track of all of your operations.

> **The main thing in PyTorch is to create this computational graph that keeps track of all of your operations.**

Take our function $f(x) = 3 (x+2)(x+2)$. The graph opens up all of the operations into addition and multiplication nodes.

### Diagram: Computational Graph for $f(x) = 3(x+2)^2$ *(reconstructed)*

```
                   X
                  / \
                 /   \
              (+2)  (+2)
                |    |
                A    B        A = X + 2,  B = X + 2
                 \  /
                  \/
                 (x)          C = A * B
                  |
                  C
                  |
                 (x 3)
                  |
                  f            f = 3 * C
```

- $A = X + 2$ (addition operation)
- $B = X + 2$ (addition operation)
- $C = A \times B$ (multiplication operation)
- $f = 3 \times C$ (output)

### Calculating Gradients Through the Graph

When you call `backward`, PyTorch calculates the gradients through the **chain rule**.

$$\frac{df}{dX} = \frac{df}{dC} \cdot \left( \frac{dC}{dA} \cdot \frac{dA}{dX} + \frac{dC}{dB} \cdot \frac{dB}{dX} \right)$$

Step by step:

| Derivative | Value |
|------------|-------|
| $df/dC$ | $3$ |
| $dC/dA$ | $B$ (since $C = A \times B$) |
| $dC/dB$ | $A$ |
| $dA/dX$ | $1$ (since $A = X + 2$) |
| $dB/dX$ | $1$ (since $B = X + 2$) |

At $X = 1$: $A = 3$, $B = 3$.

$$\frac{df}{dX} = 3 \cdot (3 \cdot 1 + 3 \cdot 1) = 3 \cdot 6 = 18$$

Which matches our earlier analytical answer.

### The Question Gradients Actually Answer

The real question you are asking through backpropagation is:

> **If I made a small change in my learnable parameter, how would that change my loss?**

For each learnable parameter, you are asking:

- How much does this parameter contribute to my loss?
- In which direction should I change it so my loss is minimized?

You ask the same question at every node (A, B, C, X), not just at the leaves.

### Gradient Accumulation at Each Node

Each node in the computational graph saves the gradients flowing through it. You can think of each node as having storage where it stores the gradient at each iteration.

- **Forward pass**: take input, pass it through the graph, get the output.
- **Backward pass**: call `backward()`. This calculates all gradients through the chain rule and stores them.

This stored information is useful for:

- Debugging neural networks (checking gradient values across iterations).
- Interpretability.
- Training techniques that leverage these gradients (e.g., **gradient clipping**).

### Gradient Clipping *(course context)*

One of you had **gradient clipping** in your hyperparameters. The problem when training really deep neural nets is that gradients can:

- **Explode**: become too large.
- **Vanish**: shrink toward zero.

When they explode, you **clip** them. Gradient clipping is another way of regularizing the networks.

---

## PyTorch vs TensorFlow

### History

- **PyTorch** was built by **Facebook**.
- **TensorFlow** was the main competitor.

Before PyTorch, the way to do deep learning back in the **2014 to 2015 era** was to manually derive all of the backpropagation equations, calculate the gradients for each layer, and implement all of that yourself.

### What PyTorch Solved

You can just set up your model architecture and PyTorch will **automatically calculate the gradients** for you. That is the biggest advantage you get from PyTorch.

### Static vs Dynamic Graphs

With **TensorFlow** (originally), you created this graph and it was **static**. You compiled it into a graph object, and you could not change any nodes without recompiling the entire graph. With **PyTorch**, the graph is **dynamic**, so if you want to change a node, you can change it. You do not have to recompile everything. You can just recompute the gradients.

### Full Comparison Table

| Feature/Aspect | PyTorch | TensorFlow |
|----------------|---------|------------|
| **Primary Language** | Python | Python, with APIs in other languages |
| **Computation Graphs** | Dynamic (Define-by-Run) | Static (Define-and-Run) |
| **Ease of Use** | More user-friendly and intuitive | Steeper learning curve, improved with Keras |
| **Debugging** | Easier due to dynamic graphs and Pythonic nature | More complex, requires separate tools |
| **Performance** | Comparable, with slight variations based on use case | Comparable, with optimizations for large-scale |
| **Community & Support** | Strong community, especially in research | Strong community, less popular in research |
| **Deployment** | Growing in mobile and web deployment | Extensive deployment options, including TFLite |
| **Pre-Trained Models** | Available through TorchVision, etc. | Extensive range in TensorFlow Hub |
| **Distributed Training** | Supported with PyTorch Distributed | Advanced options with TensorFlow Distributed |
| **Integration** | Seamless with Python libraries | Integrates well with TensorFlow ecosystem |

### When to Pick Which

> **Use PyTorch if:** you are a beginner, doing research, or need flexibility (e.g., NLP, CV).
>
> **Use TensorFlow if:** you need enterprise-level solutions, mobile deployment, or production-ready models.

---

## Tracking Gradients with `requires_grad`

You can choose whether a tensor tracks gradients. If `requires_grad` is `True`, the tensor participates in autograd. If `False`, it does not.

### Basic Usage *(reconstructed code)*

```python
import torch

x = torch.tensor(1.0, requires_grad=True)
f = 3 * (x + 2) * (x + 2)
f.backward()
print(x.grad)   # tensor(18.)
```

If you forget `requires_grad=True` on your tensor, calling `backward()` will error because the tensor does not have `requires_grad` set.

### When to Turn Gradient Tracking Off

- **Validation, evaluation, and inference**: you do not really need gradients. Keeping `requires_grad=True` for everything can blow up your memory, because you can think of it as setting up arrays at every node that save gradients.
- **Frozen parameters**: if you have learnable parameters you do not want updated (e.g., during transfer learning), set `requires_grad=False` on them.

### Accessing the Gradient

You can access the gradient directly after calling `backward()`:

```python
print(x.grad)   # prints the accumulated gradient on x
```

This is one way of debugging your neural nets, seeing the gradients across different iterations.

---

## Deep Learning Training Process

At the highest level, the deep learning workflow in PyTorch has three stages.

### 1. Data Preparation

Convert generic data (**text**, **image**, **video**, **audio**, and so on) to **numerical values** in the form of **tensors**.

Tensors are:

- **Pre-processed** during transforms.
- Grouped into **batches** before being passed into the model.

### 2. Model Development

Involves:

- **Model design**
- **Training**
- **Testing performance**

The dataset is divided into **training data**, **validation data**, and **testing data**.

### 3. Model Deployment

- **Save the model to a file.**
- **Deploy** the model to a product or service, usually on a **cloud server** or to an **edge device**.

---

## Basic Building Blocks of a PyTorch Pipeline

If you are writing in PyTorch, what are the main things you need in your pipeline?

1. **Data loading** (function that prepares and loads data)
2. **Model definition**
3. **Training loop**
4. **Evaluation**
5. **Main function** to tie it all together

Let us go through each one.

---

## Dataset and DataLoader

In PyTorch, you have a notion of a **Dataset** and a **DataLoader**.

PyTorch's `Dataset` and `DataLoader` classes **streamline data preprocessing and loading**:

- **`Dataset`** allows for custom data handling.
- **`DataLoader`** efficiently batches and loads data, offering options like **shuffling** and **multiprocessing**.

For instance, in image classification, `DataLoader` can automate the process of loading and transforming images into tensor format, ready for model input.

### Preloaded Datasets

There are multiple existing datasets preloaded into PyTorch:

- **MNIST**
- **CIFAR**
- and many others.

### Your Own Data

But you have your own cat and dog images, or your flower dataset on disk. You set it up in a folder structure similar to what you used for your MLP:

```
flowers/
├── daffodil/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── tulip/
│   ├── image1.jpg
│   └── ...
└── ...
```

### Custom Dataset: Three Methods You Must Define

To create a custom dataset, inherit from `torch.utils.data.Dataset` and define three things:

1. **`__init__`**: defines the size of the dataset, path to the data, and how to assign labels. You say your data is at this path to the flowers directory and take the folder name as the label. Could also be reading from a **CSV** where the paths are, then assigning labels for each data point.
2. **`__len__`**: returns the size of the dataset. When you call `len` on the dataset, it gives the size of all training examples.
3. **`__getitem__`**: the DataLoader is a generator. When you call `next`, `__getitem__` defines how to retrieve and display the training example. If your dataset is an image, maybe you want to show the image, so put code here to show it. Maybe your dataset is an image with facial landmarks and you want to show those too. All of that code goes here.

### Minimal Custom Dataset *(from slides)*

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, size, num_classes):
        self.size = size
        self.num_classes = num_classes
        self.data = torch.randn(size, 10)  # Random features (size x 10)
        self.labels = torch.randint(0, num_classes, (size,))  # Random labels

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create an instance of the CustomDataset
dataset = CustomDataset(size=1000, num_classes=5)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Image-Folder Custom Dataset *(reconstructed code)*

```python
import os
from torch.utils.data import Dataset
from PIL import Image

class TwoClassDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {"class_a": 0, "class_b": 1}
        self.samples = []
        for cls_name, label in self.classes.items():
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
```

In `__init__`, you:

- Provide the **root directory**.
- Set up any **transformations** specific to that dataset (e.g., cropping, whatever chained pipeline of data transformations you want).
- **Assign labels** to the different classes.

You are just reading the data and assigning labels. If you read from this directory, all of these images are this class name, all of those are another.

### `ImageFolder`: A Shortcut

In the assignment, you use `ImageFolder`, which at the back end is implementing exactly what we just did.

**`ImageFolder`** creates the label mapping based on the folder structure of the data. It automatically takes the folder name as the label.

> **Course note**: For all of you working on lab 5, I highly suggest trying to write your own custom dataset for the same thing, just so you know where to give the paths, how to give the labels, and how to read in the data. You already use `ImageFolder`, and this is a good exercise to understand what it does.

### DataLoader

Once you have defined your dataset, the way you use it is by creating an instance and passing it to `DataLoader`.

> **DataLoader** is a generator that wraps your dataset class. You can call `next` on it to get the next batch.

### Training, Validation, Test DataLoaders *(reconstructed code)*

```python
from torch.utils.data import DataLoader
from torchvision import transforms

universal_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet mean
                         std=[0.229, 0.224, 0.225]),   # ImageNet std
])

train_dataset = TwoClassDataset("data/train", transform=universal_transform)
val_dataset   = TwoClassDataset("data/val",   transform=universal_transform)
test_dataset  = TwoClassDataset("data/test",  transform=universal_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)
```

You have a training DataLoader, a validation DataLoader, and a test DataLoader. The dataset that goes in would be the training dataset for the training loader, the validation dataset for the validation loader, and so on.

---

## Transforms and Preprocessing

### Why Resize

The main reason to resize is that in your dataset, each image is a different size. If each image is a different size, it will give you an error because the DataLoader is trying to **batch** them, and to stack them together they all have to be the same size.

### Other Preprocessing

- Cropping and resizing for images.
- For **text**, transformations would include **tokenization**, **padding**, and so forth.

> **Under your transforms, all of the preprocessing for the data goes in.**

### ImageNet Normalization Values

Common ImageNet mean and standard deviation values:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

These are from **ImageNet**. It is kind of standard practice to use them in computer vision, especially for classification tasks like cats and dogs.

- **Question:** Do we have to always enter those manually?
- **Answer:** Generally, if you are using any other dataset, you would calculate the mean and standard deviation from that dataset, from just your **training split**.
- In computer vision, people often just use the ImageNet values because you also want to generalize well, and ImageNet was a huge dataset back in 2017. These days it is considered a smaller dataset, but this practice has carried on.

> **Critical rule**: Never use your **whole dataset** for computing mean and standard deviation, because then you are cheating. That is **data leakage**.

---

## Neural Network Module (`nn.Module`)

PyTorch's **`nn` module** is a comprehensive library that includes:

- A wide range of **pre-defined layers**.
- **Loss functions**.
- Utilities essential for building neural networks.

It provides an easy way to construct network architectures, enabling:

- The simple assembly of standard layers like **convolutional** and **linear** layers.
- The **customization** of more complex models.

This module greatly simplifies the process of defining a network's forward pass, with its **intuitive and Pythonic approach**, allowing for clear and readable code that closely resembles the actual architecture of the model.

---

## Custom Layers

PyTorch gives you a whole bunch of different layers: **convolutional**, **linear**, and so on. But say you want to create your own small module to reuse each time.

### Two Methods You Must Define

For any custom layer, define:

1. **`__init__`**: initialize parameters and define layer sizes.
2. **`forward`**: define the function the layer computes.

You do **not** have to define `backward`. Autograd handles it automatically. Unless the differentiation is not defined for some operation, then you might have to tell it how to calculate the backward pass. There are some cases where there are functions that are not fully differentiable, and then you have to do approximations for those differentiations.

### Custom Linear Layer Example

The linear layer implements $f(x) = W x + b$. To build it yourself:

```python
import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias   = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return x @ self.weight.T + self.bias
```

- `in_features` and `out_features` are parameters you take in your initialization, defining the size of that layer.
- You define the **weight** and the **bias** and initialize those to random values.
- Those are now your two **learnable parameters** for your custom linear class.

### Why Custom Layers *(reinforcement)*

PyTorch gives you a fully connected layer, a convolutional layer, and so on. But maybe you want to create a custom layer that has one type of operation you keep wanting to reuse and you do not want to keep rewriting. So you create your own custom layer.

Say it is a very arbitrary function. You create that where in `__init__` you define learnable parameters, initialize them, and then in `forward` you implement that function. It is just telling PyTorch how to take your input and transform it. Here, we implemented $Wx + b$, but this can be anything.

> **Big picture**: PyTorch is autograd, and everything around it is bells and whistles. You can write your own custom layer, your own custom DataLoader, your own custom loss, your own custom optimizer.

---

## Optimizers and Loss Functions

### Optimizers

PyTorch offers various optimizers, each providing different approaches to navigating the loss landscape:

- **SGD** (Stochastic Gradient Descent)
- **Adam**
- **RMSprop**

### Key Loss Functions

- **CrossEntropyLoss**: used for classification tasks.
- **Mean Squared Error (MSE)**: commonly used in regression.

These functions measure the **difference between the predicted output and actual data**, guiding the model's improvements during training.

### Cross-Entropy Code Example *(from slides)*

```python
import torch
import torch.nn as nn

# Example tensors representing predicted outputs and actual labels
# Predicted outputs (logits) from a neural network,
# for a batch of 3 samples and 4 classes (unnormalized scores)
predicted_logits = torch.tensor([[1.5, 0.5, -1.2, 0.7],
                                 [1.2, 0.2, 0.5, -1.0],
                                 [0.3, -0.9, 1.0, 1.1]])

# Actual labels (indices of the correct class), for the same batch of 3 samples
actual_labels = torch.tensor([0, 1, 3])  # Assuming class indices are 0, 1, 2, 3

# Define the cross entropy loss function
loss_fn = nn.CrossEntropyLoss()

# Calculate the loss
loss = loss_fn(predicted_logits, actual_labels)

print("Cross Entropy Loss:", loss.item())
```

**Output:**

```
Cross Entropy Loss: 1.075467824935913
```

### Custom Loss Functions

Losses are inherited from `nn.Module`, which is the main thing you use for losses, layers, and everything in PyTorch.

Again, for the loss, it is just `__init__` and `forward`.

### Mean Squared Error *(reconstructed example)*

The mean squared error loss:

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (x_{\text{pred}, i} - x_{\text{actual}, i})^2$$

```python
import torch.nn as nn

class CustomMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        return ((prediction - target) ** 2).mean()
```

Just the mean of the prediction minus target, squared.

### Summary: Three Customization Patterns

| Customization | Methods to Define |
|---------------|-------------------|
| **Dataset** | `__init__`, `__len__`, `__getitem__` |
| **Layer** | `__init__`, `forward` |
| **Loss** | `__init__`, `forward` |

---

## Quick Check: Autograd

> **Question**: What parameter would you change if you did not want to track the gradients on a particular tensor?
>
> **Answer**: `requires_grad`.

---

## Data Preparation in Your Assignment

In your assignment, you are asked to create a `load_dataset` method that takes a dataset. The custom dataset approach we discussed, or you can use `ImageFolder` as a shortcut, which implements getting the data paths and loading it. The difference: `ImageFolder` requires your data to be in the format where each class label has a folder with all of the images under it. It automatically takes the folder name as the label.

### Full Data Pipeline Steps

1. Apply **transforms** (resize, normalize, etc.).
2. Read your **dataset** (custom class or `ImageFolder`).
3. **Split** into training, validation, and test.
4. Wrap each split in a **DataLoader**.

---

## Model Definition: The Lego Block Analogy

> **PyTorch model definition is kind of like Lego.**

The basic layer functions have already been defined for you:

- **Convolution**
- **ReLU**
- **Pooling**

If any of these were not defined, you would create your own custom layer for that function. But in PyTorch, all of these have already been defined. So you look at your desired architecture, and you implement it like Lego blocks.

### Two-Phase Definition

Building a neural network in PyTorch involves defining a model class that inherits from `nn.Module`. The class typically includes:

1. **`__init__`**: define all of the layers that you need as part of your model, like grabbing your Lego blocks and setting them up. You define conv layer one, conv layer two, the flatten layer, fully connected layer one, fully connected output layer, in terms of dimensions, kernel sizes, and so on.
2. **`forward`**: put together the function with the different layers. You should always think of a neural network as a function, and in `forward` you are putting together that function.

For example, a simple network for image classification might include **convolutional layers**, activation functions like **ReLU**, and a final **fully connected layer**.

### SimpleCNN Example *(from slides)*

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layer (input channels, output channels, kernel size)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Assuming input images are 32x32 pixels
        self.fc2 = nn.Linear(512, 10)           # Output layer for 10 classes

    def forward(self, x):
        # Apply convolutions and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 128 * 4 * 4)

        # Apply fully connected layers with ReLU and output layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### What Does `-1` Mean in Flatten?

Say your convolutional layer is using 32 filters. Your output is going to have 32 channels. So your output from the conv block is:

$$\text{batch} \times \text{filters} \times \text{height} \times \text{width}$$

As a concrete example, your batch size might be 64 and the conv output might be `32 by 32 by 1`, giving a shape of `64 × 32 × 32 × 1`. When you flatten it, you still want `64 by the product of the rest`, because the input is going in as batches.

> **`-1` tells PyTorch to infer the batch size.**

Your first dimension is almost always your batch size. You want to flatten everything except that first dimension, but you do not want to hardcode the batch size. `-1` means: infer this dimension by multiplying all the others together.

### Why Define `forward`?

- **Question**: These layers are already built in, so why are we defining `forward`?
- **Answer**: Because `forward` defines **your arbitrary model**. Maybe you want three convolutional layers, two feed forward layers, one max pooling, one mean pooling. These Lego blocks are predefined, but your custom architecture is not, because **that design space is infinite**.

> **Analogy**: Red block, blue block, green block are predefined. But how you combine them, two red blocks, one blue block, one green block, one yellow block, that is **your** forward function, custom to you.

### Layer Dimensions as Design Choices

Numbers like 64, 56, 128 are the layer dimensions and kernel sizes, not fixed values. If you want a layer that is 4 by 6 in one place, and 3 by 3 in another, you have to define each of those **separately as two different layer instances**. It is still the same type of layer (a linear layer), just with different sizes.

And these could also be your own custom layers. Transformer models that have modules within them create those modules as separate blocks so it is easy for you to build.

---

## Training Loop

Once the DataLoader and model are defined, the training loop ties them together.

### The Process

1. Iterate through the **DataLoader**, which gives you a batch each time.
2. Each batch is passed through the model in the **forward pass**.
3. Compute the **loss**.
4. Call **backward pass** to compute and accumulate gradients.
5. Pass them back to your learnable parameters to update them.

That is basically what training a neural network involves.

### The Two Most Important Things

The two most important things to define:

- **Loss function**
- **Optimizer**

The **hyperparameters** are associated with the loss and optimizer. For example, **learning rate** is a hyperparameter for your optimizer, along with **momentum** and so on.

### Training Loop Code *(reconstructed)*

```python
import torch
import torch.nn as nn
import torch.optim as optim

model     = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()              # zero out previous gradients
        outputs = model(inputs)            # forward pass
        loss    = criterion(outputs, labels)
        loss.backward()                    # compute gradients
        optimizer.step()                   # update weights
```

### Why Zero the Gradients?

Going back to the computational graph: when PyTorch calculates gradients, it calculates them at each node. But **it does not automatically flush these buffers**. If you do not manually flush them, PyTorch keeps adding to the previously calculated gradients.

### Why Doesn't PyTorch Flush Automatically?

Because when you are doing **sequence labelling tasks**, you want to backpropagate gradients over time. In some cases, you need to keep accumulating them.

> **So you manually zero out the gradients with `optimizer.zero_grad()`**. Otherwise PyTorch will keep accumulating older gradients and add them to your current ones.

### `model(inputs)` Convention

There is a convention in PyTorch: `model(inputs)` is the same as `model.forward(inputs)`. You take your inputs and apply the transformation defined in `forward`.

---

## Convergence and Overfitting Monitoring

### Plotting Losses

> **Always keep plotting your training loss as you are training.**

- Every iteration, plot **training loss**.
- Every 5 or 10 epochs, also plot **validation loss**.

You will see your training loss going down, and your validation loss should also be going down. But sometimes at some point your validation loss might start increasing. That is when you **cut off training**, because otherwise you **overfit**.

### Training Curve Diagram *(reconstructed)*

```
Loss
  ^
  |\
  | \
  |  \
  |   \___         <-- training loss keeps dropping
  |    |  \___
  |    |      \___
  |    |
  |    |           <-- validation loss stops dropping, starts rising
  |    |______
  |    |      \
  |    |       \___         OVERFITTING STARTS HERE
  |    |            \____
  +--------------------> Epochs
```

### Sanity Check: Loss Going Up

If you see your training loss in the **initial iterations going up**, there is something wrong:

- In your **data loading**
- In the way you are setting up your **learning rate**
- In your **code**

So go find the bug and fix it, because at least initially, training loss should decrease.

### Debug Trick: Overfit to a Small Dataset

> **Hack when training neural networks**: when you have written your code, start with a really small dataset and try to overfit to that dataset.

If your model does not overfit to that, there is a bug in your code. Why? Because neural networks are **function approximators**. Whatever input and output you give it, it should learn that function. If your dataset is really small, it should memorize that data. If it does not, then something is wrong.

> **Course note**: This is a very easy way of debugging, because it is very easy to write wrong code and still get seemingly correct results.

---

## Evaluation

Once you have done your training, you evaluate.

### `torch.no_grad()` Block

For evaluation, wrap everything in `torch.no_grad()`:

```python
model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        # calculate metrics, do not track gradients
```

`torch.no_grad()` sets `requires_grad` to `False` for all tensors inside the block. For any learnable parameters, anything under this block, none of these are tracking gradients, because this is evaluation and you do not need to track gradients for these operations.

### Why We Need `no_grad`

During training, you set `requires_grad=True`, and those gradients were accumulating. When you again pass through those same learnable parameters during evaluation, if you do not set `no_grad`, it will still track those gradients. You do not really need it. It will not affect anything, but it is just **best practice**, because sometimes you do things in the evaluation block that might cause gradient calculation. If you do any non-trivial operation, it might calculate the gradients for those.

### Test Set Evaluation

The test set is a totally **separate held-out set** that you want to get predictions for, and calculate your confusion matrix and so forth. You use the same `torch.no_grad()` pattern for the test set evaluation.

### Main Function Pipeline

The main function ties it all together: load data, define model, train, evaluate.

---

## Validation Strategies

### Validation in the Training Loop

You had a validation set. When training, every N epochs (every 5 or 10), you do a pass through your validation set to get predictions and loss, so you can plot it on your **learning curve**.

### After Validation: Put It Back?

- **Question**: Do you add the validation back to the training and retrain?
- **Answer**: It depends on how you set up your training.

### K-Fold and Leave-One-Out Validation

If you have **very little data**, you can do:

- **Leave-one-out validation**: If you have 24 training examples, take 23 as training, keep 1 as validation. Shuffle, put that one back, keep another one out. Repeat.
- **K-fold validation**: separate your set into K folds, take K−1 as your training set and the last one as validation. It is basically leave-one-out but on groups of samples.

### When to Use Separate Splits

But these days, datasets are pretty big, so the easier approach is:

- Separate **training split**, **validation split**, and **test split**.
- You have enough data for statistically significant results.

> **K-fold is valid, but with slightly bigger data, separate splits are the easiest approach.**

---

## Computer Vision with PyTorch

PyTorch's robustness in computer vision comes from its comprehensive libraries, especially **`torchvision`**, which includes:

- **Pre-trained models**
- **Datasets**
- **Image transformation tools**

It enables tasks such as:

- **Image classification**
- **Object detection**
- **Segmentation**

> **Example**: using a pre-trained **ResNet** model from `torchvision`, one can easily implement **transfer learning** for custom image classification tasks.

### Transfer Learning Sketch *(added)*

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Load a pretrained ResNet50
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze feature extractor
for param in resnet.parameters():
    param.requires_grad = False

# Replace final classifier for a new task with 5 classes
resnet.fc = nn.Linear(resnet.fc.in_features, 5)
```

This pattern will be covered in detail in tomorrow's class.

---

## Advanced Features in PyTorch

PyTorch's advanced features enable performance and deployment beyond research prototypes.

### CUDA Support

PyTorch supports **CUDA**, enabling **GPU acceleration** for faster computation.

### Distributed Training

PyTorch offers **distributed training capabilities**, essential for handling **large datasets** and **complex models**. This is covered in the next section on multi-GPU training.

### JIT Compiler and TorchScript

PyTorch's **JIT Compiler** improves performance by converting models to **optimized TorchScript**. TorchScript lets models be serialized and run independently of Python, which is important for deployment.

### C++ Front-End

PyTorch's **C++ front-end** allows integration with **C++ codebases**, enhancing flexibility and efficiency in model deployment.

---

## Multi-GPU Training

There are more advanced things you can do with PyTorch: train on **multiple GPU devices**. You can say you want to load these tensors to this device and those tensors to that device.

### The VRAM Problem

Your GPU has limited memory. The **VRAM** of your GPU is finite. If you have a model like **LLaMA** that is around **400 gigabytes**, that is not going to fit on one GPU.

### Model Sharding

Think of the model as one big matrix.

- Take this part of the matrix, put it on **GPU 1**.
- Take this part of the matrix, put it on **GPU 2**.
- And so on.

### Data Sharding

Similarly, with your data, if you have a huge dataset:

- Take this part of the dataset, put it on **GPU 1**.
- Take this part on **GPU 2**.
- And so on.

### Gradient Synchronization

The whole challenge is how to accumulate the gradients, because each GPU is running its own part. You have to **synchronize training**: one GPU finishes, then you take the gradients from there and update everything else.

> **Advice**: Once you get comfortable with PyTorch, look into model sharding and data sharding so you know how to leverage multiple GPU devices.

### CPU vs GPU Specialization

- **GPU**: good at **repeatable actions**. You give it one function and it will repeat that on all your data in parallel.
- **CPU**: good at **sequential operations**.

You should know what operations should run on CPU, what operations should run on GPU, and how to do that.

---

## Advanced Training and Efficiency

A lot of work these days is basically how do you make all of this more efficient.

### Study Recommendation: Micrograd

You can write autograd code in about **100 lines**. I highly suggest looking at **Andrej Karpathy's series called `micrograd`**, where he implements PyTorch autograd from the bottom up. You get a good sense of what exactly is happening when you do backpropagation.

> **Course note**: None of this is strictly necessary. Without knowing all of this, you can still implement neural networks in PyTorch. The problem comes when you are **debugging**. To really understand what is happening in the backward pass, it helps to know how gradients are calculated and where you might be getting **exploding** or **vanishing** gradients.

### Recap: What PyTorch Gives You

| Concept | What It Is |
|---------|-----------|
| **Tensor** | A data type. The only point is that tensors allow you to keep track of gradients, so you use tensors instead of regular arrays. |
| **Autograd** | Keeps track of gradients at every computation. |
| **`nn.Module`** | The main building block in PyTorch. Used for losses, layers, and everything. |
| **Training loop** | Where you define your optimizer and loss function. |

---

## Lab 5 Recommendations

> **Course note**: If you really want to understand how to use PyTorch, for lab 5:
>
> - Implement **cross-entropy** as a custom loss.
> - Implement one of the modules as a **custom layer**.
> - Implement a **custom dataset**.
>
> You already have the ground truth, because everything is implemented in PyTorch and all the code is open source. You can look at how `Conv2d` is implemented, then implement your own custom convolution and compare where you went wrong. That is how you learn.

---

## Debugging Tips

### Purposeful Overfitting on Small Data

> **Debugging process**: once you have written your code, instead of training on your entire dataset, take a very small subset, like **10 examples**, and it has to overfit.

If your training curve does not go down very fast, then there is something wrong in your code. Overfit to that small dataset so that every time you give it an example, it has to predict the correct class most of the time. If it is not doing that, something is wrong.

> **Very easy way of debugging, because it is very easy to write wrong code and still get seemingly correct results.**

### Looking at Gradients

There are different ways of looking at the gradients to make sure things are working, especially when initializing parameters.

### Initialization

> **Do not initialize everything to zeros. That will really hurt your training.**

Initialization is very important in neural networks because backpropagation is very sensitive. The gradients can blow up or they can vanish very easily, so you need various tricks to handle that.

### *(Additional example)* Xavier/He Initialization

Standard initializations you will encounter:

```python
import torch.nn as nn

# Xavier for tanh/sigmoid
nn.init.xavier_uniform_(layer.weight)

# He for ReLU
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
```

These scale the initial weights based on layer size, which helps keep activations and gradients in a stable range.

---

## Best Practices in PyTorch

To ensure efficient and effective model training in PyTorch, follow these best practices:

1. **Use GPU acceleration** where possible.
2. **Properly split data** into training and validation sets.
3. **Utilize PyTorch's inbuilt functionalities** like `DataLoader` for data management.
4. **Regularly save and load models** during training:
   - Prevents data loss.
   - Allows for fine-tuning.
5. **Keep code modular and well-documented** to enhance readability and maintainability.

### Saving and Loading Models *(added)*

```python
# Save
torch.save(model.state_dict(), "model.pth")

# Load
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

---

## Community and Ecosystem

PyTorch is supported by a **strong community** of developers and researchers. The PyTorch ecosystem includes:

- **Extensive documentation**
- **Tutorials**
- **Forums** for discussions

Notable contributions come from both **academic** and **industry leaders**, ensuring continuous improvements and updates. This robust support system is invaluable for both beginners and advanced users for:

- **Troubleshooting**.
- **Keeping up with the latest developments** in deep learning.

---

## Industry Relevance and Tomorrow's Class

### What's Next

In tomorrow's class, we will cover **transfer learning**.

### Syllabus Update: MMPretrain to Hugging Face

> **Course note**: The syllabus originally had **MMPretrain**, but MMPretrain was a good library about three years ago. The professor who created it passed away, nobody is maintaining it anymore, and it has become outdated. The **Transformers ecosystem** has moved on and MMPretrain has not been updated. The engineering behind it is still good.

You should also learn **Hugging Face**. There is no Hugging Face in your syllabus, but it has been added because you should know it.

Tomorrow's class will show transfer learning through:

- **Hugging Face**
- **PyTorch**
- **PyTorch Lightning**
- Other tools

Then you can decide what you want to use.

### What to Know for ML Engineering Jobs

If you want to work as an engineer, the things you need to know:

- **Python**
- **PyTorch**
- **Hugging Face**
- **LangGraph**
- An **observability library**, like **LangSmith** or **Arize**

There are a lot of job openings. If you look at job postings, there are a lot of openings for **AI engineer**. The description varies: some want a software engineer who can do API calling, some want data scientists. Nobody really knows what they want these days.

> **Key takeaway**: If you just know **LangGraph** and the **agent stuff**, you are in good shape.

### Agents: Engineering, Not ML

A lot of the agent stuff is more **engineering than ML**, because it is basically how to orchestrate different things. It is knowing how to do **distributed programming** and those kinds of things.

These days, you are not really training models from scratch that much. And even if you are training, people are doing this thing called **LoRA**, which is where you freeze most of the model and train just a small part, because the models are so big.

### Where Model Development Actually Happens

- **Actual model development**: relegated to the really big **AI labs**.
- **Startups and small/medium businesses**: mostly **API calls**.

But you need to know how to debug stuff. For example, knowing things like someone using validation data in the training set. People will also ask why you would use this model versus that model. So you should know the advantages of different models and be able to talk about them.

### Data Engineer vs AI Engineer

A lot of times people conflate **data engineer** with **AI engineer**.

- A **data engineer** sets up data pipelines. For that, you need to know how to do **MapReduce pipelines** and those things.
- An **AI engineer** is more focused on model use, agents, and orchestration.

### Keeping Up with the Industry

A lot of schools are facing the problem that their established syllabus does not match what is required in industry these days.

> **Highly suggested**: Learn how to use **code generation tools**. Become efficient with **Cursor** and similar tools, because a lot of the code is going to be generated for you. You should know how to guide it, how to set up the guardrails so that it only modifies that one function and not your entire codebase.

Stanford has a modern software development course with modules on how to use a lot of these tools. Highly recommended.

---

## Conclusion and Key Takeaways

**PyTorch** stands out as a **flexible**, **intuitive**, and **powerful** tool for deep learning, especially in **computer vision**. Its **dynamic nature**, **strong GPU support**, and **extensive community** make it a top choice for both researchers and industry professionals. As we continue to witness rapid advancements in AI and machine learning, PyTorch is well-positioned to remain at the forefront of innovation.

> **Once you understand autograd, that basically is PyTorch.** In about 100 lines of code, you can write PyTorch. Everything around it is about increasing the efficiency of how you move tensors from GPU to CPU, and so on.

The PyTorch pipeline breaks into:

1. **Data** (Dataset + DataLoader, custom or `ImageFolder`)
2. **Model** (Lego-block composition of layers in `__init__` + `forward`)
3. **Training loop** (zero grads, forward, loss, backward, step)
4. **Evaluation** (wrapped in `torch.no_grad()`)
5. **Main function** (ties it all together)

Customization patterns all look the same: inherit from `nn.Module` (or `Dataset`), then define `__init__` + one more method (`forward` for layers and losses, or `__len__` + `__getitem__` for datasets).

> **This will be like your bread and butter. If you do any kind of ML engineering, this is what you do.**
