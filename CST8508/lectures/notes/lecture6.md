# Lecture 6: PyTorch Code Walkthrough

## Topics
1. Autograd Example
2. PyTorch Pipeline Building Blocks
3. Custom Dataset and DataLoader
4. Custom Layer
5. Custom Loss
6. Data Preparation with ImageFolder and Transforms
7. Model Definition (Lego Blocks)
8. Training Loop
9. Evaluation

---

## Autograd Example

We start with a simple function, f(x) = 3(x + 2)^2. PyTorch creates a **computational graph** that breaks this into individual operations: **addition** (x + 2) and **multiplication** ((x+2) * (x+2) * 3). Each node in the graph saves the **gradients** flowing through it.

When you call **backward()**, it calculates all gradients via the **chain rule**. For this function, you compute df/dC, then dC/dA, dC/dB, dA/dX, dB/dX and accumulate them.

To actually compute gradients, the tensor needs **requires_grad=True**. With X = 1, calling backward gives a gradient of **18**, because the derivative of 3(x+2)^2 is 6(x+2), and 6(1+2) = 18.

You can access the gradient value via the **.grad** attribute, which is useful for debugging (e.g., checking for **exploding** or **vanishing gradients**). If you don't need to track gradients (validation, inference), set **requires_grad=False** to save memory.

---

## PyTorch Pipeline Building Blocks

A PyTorch pipeline has five parts:
1. **Data loading**
2. **Model definition**
3. **Training loop**
4. **Evaluation**
5. **Main function** to tie it all together

---

## Custom Dataset and DataLoader

### Dataset Class

To create a custom dataset, inherit from **torch.utils.data.Dataset** and define three methods:

1. **\_\_init\_\_**: Define the data path, load file paths, and assign labels. E.g., read a folder structure where each subfolder name is a class label, or read from a CSV.

2. **\_\_len\_\_**: Return the total number of samples. Called when you do `len(dataset)`.

3. **\_\_getitem\_\_**: Define how to retrieve one sample. This is called by the DataLoader when it calls `next`. Open the image, apply transforms (e.g., grayscale to RGB), and return it.

### Example: Two-Class Dataset

In **\_\_init\_\_**:
- Provide the root directory
- Set up a transform pipeline (crop, resize, etc.)
- Assign labels to each class folder

In **\_\_len\_\_**: return the dataset size.

In **\_\_getitem\_\_**: open the image, apply transforms, return it.

### ImageFolder Shortcut

**ImageFolder** does the same thing automatically. It requires data organized as:
```
root/
  class_a/
    img1.jpg
    img2.jpg
  class_b/
    img1.jpg
    img2.jpg
```
It reads the folder structure and assigns folder names as labels.

### DataLoader

The **DataLoader** wraps your dataset class. It is a **generator**: calling `next` on it yields the next **batch**. You create separate DataLoaders for training, validation, and test sets.

---

## Custom Layer

To define a custom layer, inherit from **nn.Module** and define two methods:

1. **\_\_init\_\_**: Initialize **learnable parameters** (weights, biases). Define input/output sizes.

2. **forward**: Implement the function. E.g., for a linear layer: `output = W * x + b`.

Example shown: a custom linear layer implementing **AX + B**, where W and b are initialized as random learnable parameters.

You do **not** need to define the backward pass. PyTorch computes it automatically via autograd. The only exception is functions that are not fully differentiable, in which case you provide an approximation in a custom **backward** method.

---

## Custom Loss

Also inherits from **nn.Module** with **\_\_init\_\_** and **forward**.

Example shown: **mean squared error loss**
```
loss = mean((prediction - target)^2)
```
The forward method computes the mean of the squared difference between predictions and targets across all samples.

---

## Data Preparation with ImageFolder and Transforms

In the code example, data preparation involves:

1. **Transforms**: Resize images so they are all the same size (required for batching/stacking). Other transforms include cropping, normalization, etc. All preprocessing goes under transforms.

2. **Normalization**: The mean and standard deviation values used (e.g., [0.485, 0.456, 0.406]) come from **ImageNet**. This is standard practice in computer vision. You could also calculate them from your own training split, but never from the full dataset (that would be **data leakage**).

3. **Read the dataset** using ImageFolder (or your custom dataset class).

4. **Split** into training, validation, and test sets.

5. **Create DataLoaders** for each split.

---

## Model Definition (Lego Blocks)

Model definition in PyTorch is like **Lego**. Individual layers (convolution, ReLU, pooling, linear) are predefined blocks. Your job is to assemble them into a custom architecture.

### \_\_init\_\_: Define all layers

The architecture shown has:
- **Conv layer 1** and **Conv layer 2** (with specified kernel sizes and filter counts)
- **Flatten layer**
- **FC1** (fully connected, e.g., 64 inputs to 32 outputs)
- **FC2** (fully connected output layer)

Each layer with different dimensions needs its own instance. E.g., a linear layer with (64, 32) is FC1, and a linear layer with (32, 10) is FC2. Same type of block, different sizes.

### forward: Assemble the function

The forward pass chains the layers together:
1. Input goes through **Conv1**, then **ReLU**, then **max pooling**
2. Then through **Conv2**, then **ReLU**, then **max pooling**
3. **Flatten** with `view(-1, ...)` where **-1 infers the batch size**. The first dimension is always the batch. Flattening multiplies all other dimensions together (e.g., 32 filters x 32 x 1 becomes a single vector per sample).
4. Through **FC1**, then **FC2** for the output

The individual Lego blocks are predefined, but the **forward function** (how you assemble them) is custom. That design space is infinite.

---

## Training Loop

### Setup
- Define the **loss function** (e.g., cross-entropy)
- Define the **optimizer** (e.g., SGD, Adam) with hyperparameters like **learning rate** and **momentum**

### Loop
For each batch from the DataLoader:
1. **Zero the gradients** (`optimizer.zero_grad()`). PyTorch does not automatically flush gradient buffers because some tasks (e.g., sequence labelling) require accumulating gradients over time. If you don't zero them, old gradients keep adding up.
2. **Forward pass**: `outputs = model(inputs)` (equivalent to `model.forward(inputs)`)
3. **Compute loss**: `loss = criterion(outputs, labels)`
4. **Backward pass**: `loss.backward()` to calculate all gradients
5. **Update weights**: `optimizer.step()`

Repeat until convergence.

### Monitoring Convergence
- Plot **training loss** over iterations. It should decrease.
- Every 5 or 10 **epochs**, also plot **validation loss**.
- If validation loss starts increasing while training loss decreases, **stop**: you are overfitting.
- If training loss goes up in early iterations, there is a **bug** in your code.

### Debugging Hack
Start with a very small subset (~10 examples). Your model **must** overfit to it. If it cannot memorize 10 samples, something is wrong in your code.

---

## Evaluation

### Validation / Test

Wrap the evaluation block in **torch.no_grad()**, which sets **requires_grad=False** for all operations inside it. This:
- Prevents unnecessary gradient tracking
- Saves memory
- Is best practice even though it does not affect predictions

The **validation set** is evaluated periodically during training (every N epochs) to monitor overfitting. The **test set** is a completely separate held-out set used only for final evaluation (confusion matrix, accuracy, etc.).

---

## Summary Table

| Component | Methods to Define |
|---|---|
| **Custom Dataset** | `__init__`, `__len__`, `__getitem__` |
| **Custom Layer** | `__init__`, `forward` |
| **Custom Loss** | `__init__`, `forward` |
| **Model** | `__init__` (define layers), `forward` (assemble them) |

Everything inherits from **nn.Module**. You never need to define **backward** because autograd handles it automatically.
