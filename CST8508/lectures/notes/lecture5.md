# Lecture 5: CNN Classification Fundamentals

## Topics
1. CNN Overview and Architecture Recap
2. Data Set Preparation (Annotation, Pre-processing)
3. Training, Validation, and Test Splits
4. Data Augmentation
5. Deciding the CNN Architecture
6. Activation Functions (Sigmoid, Softmax, Tanh, ReLU)
7. Loss Functions (MSE, Cross Entropy)
8. Gradient Descent
9. Back Propagation (with Worked Example)
10. Optimizers (SGD, Adam, RMSProp)
11. Best Practices for Training
12. Overfitting and Underfitting
13. Hardware Resources (CPU, GPU, TPU)
14. Optimization Techniques (Pruning, Quantization, Efficient Architectures)
15. CNN Integration with Other Techniques (RNN, NLP)
16. Troubleshooting Common Issues

---

## CNN Overview and Architecture Recap

So this week we will be covering the fundamentals of **CNN classification**. Last week itself, we did an overview showing the **input layer**, the **convolutional layers**, the **pooling layers**, the subsequent layers, and then finally the **output layer**. So continuing from there, we are going to cover **data set preparation**, **data augmentation**, how to design a **CNN**, the different **activation functions** that we may be using, **loss functions**, important algorithms like **back propagation**, **computational resources**, and common issues and troubleshooting.

**CNN** is a subset of **machine learning**, and it uses **neural networks**. Last week, we saw the layers of CNN. So we can have multiple layers like **convolutional layer**, **pooling layer**, and so on for a particular number of layers. CNN can be used for tasks like **object detection**, **image classification**, and **semantic segmentation**. Do you know what **semantic segmentation** means? **Semantic segmentation** is a task in **computer vision** which classifies each pixel in the image to a category. For example, an easier way to say it is, if you ask a model to say if this image contains a cat or not, the model can say, yes, this image contains a cat. But for semantic segmentation, how it works is that it tells exactly which pixels belong to that cat. So not only does it tell if it is present or not, it can exactly delineate the pixels belonging to that image of the cat. That is why it is called **semantic segmentation**. So all these are machine learning models that can identify patterns and features present in the images.

### CNN Architecture Diagram

So I hope you are familiar with this diagram now. We have one image, which is being fed into the **input layer**, and the input layer will convert the image data into **numerical data**. And this data is being passed to the next layer, which is the **convolutional layer**. That is the layer where the network is extracting the **features**. The reason we use convolutional layers is that we do not want every pixel in this image for training the model. We just want what is needed, what is special, or what is important from the image. We extract that in the convolutional layer, and the output of the convolutional layer is called a **feature map**. This feature map goes into the next layer, which is the **pooling layer**. In the pooling layer, what happens is it **downsamples** the dimensionality of that feature map. It does that depending on which type of pooling you use, and depending upon the size of the filter. From the pooling layer, the feature maps will be **flattened**. So by flattening, the output of the convolutional layers is multidimensional. We need to make it one dimensional so that it can be fed to the **fully connected layer**. That is where we do that.

In the **fully connected layers**, when the data is being passed on, we do some calculations and we compute the **loss**. Using multiple **iterations** of this, we will try to minimize the loss. Finally, in our **output layer**, the number of **neurons** equals the number of **classes**. So you can see here we have a few classes. For example, there is a sunset class, a dog class, and a cat class. And you can see a **probability score** for each class. Whichever class has the highest probability, that will be the **prediction** of that model. In this case, it should be the correct class. That should have the highest probability. Otherwise it is a wrong prediction. So this is how **image classification** works. We are categorizing and labelling images into their corresponding classes.

---

## Data Set Preparation

So now, **data set preparation**. This is one of the most important parts if you are doing **image classification**. We need to prepare our data because what happens is the more and better data we have, the more efficiently the model can perform. You can create a model, but it may not be accurate or the performance will not be that good. So we need to make sure that we prepare the data properly. The main area of preparing the data is that we need to make sure that a **diverse set of images** is present in our data. Otherwise, the model will be **biased** towards what you are providing it. So we need to have diversity. In this example, we can see we have different classes. When you train your models, we need to provide a diverse set of images so that the model will not be biased and can better handle any kind of variation in images because it has already seen that.

### Annotation

There is one technique which helps us, and that is **annotation**. **Annotation** means it is the process of **labelling data**. You can say this image is labelled as a certain class. So each image is labelled with its **class label**. This process is called annotation. And this is mainly used for **supervised learning**, because we know the labels. We know that the quality and accuracy of the labelled data directly impacts the model's ability to learn and identify patterns. That is why they say you need to train a model with a good data set which has a diverse set of images. For **supervised learning**, we can use techniques like annotation to label the images.

### Pre-processing Techniques

There are different types of **pre-processing techniques** for preparing the images:

- **Resizing:** If your data has images of different sizes, the input is not consistent. CNN would have trouble extracting the features or performing on that kind of data. So if you make them consistent, we can prepare our data. For example, if images are of different sizes, we can resize them to a **uniform size**.
- **Normalization:** You can **normalize** the pixel values into a specific range. If the intensity values are varying drastically, that is not ideal for the CNN. So we can make it consistent by normalizing the values.
- **Colour Space Conversion:** For some techniques, we can convert colour images to **greyscale** or other **colour spaces**.

While doing all these methods, we are making sure that we are providing **consistent input data** to the CNN. So by doing this, we can improve the CNN's performance. That is the reason why we give importance to the pre-processing step.

---

## Training, Validation, and Test Splits

Do you know that the input data is divided into **training**, **validation**, and **test** sets? Do you know why? Why can we not just use the training data? What happens if we just have training data? We do not know if our model is good for anything because it can learn that data and just memorize the values.

### Standard Split

> **Standard split:** 80% Training, 10% Validation, 10% Testing

The majority of our input goes towards training the model, some percentage for validating, and some percentage for testing.

### Training Set

In the **training set**, this is where the model **learns**. We present 80 percent of the data to the model over and over again, and then it learns the **patterns** from that data. By passing the data multiple times, the model can update its **parameters** on each **iteration**.

What happens if we just have the training data? Because we are repeatedly feeding this data to the model, the model can **memorize** the values. So it can perform very well on any input from the training set. But if you give it data that it has never seen, or new data, it cannot perform well because it just memorized the training data.

### Validation Set

So **validation** is necessary. The **validation set** is used for **tuning the hyperparameters**. This is where we tune things like the **learning rate**, the **number of layers**, and all those things. It also helps us decide if we should **stop early**, because we can compare the training data and validation data. If the accuracy of the model is high for training but low for validation, that is a sign of trouble, because there is no point in continuing to train since it would not perform better. We should have the validation accuracy comparable to or higher than the training accuracy, because validation data is like real world data that we are giving to the model, and it should be able to perform well on that. That is the reason why validation is important. This is also where we can detect **overfitting**. Overfitting means the model works very well on the training data but cannot perform well on unseen data.

### Test Set

The **test set** is different from validation because validation is done multiple times during training. The test set is where you actually evaluate the **final performance** of your model. This data is **purely unseen** by the model. Depending on how your model performs on this data, you can assess its quality. That is the reason why we need test data. This is the ideal way, but usually at minimum we should have training and test sets, though it is recommended to have all three.

---

## Data Augmentation

Why is more data important? As I mentioned, the model has to be exposed to as much diverse data as possible. We need to train it with a diverse set of data, then only our model can better deal with variability. But sometimes we do not have enough data. So there is a way to **augment** the images. From just one single image, you can generate 10 or more images by using some **transformation techniques** like:

- **Rotations** (e.g. +45°, −45°)
- **Scaling**
- **Flipping**
- **Cropping**

By applying these techniques, you will get multiple images from a single image. So you are actually **expanding the data set**. If you have a small data set, you can expand it by using **augmentation techniques**. This process helps in reducing **overfitting** because we are already exposing the model to a wide variety of features. So the model can better deal with **generalization**. We are training it with different features and different scenarios that it could encounter. So **augmentation** helps you expand your data set.

---

## Deciding the CNN Architecture

In most cases, we are not designing it from scratch. Usually, we take a **state of the art** model. Models for image classification or other tasks are already available. But in case you are designing one from scratch, you need to consider the architecture. You have to think about:

- **Number of layers**
- **Type of layers** (convolutional, pooling, fully connected)
- **Type of convolution**
- **Parameters:** filter size, **stride**, **activation function**

You remember **stride**, right? The step size with which the filter is moving.

### Architecture Guidelines

All these are key points that we consider when deciding the CNN architecture. There is **no perfect formula**. No one can tell you a perfect formula. But there are guidelines:

| Task Complexity | Suggested Layers |
|-----------------|-----------------|
| Small/simple | 5 to 10 layers |
| Medium | 50 to 200 layers |
| Complex | Up to 500 layers |

Depending on how complex your application is, you can decide how many layers and what type. Sometimes we just need one convolutional layer and one fully connected layer for a simple task. But sometimes that is not enough, so we need to have multiple convolutional layers and pooling layers.

### Filter Depth Pattern

Another thing is that typically the **number of filters increases** as you go deeper:

> **Example:** Layer 1 → 16 filters → Layer 2 → 32 filters → Layer 3 → 64 filters → ...

In the first few layers, we are extracting **simpler features**. The deeper we go, the more **complex features** will be extracted, and that requires more filters. So the number of filters in each layer may increase as you go deeper into the network. That is a key point.

The architecture should match the **complexity of the task**. For more complex tasks, you need more layers and **computational efficiency** considerations. If your model is complex and you choose a very simple architecture, it will not be computationally sufficient. Your model cannot handle that. So you need the computational power and layers to deal with that kind of task. You need to keep all these considerations in mind: the **number of layers**, **types of layers**, and their **parameters**. There is no perfect formula, but depending upon the requirements, we can decide. It is always **trial and error**.

---

## Activation Functions

We talked about **activation functions** last week. The activation function is a function that introduces **non-linearity**. In a **fully connected layer**, there is a connection between all the neurons. We have input data that is being transmitted through these connections. But sometimes we do not need all the neurons to transmit to the next layer because they carry unneeded information. Maybe it is the background, which is not needed for further processing, so we can ignore it. That is what an activation function does. It decides that some neurons are not needed, so it can **zero out** or reduce the output of those neurons. The activation function, depending upon the output, decides whether to pass the information forward or not.

Another point is that the activation function introduces **non-linearity**. If our model is linear, it is just like a simple linear equation. A linear model cannot learn complex patterns like images or similar data. In that case, we need to introduce some **non-linearity** to our network, and for this, we can use activation functions.

### Neuron Calculation

In neurons, we have inputs with **weights**. Each input is multiplied by its weight, and we add a **bias**. This **weighted sum** plus bias is then passed to the activation function.

> **Neuron output:** z = (Σ wᵢ × xᵢ) + b
>
> **Activated output:** a = f(z)
>
> Where:
> - **xᵢ** = input values
> - **wᵢ** = weights
> - **b** = bias
> - **f** = activation function

**Example:**

Given inputs x₁ = 2, x₂ = 5, x₃ = 1 with weights w₁ = 0.3, w₂ = 0.5, w₃ = 0.2 and bias b = 0.1:

> z = (2 × 0.3) + (5 × 0.5) + (1 × 0.2) + 0.1
>
> z = 0.6 + 2.5 + 0.2 + 0.1 = **3.4**
>
> Then a = f(3.4), where f is the chosen activation function.

### 1. Sigmoid

**Sigmoid** is one of the popular activation functions, used for **binary classification**, and it always ranges between **0 and 1**.

> **σ(x) = 1 / (1 + e⁻ˣ)**

| Input x | σ(x) |
|---------|-------|
| −5 | ≈ 0.007 |
| 0 | 0.5 |
| 5 | ≈ 0.993 |

Towards the edges, it flattens out and becomes stable. This leads to the **vanishing gradient problem**, but we still use sigmoid for **binary classification**.

### 2. Softmax

**Softmax** is similar, ranging between 0 and 1, and it is mainly used for **multi-class classification**.

> **softmax(zᵢ) = eᶻⁱ / Σ eᶻʲ** (for all classes j)

**Example with 3 classes:**

Given raw scores z = [2.0, 1.0, 0.5]:

> e²·⁰ = 7.389, e¹·⁰ = 2.718, e⁰·⁵ = 1.649
>
> Sum = 7.389 + 2.718 + 1.649 = **11.756**
>
> P(class 1) = 7.389 / 11.756 = **0.628**
>
> P(class 2) = 2.718 / 11.756 = **0.231**
>
> P(class 3) = 1.649 / 11.756 = **0.140**

Whichever class has the highest probability, you can assign that classification. So **softmax** is mainly used for **multi-class classification**.

### 3. Tanh

**Tanh** has a wider range compared to sigmoid. Instead of 0 to 1, it ranges from **−1 to +1**.

> **tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)**

| Input x | tanh(x) |
|---------|---------|
| −2 | ≈ −0.964 |
| 0 | 0 |
| 2 | ≈ 0.964 |

So it is more balanced because it includes negative values as well. It is a **differentiable** activation function, and it also has the **vanishing gradient problem** because towards the edges, it goes flat and stabilizes, similar to sigmoid. It is useful when dealing with **negative input values**.

### 4. ReLU

**ReLU** (Rectified Linear Unit) is very popular.

> **ReLU(x) = max(0, x)**

| Input x | ReLU(x) |
|---------|---------|
| −3 | 0 |
| 0 | 0 |
| 5 | 5 |

It takes only values that are greater than zero, and for all other values, the output is zero. So if a neuron's output is negative, it just keeps it at zero. ReLU ranges from **zero to infinity**. It takes away the negative values while keeping positive values. It is **computationally efficient** and is commonly used in **hidden layers** and **feed-forward networks**. It is not commonly used in the output layer.

### Activation Function Summary

| Function | Range | Best For | Drawback |
|----------|-------|----------|----------|
| **Sigmoid** | (0, 1) | Binary classification | Vanishing gradient |
| **Softmax** | (0, 1) | Multi-class classification | Computationally heavier |
| **Tanh** | (−1, 1) | Negative input values | Vanishing gradient |
| **ReLU** | [0, ∞) | Hidden layers | Dead neurons (output always 0) |

---

## Loss Functions

The **loss function** measures how well the model's predictions match the true outcomes. So we have the expected output, but what we actually get from the model is different from that. We take the difference between the two. If the loss is **high**, it means the difference is large, and the model is not matching well. If the loss is **low**, it means it closely matches or is giving a good prediction. So we find the difference between the prediction and the actual value. This is also called the **cost function**. Based on this, we can use **back propagation** to adjust the **weights** and **bias** parameters. Our goal is to get the **minimum loss**.

### Mean Squared Error (MSE)

> **MSE = (1/n) × Σ (Yᵢ − Oᵢ)²**
>
> Where:
> - **Yᵢ** = expected (true) output
> - **Oᵢ** = actual (predicted) output
> - **n** = number of output neurons

**Example** (from the binary classification walkthrough):

Expected outputs: Y = [1, 0]. Predicted outputs: O = [0.2, 0.49].

> MSE = (1/2) × [(1 − 0.2)² + (0 − 0.49)²]
>
> MSE = (1/2) × [(0.8)² + (−0.49)²]
>
> MSE = (1/2) × [0.64 + 0.2401]
>
> MSE = (1/2) × 0.8801 = **0.44005**

The individual squared errors: (1 − 0.2)² = 0.64 and (0 − 0.49)² = 0.2401, summing to approximately **0.042601** per the lecturer's example (using a slightly different formulation without the 1/n divisor per neuron).

### Cross Entropy Loss

**Cross entropy** comes in two types:

- **Binary Cross Entropy:** for **binary classification**
- **Categorical Cross Entropy:** for **multi-class classification**

> **Binary Cross Entropy = −[Y × log(O) + (1 − Y) × log(1 − O)]**

Our goal is to reduce the loss. We compute the loss for each iteration, and we continue until we get the minimum loss.

---

## Gradient Descent

**Gradient descent** is an **optimization algorithm** that we use to minimize the loss. How we do this is, when we calculate the **gradient** at a current position, if it is pointing towards the left, we move it towards the right direction, and vice versa. That is how gradient descent works.

> **Weight update rule:** w_new = w_old − α × (∂Loss/∂w)
>
> Where:
> - **α** = learning rate
> - **∂Loss/∂w** = gradient of the loss with respect to the weight

Depending upon the gradient at the current position, we can compute how to move towards the **optimal position** (the minimum of the loss curve). Our goal is to reach the minimum. When the model is trained, our goal is to have the minimum loss. When we compute the loss, if it is on one side, it should move towards the other side to reach the minimum point.

---

## Back Propagation

This brings us to **back propagation**, which is very important. **Back propagation** is the algorithm that helps you adjust the **weights** and **parameters** so you can optimize your network to give the minimum loss. All the functions which I mentioned earlier, like the **loss function** and **gradient descent**, are all used in back propagation. At a higher level, all these things are happening in back propagation. It will help us to minimize the loss.

### How Back Propagation Works

For a model, what happens is that we present the input to the model, we pass the input through different layers, like the **convolutional layer**, **pooling layer**, and **fully connected layer**. While the data is being passed through this model, we do some calculations, and in our **output layer**, we get the loss. Since we have the loss or error, our goal is to minimize it. For that, we use back propagation. We **propagate that error back** from the output layer to the layer before it, then to the layer before that. And we adjust the weights and biases. If the prediction is still not correct, then again, we propagate this error back, and we adjust the weights again. This is the idea of back propagation.

> **Back propagation happens only during training**, not during the validation or testing phases. During testing, we are just passing the input to see how the model predicts. We are not training or adjusting any weights.

### Steps of Back Propagation

1. **Feed Forward:** Feed the sample to the network and get the output.
2. **Calculate the Error:** Compare predicted output with expected output (MSE). For **supervised learning**, we know what the expected output should be. For example, if it belongs to the class cat, we expect that output.
3. **Compute Output Error Terms:** Compute the error term for each output neuron, which represents how much each neuron is contributing to the error.
4. **Propagate Error to Hidden Layers:** Hierarchically propagate the error backward.
5. **Apply Delta Rule:** Adjust the weights.
6. **Repeat:** Run forward propagation again. Repeat until minimum loss is achieved.

**Feed forward** (also called **forward propagation**) means we are taking the input, passing it through the inner layers, and finally getting the output. In the back propagation part, we take the error, compute the error term for each neuron, and hierarchically propagate the error through the hidden layers. We adjust the weights and do it again until we get the minimum loss.

### Worked Example: Binary Classification

Let us say this is a **binary classification**. In binary classification, we have just two outputs. Either it should be a **0** or **1**. 1 means it belongs to that class, 0 means it does not belong to that class.

#### Step 1: Forward Pass

Let us say we have input values x = [x₁, x₂, x₃] and we have weights for the connections between layers, and a bias. What we do is we take each input, multiply it with its weight, add the bias, and apply the activation function.

> **For each neuron:** z = Σ(xᵢ × wᵢ) + b, then a = f(z)

In the output layer, we can see only **2 neurons** because the number of neurons in the output layer equals the number of classes. In binary classification, we have just 2 classes, so we have 2 neurons in the output layer.

Let us say we got output values of **O = [0.2, 0.49]**. This is not what we expected (Y = [1, 0]), but that is what we got in the first iteration.

#### Step 2: Calculate Mean Squared Error

We take the difference between expected and actual output.

> **MSE = Σ (Yᵢ − Oᵢ)²**
>
> = (1 − 0.2)² + (0 − 0.49)²
>
> = (0.8)² + (−0.49)²
>
> = 0.64 + 0.2401
>
> = **0.8801**
>
> Per the lecturer's working: the total error comes out to approximately **0.042601**

#### Step 3: Compute Output Error Terms

For each neuron in the output, we compute the error term using the **derivative of the activation function** (sigmoid derivative):

> **δₒᵤₜ = Oᵢ × (1 − Oᵢ) × (Yᵢ − Oᵢ)**

**For output neuron 1** (O₁ = 0.2, Y₁ = 1):

> δ₁ = 0.2 × (1 − 0.2) × (1 − 0.2)
>
> δ₁ = 0.2 × 0.8 × 0.8 = **0.128**

**For output neuron 2** (O₂ = 0.49, Y₂ = 0):

> δ₂ = 0.49 × (1 − 0.49) × (0 − 0.49)
>
> δ₂ = 0.49 × 0.51 × (−0.49) = **−0.1224**

#### Step 4: Calculate Hidden Layer Error Terms

Once we have the output layer error, we **propagate it back** to the hidden layers. The main idea is that whatever error we have in the output layer, we are propagating it back through the layers. That is why we call it **back propagation**.

> **δₕᵢddₑₙ = Oₕ × (1 − Oₕ) × Σ(δₒᵤₜ × wₕₒ)**
>
> Where:
> - **Oₕ** = output of the hidden neuron
> - **δₒᵤₜ** = error term of the connected output neuron
> - **wₕₒ** = weight connecting hidden neuron to output neuron

We compute this for each neuron in each hidden layer, propagating from the output layer backward to the input.

#### Step 5: Apply Delta Rule and Adjust Weights

After computing all the error terms for each neuron in the network, we apply the **delta rule**:

> **ΔW = η × δ × input**
>
> Where:
> - **η** (eta) = learning rate
> - **δ** = error term for the neuron
> - **input** = the input to that weight (output of the previous layer)

**Example weight update:**

Given learning rate η = 0.5, error term δ = 0.128, and input = 0.6:

> ΔW = 0.5 × 0.128 × 0.6 = **0.0384**
>
> w_new = w_old + ΔW

We compute the **ΔW** for each connection in the network. Once we get the delta value, we apply it to each weight. That is how we adjust the weights.

#### Step 6: Repeat

After adjusting the weights, we again do the same process, the **forward pass**, and compute the error again. Basically, the input is being fed to the network, we compute the mean squared error, compute the error term for each output neuron, propagate the error to the hidden layers, apply the delta rule, and finally adjust the weights. After adjusting the weights, you run it again to make sure the model is improving. That is how we do the training. This is just **one iteration**. We need to do **many iterations** until we get the minimum loss.

---

## Optimizers

Common **optimizers** include:

| Optimizer | Description |
|-----------|-------------|
| **SGD** (Stochastic Gradient Descent) | Classic optimizer, simple to implement |
| **Adam** (Adaptive Moment Estimation) | Adaptive learning rates, works well for different problems. **Recommended.** |
| **RMSProp** | Root Mean Square Propagation, adapts learning rate |

**Adam** is very popular and is a recommended one. The choice of optimizer affects the **speed and quality of training** because if you select the right optimizer, the training will be faster and more efficient. It is another factor to consider when you are designing your CNN. Most commonly, **SGD** and **Adam** are used. Adam is mostly recommended.

---

## Best Practices for Training

As I mentioned, it is best to have **training, validation, and test sets**. It is not mandatory, but it is the best practice. We should have **high accuracy** and **low loss** for validation data.

### Training Process Summary

Each iteration in training consists of:

1. **Forward pass:** Data flows through the network from input to output.
2. **Compute the loss:** Compare prediction with expected output.
3. **Back propagation:** Calculate the gradients.
4. **Adjust weights:** Using the optimizer.

Writing a CNN involves: **forward propagation**, where you get the prediction; computing the **loss**; doing **back propagation** to calculate the gradients; and finally using **optimizers** to adjust the weights.

### Hyperparameters vs Learnable Parameters

| Type | Examples | Set When? |
|------|----------|-----------|
| **Hyperparameters** | Learning rate, number of layers, batch size | Initialized before training; some can be adjusted during training |
| **Learnable parameters** | Weights, biases | Learned during training |

### Key Best Practices

- **Use a validation set.** Most of the time, people tend to use training and test only, but the best practice is to use validation. It helps us tune the **hyperparameters**.
- **Apply early stopping** to prevent **overfitting**. If you train too long, the model becomes too fitted to the training data. If we see that the accuracy of the validation set is less than training, it is an indication that we should stop training.
- **Periodically save your model state.** You can save the model state for recovery. Things can happen at any time, so you do not have to start the training from scratch if something goes wrong. If some interruption happens, it will be saved, and when it resumes, you can continue from where you left off.
- **Monitor the training process** by tracking **loss** and **accuracy**, both on training and validation data. If the model accuracy is too good on training but keeps increasing in loss for the validation, it is not going in a good direction. But if the accuracy is good for both, the model is performing well.

---

## Overfitting and Underfitting

### Overfitting

**Overfitting** is when the model becomes too good at learning the training data, including its noise. It can even learn the noise from the data, and it shows poor performance on new or unseen data like the test data. It happens usually with complex models that have too many parameters.

**Model Loss Graph Example:**

> Training loss: **decreasing** ↓
>
> Validation loss: **increasing** ↑
>
> This divergence is a sign of overfitting, because the training loss is getting minimized but the validation loss is not.

**Symptoms:** Much higher accuracy on the training data compared to validation data.

#### Techniques to Prevent Overfitting

1. **Dropout Layers:** Randomly deactivate some neurons during training. In one iteration, you can drop one set, and in the next iteration, you can drop a different set. By deactivating random neurons during training, it can prevent the **co-adaptation of features**. Otherwise, the model will memorize the patterns.

2. **Regularization Methods:** Techniques like L1/L2 regularization help reduce overfitting.

3. **Data Augmentation:** Expand your data set by applying transformations like rotation, shearing, flipping, zooming.

4. **Simplify the Model:** Reduce the number of layers. Depending upon the complexity, use as few as possible.

5. **Early Stopping:** Halt training when performance on the validation set starts to degrade.

### Underfitting

**Underfitting** is the opposite of overfitting. It means the model cannot learn anything meaningful from the data. The current architecture is not good enough to learn the patterns in the data.

#### Techniques to Address Underfitting

1. **Increase model complexity:** Add more layers, because the current architecture is not sufficient.
2. **Provide more diverse data** so that the network can learn.
3. **Train for a longer duration.**
4. **Use more powerful feature extraction techniques.**
5. **Re-examine the data pre-processing steps:** Annotation, normalization, resizing, etc. Whatever pre-processing was done, maybe it needs improvement.

---

## Hardware Resources for Deep Learning

We have different options starting from **CPU**, **GPU**, and **TPU**.

| Hardware | Best For | Details |
|----------|----------|---------|
| **CPU** | Small operations, class assignments | Can handle basic operations but slower for deep learning |
| **GPU** | Medium to large tasks, projects | Thousands of cores, ideal for parallel processing |
| **TPU** (Tensor Processing Unit) | Large scale neural network operations | Designed specifically for matrix operations in neural networks |

Deep learning requires significant computational resources. For our class assignments, we can use CPU, but in your projects, you can introduce GPU because if you are training larger models, it is more computationally demanding.

**TPU** stands for **Tensor Processing Unit**. It is designed specifically for neural network operations. Neural networks deal with **matrix operations**, and depending upon the size of the data, it can be very computationally intensive, so TPUs can help. A student asked what a TPU is and whether it fits in a computer. TPUs are available as **cloud services**, for example, on Amazon you can rent servers that have TPUs. CPUs and GPUs are available locally on your machine, and TPUs are also available through cloud platforms.

TPU is the most powerful option. The choice of hardware can significantly impact the training time. If you are using more powerful hardware, you can finish the training faster. Some training can take **overnight or even 2 days**, depending upon the complexity of the model.

---

## Optimization Techniques

Sometimes we cannot use very high-end processors due to resource constraints. So there are techniques we can use to reduce the computational complexity of the model. You can even use a GPU instead of a TPU if you apply these optimization techniques.

### 1. Pruning

In the network, we have some **redundant neurons**. We do not need redundant neurons, so we can remove them. By doing so, you are reducing the complexity, perhaps by half or more, depending upon how much redundancy there is. There is a linked resource for more details on pruning and quantization that you can refer to for a deeper understanding.

### 2. Quantization

**Quantization** reduces the **precision of the numbers**. We are dealing with numbers in neural networks. Images have pixels with values, and all the computations are mathematical operations. The precision of the numbers matters. We can reduce the precision without significantly affecting accuracy.

**Example:**

> **Full precision (FP32):** 3.141592653589793 (32 bits)
>
> **Half precision (FP16):** 3.140625 (16 bits)
>
> **Integer quantization (INT8):** 3 (8 bits)
>
> Less precision = less memory = faster computation

### 3. Efficient Architectures

There are pre-tested, well-known architectures that are **state of the art**. These architectures are already optimized with the right number of layers, parameters, and configurations. They take care of a lot of computational optimization for us.

For example, **mobile and edge devices** need efficient models. For applications running on devices like phones, we cannot accept heavy computational loads. Everyone wants responses to be fast. So we can use existing efficient architectures that help us achieve the task faster.

One of the key challenges in computer vision is that we deal with images and videos, which require a lot of computational capacity and resources. Research is being carried out to minimize the use of resources while providing faster and more accurate results.

---

## CNN Integration with Other Techniques

The future of CNN will involve integration with other **deep learning techniques**. As I mentioned, deep learning is a broader field, and we have techniques like **natural language processing** and **RNN** (**Recurrent Neural Network**). We can combine CNN with these models to create new models that provide more capabilities.

| Integration | Capability |
|------------|------------|
| **CNN + RNN** | Video classification, image captioning (RNN handles sequential data) |
| **CNN + NLP** | Understanding captions and text in images/media |

By combining different techniques, you can create more sophisticated applications. These integrations enable **multimodal learning**. The main idea is that **CNN is good at processing visual data**, and **NLP can handle the language part**. When we combine them, we get better applications. As the field of AI is evolving, CNN's scope is also growing because we can integrate different technologies.

---

## Troubleshooting Common Issues

### Major Issues

| Issue | Description |
|-------|-------------|
| **Overfitting** | Model performs too well on training data but cannot generalize to unseen data |
| **Underfitting** | Model cannot learn anything meaningful from the data |
| **Bias** | Model is biased towards a particular class |

### Strategies for Troubleshooting

1. **Adjust the learning rate.** The learning rate is a hyperparameter that you can set initially and then adjust during training.
2. **Modify the network architecture.** Replace with a state of the art architecture, or adjust the number of layers or parameters.
3. **Batch normalization and dropout.** Dropout deactivates some of the neurons in the layers.
4. **Use a diverse data set** for training.
5. **Regularly monitor performance** on the validation set and test set. If the model performs inconsistently on the validation set or the test set, that is an indication that something needs to change.
6. After applying these techniques, check **performance metrics** to see if things are improving.

### Addressing Underfitting Specifically

For underfitting, you need to:

- **Increase model complexity** by adding more layers
- **Provide more diverse data** so the network can learn
- **Train for a longer duration**
- **Use more powerful feature extraction techniques**
- **Re-examine the data pre-processing** (annotation, normalization, resizing)
