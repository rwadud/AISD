# Lecture 4: Convolutional Neural Networks (CNN)

## Topics

- Artificial neural networks and their disadvantages for image classification
- Introduction to CNN and its architecture
- CNN layers in depth
- Applications of CNN
- Performance evaluation metrics
- Ethical considerations

---

## Artificial Neural Networks (ANN)

This is taking a computer system which is inspired by the functionality and processing of the human brain. So it's similar to how the human brain works. Like our brain has **neurons**, it also has a large number of nodes which are called **neurons**. We have a lot of neurons in our brain. If we see an object, the optical signal gets transmitted to our eyes and it reaches the brain by transmitting through the neurons. Then it does some processing and we know that we are seeing an object, like maybe I'm watching this projector screen. So it tells my brain that I'm seeing this. It all happens in a fraction of a second. Our brain is that much powerful. Similarly, we are trying to simulate the same processing in our computer world by making use of **artificial neural networks**.

The specialty of artificial neural networks is that they can **learn by analyzing large data sets**. When we were a child, we were learning "this is a chair," so we know that this is a chair because we learned by seeing different shapes of chairs, and even if we see a different shape of a chair, we can identify it. So our brain has that capacity to learn from the **patterns** that we saw over the period of our time. Similarly, we can train our neural networks to learn from the data set. If we have larger data, we can more effectively train them to analyze images and even for recognizing objects from images or performing **classification**, et cetera.

The reason why we need artificial neural networks is that they are particularly effective in recognizing complex patterns. Even if we have the same object present in an image at a different scale or a different rotation, we are able to recognize that. Similarly, neural networks can also be trained for analyzing complex patterns and they can help us in processing images and making decisions. These are the reasons why we can make use of artificial neural networks for image classification. So artificial neural network is a system similar to the human brain.

### Why Not Traditional Methods? (Decision Trees)

Now let's talk about how **predictive models** use traditional methods like **decision trees**. This is an example of the **decision tree** algorithm. I don't want to go into that because this is not part of our course, but I just wanted to show you why we cannot use this in image processing or in machine vision. How predictive models use traditional methods is, on the left hand side there is a table which categorizes images based on different features like ear shape, face shape, the presence or absence of whiskers, et cetera. For example, if it identifies an animal which has a pointy ear shape, then it looks for its face shape, and if it is round, then it looks for whether they have whiskers or not. If it is there, then they categorize that animal into a cat. So this is how we make decisions based on the features that are available in the left table. Other animals with a different combination of these features are classified as not a cat. For example, if the ear shape is floppy, then we will look for the face shape and it says round. Then again, look for the other feature, is whiskers present or not, it says it's absent in the third row. So then the system categorizes it as not a cat. This is how the decision was made using the decision tree.

On the right hand side, you can see a decision tree representation of the same process which I explained in the table. How it works is that it sequentially looks for each feature and then makes the decision. First it starts with ear shape. Then it looks if it is pointy or floppy. Let's say it is pointy, then it moves to the next feature, which is face shape. In the face shape, it looks for whether it is round or not, and if it is round, they categorize that animal as cat. If it is not round, they classify it as not a cat. Similarly, on the right hand side, you can see if the ear shape is floppy, again you look for the presence or absence of whiskers. If whiskers are present, it is a cat. Otherwise, it's not a cat. This is the process of the decision tree, how it classifies an object into cat and not a cat based on the features provided.

### Why ANN Instead of Decision Trees?

Don't you think we can make use of this for **machine vision**? You know that the input we give for **machine vision** systems is mostly images or videos. So we cannot make use of this method for our purpose. We need something different. That's why we say we need neural networks like artificial neural networks. This is our data. We don't have a table with features or a decision tree to say if it is a pointy ear or a round face. We don't have that kind of data. What we have is just some images of cats and dogs. This is our image data set. This is what we are providing to our network and the network should be able to learn from these images. When we deploy this into production, if we give a cat's image, the system or the network should say this is a cat's image. Otherwise, it should say this is not a cat, maybe it's a dog.

### ANN Structure: Input, Hidden, Output Layers

Here you can see a representation of the artificial neural network. We have given the data set as the input. We give both images of cats and images of dogs to train the network. We input that into the input layer. For artificial neural networks there are three main layers: **input layer**, **hidden layer**, and an **output layer**. We are giving our image to the input layer, where the input layer can identify the shape of the ear, shape of the face, et cetera. From that, the data is passed to the hidden layer, and the hidden layer does some processing, and finally it can make a decision or classify into two classes, maybe class dog and class cat. Finally you get two output classes, which are cat and dog. We have a large number of photos of cats and dogs. We pass them into the input layer. In the input layer, they can identify the shape of the ear or the shape of the eye or the presence of whiskers, et cetera. Then it passes to the hidden layer, and the hidden layer does the processing with the features that they got from the input layer. Finally, after processing, we get just two classes which represent cat and dog. This is the overall idea of ANN.

### Limitations of ANN for Image Classification

But there are still limitations for using artificial neural networks for **image classification**. Let's say we have an image of a cat that is 1000 × 1000 pixel size. This is a coloured image, not a **grayscale**. It's a colour image. So it has three **channels**: red, green, and blue. If we represent this in an artificial neural network, the first layer is the **input layer**.

**Example: Why ANN is computationally heavy**

| Component | Calculation | Result |
|---|---|---|
| Input image | 1000 × 1000 pixels × 3 channels | **3,000,000 neurons** in input layer |
| Hidden layer | Apply 1000 filters, each neuron connected to all inputs | 3,000,000 × 1000 = **3,000,000,000 (3 billion) connections** |

Each pixel in that input image is mapped as a neuron to that input layer. Then it goes to the second layer, which is the hidden layer. Each neuron in this hidden layer is connected to every neuron in the previous layer.

You can easily see that it's computationally very heavy. This is just for one image. We have many images in our data set, so it's very heavy computationally. We cannot rely on ANN.

**Three limitations of ANN:**
1. **Computationally heavy** — too many connections (billions)
2. **Overfitting** — because we are giving too many input pixels, we train it too much
3. **Longer training time** — need to train the huge network

---

## Introduction to CNN

What is the solution for this? We cannot use traditional methods because we don't have features or data which is enough for making a decision using decision trees. And we cannot use ANN because it uses each pixel in the input image as a neuron, so the network is computationally very heavy, takes a lot of time to train, and can have overfitting problems.

This brings us to the **convolutional neural network (CNN)**. It is basically similar to ANN but with more layers in the hidden layer, which helps us to focus only on some important data. We don't want all the pixels from that input image. CNN is a **deep learning** technique which is designed specifically for analyzing, extracting, or processing images. As I mentioned, ANN can learn patterns from data. Similarly, CNN also does the same. It can automatically learn patterns from data. Another important aspect is that they can solve complex spatial tasks with **deep learning**. Even if it is a complex image, they can still handle it and then they can perform **object recognition** or **classification**, et cetera.

**Benefits of CNN over ANN:**
- Addresses all limitations of ANN
- Deals with high dimensional structured data (images, video, audio)
- Can **hierarchically** extract features
- Robust to **translation of objects**

---

## CNN Architecture

> This is very important. If you are asked in the exam to draw the architecture, this will be the answer.

We have the input layer, hidden layer, and output layer, same as ANN. But there is a difference because the hidden layer may be replaced by different layers. We have the input layer, and instead of a single hidden layer, we can have a **convolutional layer**, **pooling layer**, and **fully connected layer**. The difference is that the hidden layer is replaced by these three types of layers. We have the same structure but more layers.

```
Input Layer → [Convolutional Layer → Pooling Layer] × N → Fully Connected Layer → Output Layer
              \___________ Feature Extraction ___________/   \______ Classification ______/
```

In this architecture, you can see we are giving the input, and the input image is given to the next layer which is the **convolutional layer**. You can see a lot of passes there. It means we are applying, say, five **filters**. When we apply five filters on this input, we get five **feature maps** or five outputs. That is what the convolutional layer is doing. Then we give that to a **pooling layer**. The pooling layer actually **downsamples**. If you look at that, the size of that box got reduced in the pooling layer compared to the green and the blue box. So the **dimensionality** got reduced. That is the benefit of the pooling layer. We give the output to the convolutional layer where we are extracting all the important features. Then we pass that feature to a pooling layer where we downsize or **downsample**. The benefit of doing so is that we don't have to deal with as much computational processing because we don't have to deal with a larger size. We are reducing the dimensionality. That is the importance of the pooling layer: you can downsample it or downsize. Then this output is given to the **fully connected layer**. This is where the magic is happening. This is where we are training the model, adjusting the connections and **weights** until we get the desired output. The output layer will show, for example, three purple dots which represent three **classes**, maybe class dog, class cat, or whatever. The neurons in the output layer represent the number of classes if it is a **classification** problem.

Also, if you notice, there is a section marked for extracting features. This is where the **feature extraction** and learning process is happening. And the fully connected layer is where the final **classification** is happening.

The CNN typically consists of an input layer and multiple hidden layers. It's not necessary that we have just one convolutional layer and one pooling layer. We can have multiple convolutional layers and multiple pooling layers depending upon how complex the data set is. If you have a very large data set, you can have multiple layers. We are using already implemented algorithms and most of them use multiple layers of convolution. You will see that when you implement in your lab and project.

### Key Components of CNN

> This is what you will be asked to draw for your exam.

The hidden layer includes a series of convolutional layers, pooling layers, and the fully connected layer, and each layer performs distinct operations.

| Layer | Function |
|---|---|
| **Convolutional Layer** | Extracts spatial features from the input image |
| **Pooling Layer** | Reduces spatial dimensions and simplifies computation |
| **Fully Connected Layer** | Integrates features for final classification |

The **convolutional layer** applies the **convolutional operation**, and when we apply the convolutional operation, it extracts only the important features. The **pooling layer** performs the **downsampling** and reduces the size. The **fully connected layer** uses all the features to compute the **class scores**. Class scores means in the output layer you see, say, three nodes. Each represents a different class. For each class there are some class scores. Depending upon which class has the maximum score, that is the one the model is predicting.

---

## 1. Convolutional Layer (In Depth)

### Basic Idea of Convolution

In this video you can see there is a small **kernel** or **filter** which is looking for the shape of an eye. When we **convolve** that with the input image, we see the shape of the face. As that filter travels across the width and height of the image, it looks at each location to see if it is matching or not. The first two rows are not matching, then it comes and identifies two locations where the exact same shape is identified. In those locations, they assign some value or different weight to show that this information is needed. The rest of the pieces are not needed for further processing. This is the basic idea of convolution. We apply different types of filters. We will have, for example, a filter that can identify edges. It looks for that particular shape in the image and identifies it. Similarly, we will have any number of filters to be applied on an image so that we can extract all the important features, and then we will use this to train the model. Finally, it can give us class scores for each class.

### Features Extracted by Filters

As you can see, in these layers, small **learnable filters** slide over the input to extract features such as **edges**, **textures**, and **shapes**. When it passes through the images, it extracts shapes like it can be a circle, or even a shape like an eye, or just an edge. When we did the **Canny edge detection**, it just detected the edges. Similarly, it can extract features, and those features include edges, textures, and shapes. Each filter in a convolutional layer detects different features. We use any number of filters and all these filters can detect different features. No two filters are doing the same purpose. For example, if we are using an edge detection filter for horizontal edges, it just detects the horizontal edges, and we will use another filter for detecting vertical edges, and another one for just detecting the corners maybe. Each filter has its own purpose. We use any number of filters and all these filters will detect different features from the image. That's why we can say we can deal with complex structures because we are identifying those complex patterns by using different filters. The convolutional layer plays a crucial role in feature detection and representation. That's why we are able to use CNN to perform image classification or image recognition, because convolution is the basic principle behind this. It helps us to identify the features.

### Hierarchical Feature Extraction

As I also mentioned, it has the ability to extract features **hierarchically**. First, it can detect low level features, then mid level features, and high level features.

| Level | Examples |
|---|---|
| **Low level** | Small lines, patches, edges |
| **Mid level** | Circles, clear patches, basic shapes |
| **High level** | Wheels, windows, cars, houses |

If you look at the image of the car, it can detect **low level features** like small lines or some patches or edges. When you go to the **mid level features**, it can detect some shapes like circles or even more clear patches. Then you come to the **high level features**, and it can detect the shape of a wheel or even more precise shapes. In reality, if you go deeper in the network, it can detect more complex patterns like the shape of a house, shape of a window, or shape of a car. We have filters for that. So even when you go deeper in the network, it will extract more precise features. That's why they say that CNN can learn complex patterns. The basic principle of CNN is to automatically learn and extract **hierarchical features** from the input data.

### Feature Maps

On the left side, you can see the image of a cat. It's a coloured image, not a grayscale. The middle image is the output of a basic edge detector applied on the cat image. You can see that the edges of the cat's face are detected there. On the right hand side, you can see some of the feature maps that we got as a result of applying different filters. **A feature map is nothing but the output that we get in the convolutional layer.** The output from these layers is called feature maps. These feature maps are given to the pooling layer, and the output is called pooled feature maps. The right hand side shows four different feature maps that we get as an output by applying four different filters. Let's say one is detecting horizontal edges, one detecting vertical edges, one detecting maybe the sharp points, et cetera. These are the output of four different filters. They are similar but there are some differences in detected features.

### Why Convolutional Layer Helps

The convolutional layer helps the network focus only on the most important features. The disadvantage of ANN was that it would be hard to give all the pixels in the image into the input layer, making it computationally very heavy. The benefit of using the convolutional layer is that we don't have to use all the pixels. What we do is apply the convolution. We just pick the important features only. Then the network uses just these features. It's not taking all the pixels from the input image. It's just taking whatever we need for processing, whatever we need for making the decision. We are making the network more lightweight than ANN. Not all the pixel information in the image is relevant for training the model. We're just taking whatever we need. It improves the performance and accuracy by using this convolution operation.

### Convolution Operation (Math)

This is how the **convolution operation** is performed. On the left hand side, you can see a matrix which is the input image, and we have a **filter** or **kernel**. These words can be used interchangeably; both are the same: **kernel** and **filter**.

**Example: Convolution of a 6×6 input with a 3×3 filter**

Given a 6×6 input image and a 3×3 filter, the output is a 4×4 feature map.

```
Input Image (6×6):                  Filter (3×3):
┌───┬───┬───┬───┬───┬───┐          ┌───┬───┬───┐
│ 1 │ 0 │ 1 │ 0 │ 1 │ 0 │          │ 1 │ 0 │ 1 │
├───┼───┼───┼───┼───┼───┤          ├───┼───┼───┤
│ 0 │ 1 │ 0 │ 1 │ 0 │ 1 │          │ 0 │ 1 │ 0 │
├───┼───┼───┼───┼───┼───┤          ├───┼───┼───┤
│ 1 │ 0 │ 1 │ 0 │ 1 │ 0 │          │ 1 │ 0 │ 1 │
├───┼───┼───┼───┼───┼───┤          └───┴───┴───┘
│ 0 │ 1 │ 0 │ 1 │ 0 │ 1 │
├───┼───┼───┼───┼───┼───┤
│ 1 │ 0 │ 1 │ 0 │ 1 │ 0 │
├───┼───┼───┼───┼───┼───┤
│ 0 │ 1 │ 0 │ 1 │ 0 │ 1 │
└───┴───┴───┴───┴───┴───┘
```

**Step 1:** Place the filter over the top-left 3×3 region of the input:

```
Overlay:          Element-wise multiply:         Sum:
1×1  0×0  1×1     →  1 + 0 + 1                   
0×0  1×1  0×0     →  0 + 1 + 0                   = 1+0+1+0+1+0+1+0+1 = 5
1×1  0×0  1×1     →  1 + 0 + 1                   
```

Output[0,0] = **(1×1) + (0×0) + (1×1) + (0×0) + (1×1) + (0×0) + (1×1) + (0×0) + (1×1) = 5**

**Step 2:** Shift the filter one pixel to the right (stride = 1):

```
Overlay (columns 1-3):
0×1  1×0  0×1     →  0 + 0 + 0
1×0  0×1  1×0     →  0 + 0 + 0                   = 0
0×1  1×0  0×1     →  0 + 0 + 0
```

Output[0,1] = **0**

**Step 3:** Continue sliding across all positions...

```
Output Feature Map (4×4):
┌───┬───┬───┬───┐
│ 5 │ 0 │ 5 │ 0 │
├───┼───┼───┼───┤
│ 0 │ 5 │ 0 │ 5 │
├───┼───┼───┼───┤
│ 5 │ 0 │ 5 │ 0 │
├───┼───┼───┼───┤
│ 0 │ 5 │ 0 │ 5 │
└───┴───┴───┴───┘
```

> Notice: the size got reduced from 6×6 to 4×4. We just pick what we need; we don't want all the pixels in the input image.

You can expect on the exam to perform the convolution operation and get the output.

### Factors Affecting Convolutional Layer

There are several factors that help us to make our model better in the convolutional layer:

#### Factor 1: Filter Size

**Filter size** determines the extent of the input data that each filter covers. If you select a small filter, it can cover a small area in the input image. If you select a big filter, let's say 11×11, it can cover that much area from the input image. It affects the **granularity** of the features.

| Filter Size | Effect |
|---|---|
| **Small** (e.g., 3×3) | Captures fine details: small edges, corners |
| **Large** (e.g., 11×11) | Captures broader patterns: shapes of homes, trees |

#### Factor 2: Stride

**Stride** means the step size with which we are sliding across the input image. For example, we move one pixel to the right, which means stride is one. If we say stride equals two, we skip one column and move to the third one. That is what **stride** means: the step size with which the filter moves across the input image.

| Stride | Effect |
|---|---|
| **Small** (e.g., 1) | Output is larger, more overlapping positions |
| **Large** (e.g., 2) | Output is smaller, fewer overlapping positions, more abstracted features |

If we use bigger strides, the advantage is that we won't go through the repetitive pixels again and again. The overlap of adjacent filter positions can be avoided, and the size of the output feature map can be controlled by doing so.

#### Factor 3: Padding

**Padding** means you are adding a layer of **zeros** around the input. The advantage is, let's say that we have some important information at the corner pixel. If we are not using padding, when you do the convolution, we visit that corner pixel only once. But if you have padding, what happens is it can visit that corner pixel maybe 2 or 4 times. That's the reason why they use **padding**: if we have images which have more important information in the corners, we can use padding to protect that information.

**Example: 5×5 input with padding of 1**

```
Without Padding:             With Padding (P=1):
┌───┬───┬───┬───┬───┐       ┌───┬───┬───┬───┬───┬───┬───┐
│ a │ b │ c │ d │ e │       │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┼───┼───┤
│ f │ g │ h │ i │ j │       │ 0 │ a │ b │ c │ d │ e │ 0 │
├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┼───┼───┤
│ k │ l │ m │ n │ o │       │ 0 │ f │ g │ h │ i │ j │ 0 │
├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┼───┼───┤
│ p │ q │ r │ s │ t │       │ 0 │ k │ l │ m │ n │ o │ 0 │
├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┼───┼───┤
│ u │ v │ w │ x │ y │       │ 0 │ p │ q │ r │ s │ t │ 0 │
└───┴───┴───┴───┴───┘       ├───┼───┼───┼───┼───┼───┼───┤
                            │ 0 │ u │ v │ w │ x │ y │ 0 │
                            ├───┼───┼───┼───┼───┼───┼───┤
                            │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
                            └───┴───┴───┴───┴───┴───┴───┘
```

Padding helps in preserving edge information and allowing deeper layers to extract increasingly complex and abstract features. If you don't want to reduce the size and you need all the information from the input pixels, then you can apply padding so that you have control over the size.

> The **output** from the convolutional layer is called a **feature map**. When you pass this feature map to the pooling layer, the output from the pooling layer is called a **pooled feature map**.

### Output Size Formula

> You can expect questions from this on the exam.

$$Output\ Size = \frac{N - F + 2P}{S} + 1$$

Where:
- **N** = size of input image
- **F** = size of filter
- **S** = stride
- **P** = padding

**Example 1: With padding and stride**

| Parameter | Value |
|---|---|
| Input size (N) | 5×5 |
| Filter size (F) | 3×3 |
| Padding (P) | 1 |
| Stride (S) | 2 |

```
Output = (N - F + 2P) / S + 1
       = (5 - 3 + 2×1) / 2 + 1
       = (5 - 3 + 2) / 2 + 1
       = 4 / 2 + 1
       = 2 + 1
       = 3

Output size: 3×3
```

> When you are asked to compute the size of the output, you don't just say 3. You have to specify **3×3**.

**Example 2: Zero padding, stride of 1 (simplified formula)**

When P = 0 and S = 1:

$$Output\ Size = N - F + 1$$

```
If N = 6, F = 3:
Output = 6 - 3 + 1 = 4

Output size: 4×4
```

This matches our earlier convolution example where a 6×6 input with a 3×3 filter produced a 4×4 output.

**Example 3: Additional practice**

| Parameter | Value |
|---|---|
| Input size (N) | 32×32 |
| Filter size (F) | 5×5 |
| Padding (P) | 0 |
| Stride (S) | 1 |

```
Output = (32 - 5 + 2×0) / 1 + 1
       = (32 - 5 + 0) / 1 + 1
       = 27 / 1 + 1
       = 27 + 1
       = 28

Output size: 28×28
```

---

## 2. Pooling Layer (In Depth)

The output from the convolutional layer is given to the **pooling layer**. The pooling layer is useful for reducing the size, for **downsampling** or downsizing. The pooling layer is responsible for reducing the **spatial size** of the **feature maps** generated by the convolutional layer.

**Benefits of pooling:**
- Reduces spatial dimensions (downsampling)
- System becomes more tolerant to variations and distortions
- Enhances the ability to generalize
- Computational cost is reduced

There are **two methods** for performing pooling:

### Max Pooling

In max pooling, under each block we take the **maximum value**.

**Example: Max Pooling with 2×2 kernel**

```
Feature Map (4×4):                  After Max Pooling (2×2):
┌───┬───┬───┬───┐                   ┌───┬───┐
│ 6 │ 2 │ 7 │ 5 │                   │ 6 │ 7 │
├───┼───┼───┼───┤        →          ├───┼───┤
│ 5 │ 1 │ 3 │ 4 │                   │ 8 │ 5 │
├───┼───┼───┼───┤                   └───┴───┘
│ 8 │ 3 │ 2 │ 1 │
├───┼───┼───┼───┤
│ 4 │ 0 │ 5 │ 3 │
└───┴───┴───┴───┘

Calculation:
- Orange block [6,2,5,1]:  max = 6
- Yellow block [7,5,3,4]:  max = 7
- Green block  [8,3,4,0]:  max = 8
- Blue block   [2,1,5,3]:  max = 5
```

We get the maximum prominent feature from those blocks.

### Average Pooling

In average pooling, we take the **average** of each block: sum the values and divide by the number of elements.

**Example: Average Pooling with 2×2 kernel (same input)**

```
Feature Map (4×4):                  After Avg Pooling (2×2):
┌───┬───┬───┬───┐                   ┌──────┬──────┐
│ 6 │ 2 │ 7 │ 5 │                   │ 3.50 │ 4.75 │
├───┼───┼───┼───┤        →          ├──────┼──────┤
│ 5 │ 1 │ 3 │ 4 │                   │ 3.75 │ 2.75 │
├───┼───┼───┼───┤                   └──────┴──────┘
│ 8 │ 3 │ 2 │ 1 │
├───┼───┼───┼───┤
│ 4 │ 0 │ 5 │ 3 │
└───┴───┴───┴───┘

Calculation:
- Orange block [6,2,5,1]:  avg = (6+2+5+1)/4 = 14/4 = 3.50
- Yellow block [7,5,3,4]:  avg = (7+5+3+4)/4 = 19/4 = 4.75
- Green block  [8,3,4,0]:  avg = (8+3+4+0)/4 = 15/4 = 3.75
- Blue block   [2,1,5,3]:  avg = (2+1+5+3)/4 = 11/4 = 2.75
```

> Notice in both cases the size got reduced from **4×4 to 2×2** (halved).

---

## 3. Fully Connected Layer (In Depth)

This is where all the magic is going to happen for training our model. We get the output from the model based on the computations that happen in this layer. This is one of the most important layers.

### Flattening

The output from the convolutional and pooling layers is called a **feature map**. Before feeding it to the **fully connected layer**, we need to **flatten** it.

If you remember the CNN architecture, there is a layer called **flatten**. What it does is take the output from the pooling layer, which will be a **multidimensional** array or matrix, and convert that into a **one dimensional vector**. That is called **flattening**. We are just flattening it from multidimensional to one dimensional.

**Example: Flattening a 3×3 feature map**

```
Feature Map (3×3):              Flattened Vector (1×9):
┌───┬───┬───┐
│ 1 │ 1 │ 0 │   Row 1 →        [1, 1, 0, 4, 2, 1, 0, 2, 1]
├───┼───┼───┤
│ 4 │ 2 │ 1 │   Row 2 →
├───┼───┼───┤
│ 0 │ 2 │ 1 │   Row 3 →
└───┴───┴───┘

Process: Take Row 1 → append Row 2 → append Row 3
```

It concatenates all elements along the depth. This enables feeding into the fully connected layer directly.

### Fully Connected Layer Structure

The feature map is flattened and then given as input to the fully connected layer. In the fully connected layer, there are multiple layers. We train in the fully connected layer by adjusting the weights and other layer parameters, and finally we have an output layer which represents the number of classes. The number of neurons in the output layer represents the number of classes. If it is a binary classification, we will have two neurons. If there are, say, ten classes, we will have ten neurons in the output layer.

The fully connected layer is the layer where the **high level reasoning**, based on the features that we extracted from the convolutional layer and pooling layer, is applied to make the decision. We convert that high dimensional feature map into a **probability distribution**. If it is a binary classification, we will have class A and class B as output. Our model might predict, for example, class A has probability 0.8 and class B has 0.2. The model says the object we gave to the model belongs to class A because it has the higher score. Depending upon which class has the higher value, we say that the object belongs to that particular class.

Each neuron in the fully connected layer is connected to each neuron in the previous layer. So we have full connections between the neurons in consecutive layers. The main idea is that if you have connections from all the neurons to all the nodes in the next layer, it can represent the entire representation of the input and we are not missing any of the important data when making decisions.

---

## Weight Matrices and Biases

Another important factor in the convolutional neural network is **weight matrices** and **biases**. This is the way for all **deep learning** techniques. **Weight matrices** and **bias vectors** are the key players in the system. They are the ones which we are actually training. These values are not constant. We start with some **random value**, then we get an output, and we say this is not the output we are expecting. So we need to train them again. We go back and update the values and get another output. We see if this is better than the previous one. We continue trying until we get the best or most optimal results.

**Weight matrices and bias vectors are the learnable parameters** in the network. Over the period of time, they can learn from data and they can change their values.

### Weight Matrix Dimensions

**Example: Computing weight matrix and bias vector sizes**

```
Flattening vector (input):     X₁, X₂, X₃, X₄     →  4 values (M = 4)
Current layer neurons:         A₁, A₂, A₃           →  3 neurons (N = 3)

Weight Matrix size: N × M = 3 × 4
Bias Vector length: N = 3
```

| Component | Size | Determined by |
|---|---|---|
| **Weight Matrix** | N × M (3×4) | N = neurons in current layer, M = inputs from previous layer |
| **Bias Vector** | N (3) | Number of neurons in current layer |

### Matrix Multiplication Example

```
     Weight Matrix (3×4)          Input (4×1)     Bias (3×1)
┌─────┬─────┬─────┬─────┐      ┌─────┐          ┌─────┐
│ W₁₁ │ W₁₂ │ W₁₃ │ W₁₄ │      │ X₁  │          │ B₁  │
├─────┼─────┼─────┼─────┤   ×  ├─────┤    +     ├─────┤
│ W₂₁ │ W₂₂ │ W₂₃ │ W₂₄ │      │ X₂  │          │ B₂  │
├─────┼─────┼─────┼─────┤      ├─────┤          ├─────┤
│ W₃₁ │ W₃₂ │ W₃₃ │ W₃₄ │      │ X₃  │          │ B₃  │
└─────┴─────┴─────┴─────┘      ├─────┤          └─────┘
                               │ X₄  │
                               └─────┘

Computing each neuron output:

A₁ = (W₁₁ × X₁) + (W₁₂ × X₂) + (W₁₃ × X₃) + (W₁₄ × X₄) + B₁
A₂ = (W₂₁ × X₁) + (W₂₂ × X₂) + (W₂₃ × X₃) + (W₂₄ × X₄) + B₂
A₃ = (W₃₁ × X₁) + (W₃₂ × X₂) + (W₃₃ × X₃) + (W₃₄ × X₄) + B₃
```

**Numerical Example:**

```
Suppose:
X = [2, 1, 3, 0]       (flattening vector)
B = [1, 0, -1]          (bias vector)

Weight Matrix:
┌────┬────┬────┬────┐
│ 0.5│ 0.3│-0.2│ 0.1│
├────┼────┼────┼────┤
│ 0.1│-0.4│ 0.6│ 0.2│
├────┼────┼────┼────┤
│-0.3│ 0.7│ 0.1│-0.5│
└────┴────┴────┴────┘

A₁ = (0.5×2) + (0.3×1) + (-0.2×3) + (0.1×0) + 1
   = 1.0 + 0.3 + (-0.6) + 0 + 1
   = 1.7

A₂ = (0.1×2) + (-0.4×1) + (0.6×3) + (0.2×0) + 0
   = 0.2 + (-0.4) + 1.8 + 0 + 0
   = 1.6

A₃ = (-0.3×2) + (0.7×1) + (0.1×3) + (-0.5×0) + (-1)
   = (-0.6) + 0.7 + 0.3 + 0 + (-1)
   = -0.6

Output: A = [1.7, 1.6, -0.6]
```

### Why Nonlinearity?

The reason why we use weight and bias vectors is that we need some sort of nonlinearity. If we have a linear equation, for example f(x) = 2x + 1, this is just a linear equation. It would be a one to one connection or straight connection. If we give 1, it will be 3. If we give 2, it will be 5. This is a simple system. It cannot deal with complex patterns. For that, we need some sort of nonlinearity, like curves, sigmoid shapes, or bell shapes. All these represent some complexity. We don't want straight lines or straight output. We need some sort of nonlinearity in our system; then only can it handle more complex patterns. That's why we need nonlinearity. For this, we use another concept called **activation functions**.

---

## Activation Functions

**Activation function** is the function that determines if a **neuron fires**, or if a neuron is needed for the next level. For example, in the previous example, we had output A1, A2, A3. But sometimes we don't need to keep all of them for the next level. We just need maybe A1 and A3. How we make that decision is based on the **activation function**. It identifies if this neuron is qualified enough to go to the next level or if we need to keep it. And this can be applied at different layers. It is applied after the convolutional layer, after the fully connected layer, and even after the output layer. The most commonly used one is **ReLU**.

### ReLU (Rectified Linear Unit)

$$f(x) = max(0, x)$$

- If the input value is **greater than zero**, we take that value
- If it is **less than or equal to zero**, we make it zero

We take only the positive values and disregard the negative values. This is how we make a decision about whether a neuron can be fired or not.

**Example: Applying ReLU to our previous neuron outputs**

```
Before ReLU:   A = [1.7, 1.6, -0.6]

ReLU applied:
  f(1.7)  = max(0, 1.7)  = 1.7   ✓ (passes through)
  f(1.6)  = max(0, 1.6)  = 1.6   ✓ (passes through)
  f(-0.6) = max(0, -0.6) = 0.0   ✗ (zeroed out)

After ReLU:    A = [1.7, 1.6, 0.0]
```

> Notice: The negative value (-0.6) was replaced with 0. This neuron is effectively "not fired."

---

## Output Layer

This is the final layer where we generate the **prediction** or **classification**. The number of **neurons** in the last layer matches the number of **classes**. If it is a **binary classification**, how many neurons will it have? Two, because it matches the number of classes. If there are 10 classes, we would have 10 neurons in the **output layer**. It matches the number of classes.

The **activation function** differs from the fully connected layer. For classification, we use something called **softmax**. We will learn about the details in upcoming classes. The **highest probability neuron** represents the **prediction**. We will have different classes and at the end what we get are some **class scores**, let's say 0.5, 0.3, and 0.2. We take the one with the highest class score.

**Example: Output layer class scores**

```
Class Scores:
┌──────────────┬───────┐
│ Class        │ Score │
├──────────────┼───────┤
│ Cat          │ 0.2   │
│ Dog          │ 0.3   │
│ Bird         │ 0.5   │  ← Highest score = Prediction
└──────────────┴───────┘

Prediction: Bird (score = 0.5)
```

We have input pixels from the flattening matrix. The blue colour represents the flattening layer. We give that to the fully connected layer. We have multiple fully connected layers and we do all the training there by adjusting the weights and biases. Finally, we get the classes with class scores. The one with the highest class score is the prediction.

---

## Back Propagation

We are not going into depth today. We have another class to explain how **back propagation** works. But just an idea: it is a **supervised learning** algorithm which is used for training neural networks. It optimizes the parameters. What are the parameters? The two **learnable parameters**: **weights** and **biases**. We optimize these parameters by **minimizing the error** between the **predicted output** and the **actual target value**. That's why they say this is **supervised learning**, because we know the input image is a cat's image and we are expecting that the model should give a good score for the class cat. But sometimes our training is not so good and it gives us the output dog for a cat. Then we say this is not the output we are expecting. So we again adjust the weights.

### Error Calculation

By minimizing the error, how we calculate the error is that we know what the expected value should be and we subtract what we got from the expected value. Then we get the error.

**Example: Error reduction over iterations**

| Iteration | Predicted | Expected | Error |
|---|---|---|---|
| 1 | Dog (0.7) | Cat (1.0) | 5.0 |
| 2 | Cat (0.5) | Cat (1.0) | 3.0 |
| 3 | Cat (0.8) | Cat (1.0) | 1.0 |
| 4 | Cat (0.95) | Cat (1.0) | 0.2 |

Our idea is that after the first iteration, we calculate this error. Each iteration, the error should get smaller until we reach the smallest error.

### How Back Propagation Works

The basic step is: we pass the sample to the network, then we calculate the **mean squared error (MSE)** depending upon the output we get. Then we calculate the **error terms**. It starts from the last layer of the network. Then it **propagates back** to the layer before the output layer, and then it goes further back.

```
Forward Pass:
Input → Conv Layer → Pooling → FC Layer → Output → Error

Back Propagation:
Error → Output Layer → FC Layer (adjust weights) → Previous layers
        ←──────── propagate error backwards ────────→
```

So if you have three layers, the error is found at the output layer, then it is propagated back to the previous layer, and from this it propagates back further. That's why it is called **back propagation**. We adjust the weights from the output layer to the previous layer, going back through however many layers we have in the fully connected layer. We calculate the error terms in the hidden layers, apply the **gradient descent**, and adjust the weights accordingly. After training, we get some **delta value**, we add that to the weight, and that's how we adjust the weights. We will learn about back propagation in detail in the upcoming classes. Just as an idea, we use **weight matrices**, **biases**, and an algorithm called **back propagation**. All these are important concepts in CNN.

---

## Image Processing in CNN (End-to-End Example)

```
Input         Conv Layer 1    Pool Layer 1    Conv Layer 2    Pool Layer 2    Flatten       FC Layer    Output
(28×28)  →  32 @ 28×28    →  32 @ 14×14   →  64 @ 14×14   →  64 @ 7×7    →  3136×128  →    128    →   10
                                                                                                      (0-9)
             \_______________ Feature Extraction _______________/             \______ Classification ______/
```

We have the input image, let's say of size 28×28. We pass it to the **convolutional layer** where we apply 32 **filters**, giving us 32 **feature maps** of size 28×28. Then we pass it to the **pooling layer** which **downsamples** it to 32 feature maps of size 14×14. The dimension got halved from 28 to 14.

As mentioned, we can have **multiple convolutional and pooling layers**. In this example, we have two sets. After the first batch, the output of the first pooling layer goes to another convolutional layer where we apply 64 filters (64 feature maps of 14×14). Then again we apply pooling (64 feature maps of 7×7). We **flatten** it into a **one dimensional vector** (3136 by 128). Then we have one **fully connected layer**, and finally the **output layer** with 10 neurons representing 10 classes, including zero through nine.

These are the layers where we do the feature extraction, and the fully connected and output layers are where the classification happens. So: input, then feature extraction, we downsample, and finally we classify.

---

## Applications of CNN

Wherever we use machine vision, we can use CNN.

**Tasks:**
- Image classification
- Object detection
- Semantic segmentation
- Multi object tracking
- Re-identification
- Image generation

**Real World Applications:**
- **Healthcare:** Anomaly detection in medical scans
- **Autonomous vehicles:** Real time environment perception, obstacle detection
- **Facial recognition:** Smartphones
- **Security:** Surveillance in airports
- **Manufacturing:** Quality control and defect detection
- **Education**

---

## Performance Evaluation Metrics

We need metrics to evaluate how our models behave. Is it a good model or a bad model? We need to decide if our network or model is mature enough to give accurate results.

### Types of Metrics by Model Type

| Model Type | Metric Type | Examples |
|---|---|---|
| **Clustering** | Distance-based | Average distance to cluster centres, maximum distance |
| **Regression** | Error-based | Mean Absolute Error, Mean Squared Error, Root Mean Squared Error |
| **Classification** | Performance-based | Accuracy, Precision, Recall, F1 Score, AUC |

> **Rule of thumb:** Anything related to **distance** → clustering. Anything related to **error** → regression.

### Classification Metrics (What We Need to Learn)

#### Accuracy

The proportion of **total predictions** (both positive and negative) that the model got correct.

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

#### Precision

Assesses the accuracy of the **positive predictions** only.

$$Precision = \frac{TP}{TP + FP}$$

#### Recall

Measures the CNN's ability to correctly identify **all actual positive cases**.

$$Recall = \frac{TP}{TP + FN}$$

#### F1 Score

Provides a balance between precision and recall by calculating their **harmonic mean**.

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

#### ROC Curve & AUC

**ROC** (Receiver Operating Characteristic) plots the **true positive rate** against the **false positive rate**. The **area under that curve (AUC)** provides a single value summarizing the overall performance of the CNN across all possible classification thresholds.

```
True Positive Rate
     ↑
  1  ┤         ╭──── Best Model
     │       ╭─╯
     │     ╭─╯
     │   ╭─╯
     │  ╱        ╱ Random (diagonal)
     │ ╱        ╱
     │╱        ╱
  0  ┼──────┼──────→ False Positive Rate
     0             1

- Above diagonal → model is doing well
- On diagonal → random guessing
- Below diagonal → worse than random
- The more the curve goes towards the top left → the better
```

### Confusion Matrix

This is a tool used in machine learning to evaluate the performance of a classification model. The confusion matrix shows the comparison of **actual versus predicted** values.

**Example: Heart Disease Prediction**

```
                       Predicted
                    No         Yes
              ┌──────────┬──────────┐
  Actual  No  │   28000  │   ...    │
              │    (TN)  │   (FP)   │
              ├──────────┼──────────┤
         Yes  │   ...    │   ...    │
              │    (FN)  │   (TP)   │
              └──────────┴──────────┘
```

| Cell | Meaning |
|---|---|
| **True Negative (TN)** | Actual = No, Predicted = No (correct) |
| **False Positive (FP)** | Actual = No, Predicted = Yes (incorrect) |
| **False Negative (FN)** | Actual = Yes, Predicted = No (incorrect) |
| **True Positive (TP)** | Actual = Yes, Predicted = Yes (correct) |

**Numerical Example:**

```
                       Predicted
                    No         Yes
              ┌──────────┬──────────┐
  Actual  No  │    50    │    10    │   (TN=50, FP=10)
              ├──────────┼──────────┤
         Yes  │     5    │    35    │   (FN=5,  TP=35)
              └──────────┴──────────┘

Accuracy  = (50+35) / (50+10+5+35) = 85/100 = 0.85 = 85%
Precision = 35 / (35+10)           = 35/45  = 0.78 = 78%
Recall    = 35 / (35+5)            = 35/40  = 0.88 = 88%
F1 Score  = 2 × (0.78 × 0.88) / (0.78 + 0.88)
          = 2 × 0.686 / 1.66
          = 1.373 / 1.66
          = 0.83 = 83%
```

---

## Ethical Considerations in CNN

We are dealing with data that can have a lot of personal or private, highly secure data. We should be mindful about these factors:

1. **Privacy** — Privacy concerns arise when dealing with sensitive personal data. Sometimes we need to train on personal data like date of birth, age, or security numbers. If you train a model on such data, you should be mindful about privacy.

2. **Surveillance** — Using CCTV is again using machine vision, but we may be risking the privacy of others. Sometimes, even without the consent of people, we are taking video of them. Surveillance is also a privacy concern.

3. **Bias in AI** — Depending upon what data you are feeding the network, it will make the prediction. Never have a biased data set. Always have a diverse data set. Do not have a bias towards a particular stereotype or particular political interest.

> Whatever you are making should be good for the society, nothing that can harm or mislead people.

---

## Next Week

CNN training processes: **loss function**, **activation function**, **back propagation**, and **optimizers**.
