# Lecture 2: Image Processing

## Topics
1. What is Image Processing?
2. Why Do We Need Image Processing? (Enhancement, Feature Extraction, Segmentation, Object Recognition, Measurement)
3. Key Stages in Image Processing (9 Stages)
4. Filtering and Convolution
5. Blurring (Averaging)
6. Sharpening
7. Resizing and Scaling
8. Edge Detection (Canny Algorithm)
9. Histograms and Bins
10. Thresholding (Simple, Adaptive, Otsu's)
11. Morphological Operations (Erosion, Dilation, Opening, Closing)
12. Image Transformation Techniques (Affine, Translation, Rotation, Scaling, Shearing)

## What is Image Processing?

Without image processing, we cannot achieve any of the desired results. So today we will cover what image processing is, what the importance of image processing is, and what are the different steps involved in image processing. We will see different image processing techniques like filtering, sharpening, and finally edge detection. You have already done edge detection for your lab, but we will look at whatever steps are involved in it. Then we will also see histograms, thresholding, morphological operations, and finally image transformation techniques. So this is our agenda today.

As I mentioned, **image processing** is the core of **machine vision**. So what do you do? You have seen in your lab, you have an input image, you change it into a different **colour space**, you convert it to **greyscale**. So basically you are doing something to transform the image into a different form. You are making it more useful for your applications. For example, you want to do some transformation so that you get more information from that image, to do some operation. So that is what image processing is doing: manipulation, and by manipulating it, you are analyzing that image. So from that, if you are detecting it, if you do **edge detection**, you can see what are the clear shapes of its ears and eyes and all those features, right? So these are also helpful for training our model to identify a cat. So that is the importance of image processing, and it is also used for enhancing the quality of the input image. So you capture an image and it has some light variation, it is not that clear. Then it is very difficult for our system to process it, right? So we need to do some **preprocessing** of that image to make it more clear so that the system can better understand that picture. So it can also be used for enhancing the quality of the image. And by doing so, we can extract meaningful information from it. So that is what image processing is doing.

## Why Do We Need Image Processing?

So why do we need it? Different reasons. I am just highlighting some of them.

### 1. Enhancement

The first one is, as I mentioned, enhancement. When you have a large dataset of images, because machine vision deals with images or video, right? So we are dealing with visual data, not documents or text or details. We are mostly dealing with visual data. Our dataset comprises a large number of images or maybe videos. Sometimes each image in the data may not be good enough to do the processing or training of our model. So what we need to do is enhance the quality of that input image. For that, we can use image processing. We can enhance the quality by reducing the **noise**, like you did with **blurring**. By doing the blurring, a little bit of the noise is reduced. So that is one example. And you can enhance the **contrast**. If you do **sharpening**, it gives more details. It highlights the edges and those kind of features from the images. So by doing so, it is easier for the system to analyze what is in that image. So that is the first point, enhancement.

### 2. Feature Extraction

The second one is **feature extraction**. It can be used for identifying and extracting important features like **edges**. You have seen in edge detection, you are extracting the edges from the image, right? So this can be used for identifying edges, **corners**, or **blobs** from the image. And what we need these edges and corners for is training the model to analyze the shape of the object, or to recognize an object, or do some **classification** based on what is the object in that image. So feature extraction is another reason why we need image processing in machine vision.

### 3. Segmentation

Next is **segmentation**. This is one of the toughest tasks. Segmentation is, as its name indicates, we divide or segment the image into smaller regions. At each region, we have some useful information for us. So we need to segment it carefully. This is another reason why we use image processing. Image processing is used for segmenting the image into meaningful regions, and this is a very important technique needed in **object detection** and **object classification**, et cetera.

### 4. Object Recognition

Then **object recognition**. It is clear from the name, we recognize objects. The machine vision techniques or the image processing techniques can be used for recognizing objects from the images. This is used in **automated inspection**, **robotics**, et cetera.

### 5. Measurement

And finally, the last one is **measurement**. As I mentioned last week, one of the applications of machine vision is in **assembly lines**. It can be used for **quality control**. If we have any product which is not meeting the quality, we can identify it. For example, a PCB board, if the components are not having enough distance as specified, it can detect it. So for this, we need to measure the distance between the components or the size of the board, stuff like that. Machine vision can help us achieve that. So that is the fifth point. It allows for precise measurement of object dimensions, distances, and other parameters. This is widely used in quality control and industrial automation. So these are the reasons why we need image processing in machine vision.

## Key Stages in Image Processing

Now, let us see what are the key stages in image processing.

1. **Acquisition** — we need to acquire the image or capture the image.
2. **Enhancement** — we can enhance the image.
3. **Restoration** — if anything has some defects like gaps or something, you can even restore that by performing some image processing techniques.
4. **Morphological Processing** — this is a processing technique which is based on shapes present in that image.
5. **Segmentation** — divides the image into different segments.
6. **Object Recognition** — can recognize and identify what is in the image, if it is a cat, or a dog, or a car, or something like that.
7. **Representation and Description** — a way of representing the data. If you have captured enough information, but what we captured is not enough for the computer to understand, we need to represent it in a way that the computer can understand. So that is what representation and description means.
8. **Image Compression** — needed when we have a large size of data but we have storage limitations, then we can compress the image.
9. **Colour Image Processing** — image processing based on colours.

So these are the different stages in image processing. You can go through it in detail. I have added asynchronous material for week two. It has details on the different stages. Keep in mind that we do not need all nine stages for our application. Depending upon your application, you can make a combination of two or three of them. Based on the application, a combination of two or three steps can be used. It is not mandatory that you have to use all the nine stages which I mentioned.

### Stage Details

**Image Acquisition:** We are acquiring images using cameras or sensors. The most commonly used ones are **CCD** and **CMOS** sensors, which are used in digital cameras.

**Image Enhancement:** This is where we manipulate our input image. We are polishing our image so that it can be used for our processing. If the image is not good enough to do our processing, we need to do enhancement by doing some sharpening, bringing out the details, or blurring to remove the noise from it. So depending upon the quality of your input image, you can perform image enhancement.

**Image Restoration:** It is again the process of improving the appearance of an image. For example, it can remove noise and restore the content. Also sometimes some missing gaps will be there, like bridge connections between two objects. Sometimes those are missing, and you can restore them by doing some image processing techniques.

**Morphological Processing:** This is again connected to shapes. An example application is **fingerprint recognition**. So these are the key stages.

**Image Segmentation:** As I mentioned earlier, this is one of the most difficult tasks, but this is the core technique for performing classification. Image segmentation is very important. As its name indicates, it divides the input image into different segments or regions.

**Object Recognition:** As I mentioned, can recognize what is present in the image and label it. It labels the object based on the information provided by the description.

**Representation and Description:** As I mentioned, is converting the data into a form that is more understandable for the computer. What a human can understand cannot be understood by the machine, right? So we need to represent it in a binary form or the form which the computer can understand. That is what the step of representation and description means.

**Image Compression:** As I mentioned, you all know image compression reduces the storage required to store that image.

**Colour Image Processing:** Involves the use of colour in the image to extract meaningful information.

So these are the different stages, and we have the asynchronous material section in your Brightspace. You can go through it for the details of each stage.

## Image Processing Techniques

### Filtering

Now let us take a look at different techniques. **Filtering** is one of the image processing techniques, and it is a very useful technique which is widely used in machine vision systems. So in the animation, you can see a filtering operation. We have an input image, a **kernel**, and an output image. I am not going to explain how the operation works in detail, maybe in a later week. This is basically a **convolution operation**. We will see in detail how to perform the convolution operation in the upcoming classes, but this is giving you a high level idea of what is happening with filtering.

When you apply a filter, we have an input image, and the input image is **convolved** with a **kernel** K. **Kernel and filter are the same**. Do not get confused with the terminology, they are the same. Filter size is the same as kernel size. We are convolving it with a kernel and we get an output.

**Convolution output size example:**

> Input image: `6×6`, Kernel: `3×3`, Output: `4×4`
>
> Output dimension = (Input dimension − Kernel dimension) + 1
>
> (6 − 3) + 1 = **4**

So did you notice that the input image has a size of six by six, the kernel is three by three, and the output is four by four? Which means we take what is needed for our processing. We do not want all the pixels in the image, we just need what is important for us. So this convolution operation can help us to pick only the important information and it also helps us to reduce the size. From six dimensional to four dimensional, it reduces the size and also gives us important information. So it is a very useful operation, the convolution operation. We will learn about how to perform this operation in upcoming classes.

As you can see in the video, we have the input image and the kernel, and we are passing the kernel from left to right across the width, and then we go along the height. It traverses across all the pixels in the image. It touches all the pixels in the image, picks what is needed, and puts it in the output image. So it is an image processing technique which is used to enhance an image by altering its pixels. That is what it exactly does: it goes through each pixel in the image, performs some computation, and alters that pixel into a different form which is useful for us. By doing so, we can amplify certain features like borders or anything, depending upon the filter that we choose. And you can also suppress unwanted noise from the image as well. So this is useful for both highlighting features and for suppressing noise.

It acts like a sieve through which the original image passes. When the image is passing through the sieve, it can highlight specific attributes and it can remove all the noise from it. By doing so, we get an output image which is good enough for us to do further analysis. Our input was not so good to do the analysis, so we did some polishing using the filtering technique, and then we get a final output image which is useful for us to do the analysis. Basically we are doing some cleanup to make it more useful. So that is filtering.

### Blurring

Next one is **blurring**. It is also a filtering operation, but depending upon the kernel we use, it becomes blurring or sharpening. The basic idea is again the **convolution operation**. Here also you can see an input image and an output image. Below, you can see how a blurred image would look like. It is not so clear, right? It is just blurry. In other words, we can say it is a smoother image. Blurring is a type of filtering that helps us to smooth an image. By doing so, it can reduce both detail and noise. In some cases, this can be used as the first stage to reduce the noise. Then if you apply another filter, like sharpening, it can highlight the important features. You may wonder why we need to reduce the details. There are scenarios where we do not want much detail but we need to get rid of noise. This can be used as a first stage in most of the pipelines. For example, in edge detection, we do blurring to remove the noise, right? And even the details are also hidden, but we do not want all the details. We just want the edges from it. After that, we perform edge detection on top of the blurred image. So I am just saying that this is used for suppressing noise. That is the main idea. For removing the noise, we use blurring.

#### How Blurring Works (Averaging Example)

How it works is it works by averaging the pixels around a target pixel. For example, if we have an image which is like black and white, we want to smooth this image. What we do is we set up the target pixel. The middle one is the target pixel. We take the average of all the pixels and then replace it. So if we choose a three by three kernel size, we consider this three by three region. We take the average of all the pixel values here and replace it. So we take this target pixel and replace it with the average of all the pixels under this kernel.

**Example calculation with a 3×3 kernel:**

Consider a boundary in a binary image where the left side is black (0) and the right side is white (90):

```
 0   0   0
 0   0  90
 0  90  90
```

For the centre pixel (value = 0):

> Average = (0 + 0 + 0 + 0 + 0 + 90 + 0 + 90 + 90) / 9
>
> Average = 270 / 9 = **30**

But in the simpler case the lecturer walked through, where the region was all zeros:

```
 0   0   0
 0   0   0
 0   0   0
```

> Average = (0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0) / 9 = **0**

If you add all the zeros in this region, the average is zero. So it gets replaced by zero. What that means is that this pixel is now the representation of all the pixels around it. It is just representing all those surrounding pixels.

When you move the kernel to the next position, it comes here. Now we are processing the next set of pixels. We move our kernel one pixel to the right. Now we are processing this pixel. We take the average of all the pixels around it and replace it with that average value.

**Next position example:**

```
 0   0   0
 0   0   0
 0   0  90
```

> Average = (0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 90) / 9 = 90 / 9 = **10**

And when we move it one pixel to the right again, we continue with the same process.

You can see from zero to 10, there is a gradual change, right? From zero to kind of a transition. So where before there was a drastic change in pixel values, now there is a smoother transition. The intensity is distributed almost normally, not a drastic change. This is how we make our image blurry. Even the darker areas will look greyish. So that is the main idea behind blurring. This is used for softening the image, and the basic idea is to reduce the noise.

### Sharpening

Next one is **image sharpening**. It is exactly the opposite of what we did now. This is used actually for **enhancing the details**, not for suppressing or removing, but instead we will enhance the details present in the image. Again, it is the same operation, the convolution operation, similar to filtering. But depending upon the kernel, the output would change.

**Example sharpening kernel:**

```
 0  -1   0
-1   5  -1
 0  -1   0
```

Here if you look at the kernel, the values in the matrix show contrast. You can see that the sharpening kernels have drastic values: zero to minus one, minus one to zero, and when you go to the next layer, minus one to five, five to minus one. So it is not a smooth transition, it shows contrasting values. If you apply such a kernel on an image, what you get is it emphasizes the contrast in the adjacent pixels. So it can help you to highlight the edges or make the image look more detailed. That is the main idea of sharpening. It is the opposite of blurring. Blurring makes everything look normalized, with less contrast between the values. So even though there was contrast between zero and 90, we made it like 10, 20, so we normalized it. But here it is the opposite. Even if it is 0, 10, 20, we make it like 100 to minus 100 or values like that. It is trying to increase the contrast between neighbourhood pixels.

So why do we need sharpening? It can help us to accentuate the edges and details. If we look at an image, for example with buildings, we have the roof and the edges. We have a clear edge of that roof, right? This can be highlighted more clearly if you do sharpening. Even the corners and the design on top of it can be clearly seen if you do sharpening, so it enhances the details present in the image. This is highly useful in **medical imaging** or **precision manufacturing** because you cannot compromise on the detail of the image, we need more information in that case. So this is highly useful in medical imaging and precision manufacturing where the details are very critical.

So as I mentioned, it emphasizes the contrast between adjacent pixels to highlight the boundaries of objects within the image. It keeps more contrast between adjacent pixels, then you can easily identify corners and edges, et cetera. That is what image sharpening is doing.

### Code Examples

We can take a look at the different implementations. This is the convolution operation. So you already know how to read the image and display it, right? This is how we do the sharpening. I have already showed you the matrices that we used, like zero, minus one, zero. So those are just values showing contrast. This is the kernel that we define as a NumPy array, and then we use the `filter2D` function from OpenCV to apply that kernel. And blurring, you already saw how to do the blurring, like the Gaussian blur that you already did in the lab.

```python
import cv2
import numpy as np

# Sharpening kernel
kernel = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]])

# Apply sharpening
sharpened = cv2.filter2D(image, -1, kernel)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(image, (5, 5), 0)
```

In this image, the first one is the original. The second one is the sharpened one, it will be more clear if you open it on your laptop, but I will share it later. And the third one is the blurred image. It is not that distinguishable from here. But in the second image, you can see towards the top of the head or towards the ear portion, it is clearer in the second image, right? You can see that if you closely look at it on your laptop.

### Resizing and Scaling

And **resizing**. This is **scaling**. If you ask me what is the difference between scaling and resizing, both are the same, and we use the same function, `resize`. But the difference is that in **resizing**, you specifically say what dimensions you need, like 300 by 300 pixels width and height. You specify the required height and width of the output image. But in **scaling**, what you do is you just provide a ratio. You can say you need half of the width and double of the height. So it is a **scale factor**. That is what scaling is. And resizing is when you exactly specify the width and height.

So this is scaling, because we give the scale factors. And rotation, you already did that in your lab.

I showed you last week what `image.shape` will return. If you look at the lab, NumPy will return the shape. So how we get the height and width: `image.shape` will return, basically, if it is an RGB or coloured image, shape will return a tuple of three values which is height, width, and the number of channels.

```python
height, width, channels = image.shape  # For colour (RGB) images
# e.g. (200, 300, 3) means height=200, width=300, channels=3
```

## Edge Detection

### Canny Edge Detection

Now let us look at **edge detection**. You can see the input image and the output is just an outline around the object, right? And it can clearly detect the shape of the object by highlighting the edges. So that is why we need edge detection, to identify the objects in that image. **Canny** is one of the older and very powerful edge detection methods. It has a lot of mathematical operations going on under the hood. But we do not have to worry about it because OpenCV is doing all the operations and we just need to call the Canny function and pass our image and the other parameters as per the function definition. The **Canny filter** is an **edge detection algorithm**, and it is noted for its precision in detecting a wide range of edges. In the input image, you can clearly see it detected all the important edges, especially the edges that can identify the shape of the person, the camera, even the building in the background. It did not miss out on any important details present in the image.

### Steps of the Canny Algorithm

How this is achieved involves different stages:

#### 1. Noise Reduction

The first step for almost all image processing is some cleanup, right? The input data will not be good enough for our processing. So we need to make it good. The first thing we do is clean up by removing the noise. For reducing the noise, as I mentioned, what do we do? We blur, right? Blurring can help us to reduce the noise in the image. In Canny, they use **Gaussian blur**. They recommend using Gaussian blur to get better results because it uses a mathematical model. If you look at OpenCV and search for Canny, it will show you the equations and mathematical operations that are performed. So noise reduction is the first step.

#### 2. Gradient Calculation

For each pixel in the processed image, it will calculate the **gradient**, that is, the **pixel intensity** value and the **direction** along which the intensity changes. So it can have a **magnitude** and a **direction**. You do not have to worry a lot about the mathematics behind it, but just for your information, this is the second step, gradient calculation. It finds the intensity gradient and its direction at each pixel.

The gradient, how it works, is like if you have an edge like this, it is always perpendicular to the edge. If this is the edge, the gradient direction is perpendicular to it. The direction of the gradient at the pixels will be perpendicular to the direction of the edge.

Then it smooths the image with another kernel, which is a **Sobel kernel**, which is for removing extra noise and adding more clarity. This is what happens if you apply the **Sobel operator**: it computes the **first derivative** in both horizontal and vertical directions, and this helps us to compute the magnitude and direction of the gradient at each pixel.

**Sobel operators:**

```
Gx (horizontal):        Gy (vertical):
-1   0  +1              -1  -2  -1
-2   0  +2               0   0   0
-1   0  +1              +1  +2  +1
```

**Gradient magnitude and direction:**

> Magnitude: G = √(Gx² + Gy²)
>
> Direction: θ = arctan(Gy / Gx)

#### 3. Non Maximum Suppression

The next step is **non maximum suppression**. This is an important step where we thin out our edges by suppressing the non maximum gradient values. We get a lot of values, but we do not need everything. We just need what is an edge, right? How it works is it performs a **local maxima suppression**. It compares with the neighbouring pixels. If this is the maximum pixel value, if it is the **local maximum**, then we will retain it. If it is not the maximum, we replace it or just ignore it.

#### 4. Double Thresholding

**Double thresholding** is another step. We have two thresholds for Canny, which are **T1** and **T2**.

- If pixel gradient value **< T1** → **Ignore** (not an edge)
- If pixel gradient value **> T2** → **Strong edge** (definitely an edge)
- If pixel gradient value **between T1 and T2** → **Weak edge** (potential edge)

**Example with T1 = 100, T2 = 200:**

> Pixel gradient = 50 → 50 < 100 → **Suppressed** (ignored)
>
> Pixel gradient = 250 → 250 > 200 → **Strong edge** (kept)
>
> Pixel gradient = 150 → 100 ≤ 150 ≤ 200 → **Weak edge** (check connectivity)

#### 5. Edge Tracking by Hysteresis

This is an algorithm that tracks edges by connecting **weak edges** to **strong edges**. It checks if any of the weak edges is connected to any strong edge. If there is a connection, then we keep that weak edge. Otherwise, we just ignore it.

So basically, if you look at the diagram, A is a point on the edge, and C and B are different points which are also along the direction of the gradient. What we do is check if A is the local maximum. We compare the value of A with B and C, because they are the neighbouring ones which are also along the direction of the gradient. If A is the local maximum, we say good, A is strong, we keep it. Next we look at B and C. Both are along the gradient direction. We check if they come under strong, weak, or below the threshold.

If we draw a graph, T1 is the minimum threshold and T2 is the maximum. A is of course above T2. C is also above. Let us say B is below the threshold T1. Then we will not keep B if it does not have any connection with A. We just discard B. But C is also a weak or strong point and it has a connection with A, so we select both A and C. We just ignore B because B is not connected to A even though it appears in the gradient. A is a strong edge. So that is how edge tracking by hysteresis works.

### Summary of Canny Steps

So these are the main steps. Just keep in mind what the steps in the **Canny edge detection** are, but you do not have to worry about how to calculate the intensity direction or how to calculate the gradient angle or anything. Just make sure that you understand how **double thresholding** works and why we need two thresholds. In your lab, you pass like 100 and 200 as thresholds, but you may not know what it is doing, right? So that is what this is showing. Your thresholds are used for selecting the strong edges and the weak edges connected to the strong edges, and just ignoring anything below the first threshold. I have a slide which you can go through for all the details.

```python
edges = cv2.Canny(image, threshold1=100, threshold2=200)
```

## Histograms

Now let us talk about **histograms**. A **histogram** is a graph that shows how many pixels in the image have a particular **brightness level**. For example, maybe 100 pixels in the image have a red colour value and 200 pixels have a blue colour value. So it is just a graph that shows how many pixels in the image have a particular brightness level. The x axis shows the different brightness levels and the y axis shows the number of pixels which belong to each brightness level. The x axis shows different brightness levels from dark to light. The leftmost value is the darkest one, and the rightmost would be the brightest one. You can see from zero to 255, right? Zero means black. 255 is white. So from the left it is black, and when you move further towards the right, it becomes brighter.

So it helps us understand if an image is mostly bright, dark, or balanced. This is useful for improving the image quality. If you have a histogram and there are more bars on the left side, that means the image is darker because the left represents darker values. On the other hand, if it is showing more values on the right, you can see that it is a brighter image. So that is the idea of histograms. It gives an overall idea of the **intensity distribution** of an image, how many pixels in the image have a particular **intensity level**. It has intensity values ranging from 0 to 255 on the x axis and corresponding number of pixels on the y axis. The left region shows the amount of darker pixels and towards the right side is the brighter.

### Histogram Bins

We can also do one thing. Since we have values from 0 to 255, we have 256 values. When we plot this, it will be like a huge graph with 256 points to plot. But if we make it into smaller **bins** or smaller segments, we can make it smaller and easier to process. That is why we have the concept of **bins**.

**Example with 16 bins (256 values ÷ 16 = 16 values per bin):**

| Bin | Range       |
|-----|-------------|
| B1  | 0 to 15     |
| B2  | 16 to 31    |
| B3  | 32 to 47    |
| ... | ...         |
| B16 | 240 to 255  |

> A pixel with intensity value **18** → falls into **Bin 2** (range 16 to 31)
>
> A pixel with intensity value **6** → falls into **Bin 1** (range 0 to 15)

So depending upon the intensity value, the pixel goes into the corresponding bin. If you take a union of all these bins, B1, B2, B3, up to B16, you will get the full 0 to 255 range. So that is the concept of bins. You can read more about it in the histogram resources I have given in the link. You can use `calcHist` from OpenCV to calculate the histogram of the image, but for plotting, you should use Matplotlib.

```python
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

import matplotlib.pyplot as plt
plt.plot(hist)
plt.title('Histogram')
plt.xlabel('Intensity Value')
plt.ylabel('Number of Pixels')
plt.show()
```

## Thresholding

Now, **thresholding**. This is one of the main techniques that helps us perform **segmentation**. Do you remember I mentioned segmentation is the most difficult task, and in order to achieve segmentation, it is very much needed for **object classification** and **object detection**. So we cannot avoid segmentation. How we actually do segmentation: there are different techniques, and one of the techniques is thresholding. You can segment the image based on thresholding. In the input image, you can see different shades of grey, from white to black to different grey shades. But in the output thresholded image, what you see is just black and white. So you distinguish between **foreground** and **background**. Our focus was on that person. In order to extract that person's image, what we did is we applied thresholding.

### Thresholding Calculation

If you look at the histogram in the middle, we applied a threshold at 100. We take every pixel in the input image and we check if it is below or above 100.

**For each pixel (x, y) with value P(x,y):**

> If P(x,y) < threshold (100) → set to **0** (black)
>
> If P(x,y) ≥ threshold (100) → set to **255** (white)

**Example:**

> Pixel value = 75 → 75 < 100 → Output = **0** (black)
>
> Pixel value = 180 → 180 ≥ 100 → Output = **255** (white)

So this is how we do the thresholding, and as a result, what we get is a black and white image which clearly gives us a distinction between the foreground and the background. Thresholding is simple but it is an effective way to perform segmentation. By converting an image to black and white based on a threshold value, we can isolate objects or features easily. From the input image, you can see that the person's image is captured easily by just applying the thresholding technique. So that is the use of thresholding.

### Simple Thresholding

Thresholding can be done in different ways. One is **simple thresholding**, another one is **adaptive**. Simple thresholding means for every pixel in the image, we have a single threshold. Like in this example, we have the threshold value 100. Every pixel in the image is compared to this value of 100. So that is what simple thresholding does: for every pixel, the same threshold value is applied. If the pixel value is smaller than the threshold, we set it to zero. If it is greater, we set it to the maximum value, which is 255. So that is simple thresholding.

```python
ret, thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
```

### Adaptive Thresholding

**Adaptive thresholding** is different. The concept is always the same because we compare a particular value with each pixel and make a decision. But how we select the threshold is different. In this case, if we have an image and we divide the image into regions, each region will have its own threshold. Maybe one region has a threshold of 100, another 200, another 50. The algorithm computes which region will have which threshold. So it is not a single threshold. Depending upon the pixel intensities, the algorithm will compute a threshold value and compare all the pixel elements in that region with that particular threshold and make the decision. So that is what adaptive thresholding does. The algorithm determines the threshold for a pixel based on a small region around it. We can have different thresholds for different regions of the same image.

So why do we need such a mechanism? Some of you already did the lab assignment and chose some images with low lighting, right? Some lighting variations, somewhere it is more bright, somewhere it is very dark, with variation of bright and dark. In that case, if you apply simple thresholding, it just becomes like a black and white mess, you will not be able to distinguish properly. In that case, we use adaptive thresholding. This is very useful with images having varying illumination or lighting.

```python
adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
```

### Otsu's Thresholding

**Otsu's thresholding** is also similar, but instead of computing thresholds for a region, it uses a mathematical equation based on the **pixel distribution**. So that is why we have another type of thresholding where the mathematical equation is different, which is used in the algorithm. But it is also more useful in scenarios with poor lighting.

```python
ret, otsu_thresh = cv2.threshold(gray_image, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

## Morphological Operations

Now the next one is **morphological operations**. This is very important. **Morphology** is a broad set of image processing techniques, and it is mostly dealing with **shapes**. When any question regarding shapes comes up, you can use morphological operations. This is mostly dealing with shapes.

The four major morphological operations are:

1. **Erosion**
2. **Dilation**
3. **Opening** (Erosion → Dilation)
4. **Closing** (Dilation → Erosion)

### Erosion

First one is **erosion**. As its name indicates, it **shrinks**. From the image itself, you can see the input image where the letter is wider, but in the output, it is thinner, right? That is how erosion works. How it works is that it makes pixels darker. Around the edges, for example, we have an image and a kernel. When we pass the kernel through the image, it takes the first set of pixels and checks if all the pixels under the kernel are one. If all the pixels under the kernel are one, then it remains one. Otherwise, it replaces it with zero. So even if one pixel is zero in that set, it will mark as zero.

**Erosion rule:**

> Output pixel = **1** only if **ALL** pixels under the kernel = 1
>
> Output pixel = **0** if **ANY** pixel under the kernel = 0

**Example with 3×3 kernel on a binary image patch:**

```
Input patch:     Kernel (all ones):
1  1  1          1  1  1
1  1  0          1  1  1
1  0  1          1  1  1

Centre pixel = 1, but NOT all pixels are 1 → Output = 0
```

What this basically does is it makes the image darker. It adds more black colour to the image, especially around the edges. That is where you see the difference. The boundary between zeros and ones comes around the edges. So it changes those ones into zeros so that the image gets darker and the object shrinks.

If a pixel is originally one, it will be considered one only if all the pixels under the kernel are one. Otherwise, it is made zero. All the pixels near the boundary will be discarded depending upon the size of the kernel. So towards the boundary, the thickness of the object decreases, the foreground decreases and the background increases. Why is this useful? It is used for removing small white noises and can be used for detaching two connected objects, et cetera. If two objects are connected by a thin bridge, that bridge will be removed by erosion, because when it is reducing the size, around that area you will have some zeros where there were ones, so it can break the connection. We are not altering the input image. We are writing into a new output image. So the input will stay as it is, but the output is the eroded version.

```python
kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(image, kernel, iterations=1)
```

### Dilation

The exactly opposite operation is **dilation**. As its name indicates, it **dilates** or **expands**. You can see the object has expanded.

**Dilation rule:**

> Output pixel = **1** if **AT LEAST ONE** pixel under the kernel = 1
>
> Output pixel = **0** only if **ALL** pixels under the kernel = 0

**Example with 3×3 kernel:**

```
Input patch:     Kernel (all ones):
0  0  0          1  1  1
0  0  1          1  1  1
0  0  0          1  1  1

Centre pixel = 0, but at least one pixel is 1 → Output = 1
```

So it adds brightness to the image, especially around the borders. That is how dilation works. It increases the brightness in the image, and the size of the **foreground** object increases. Where erosion decreased the foreground and increased the **background**, here the opposite happens. Normally, erosion is followed by dilation or vice versa. This is also useful in **joining broken parts**. If there is a gap which is needed, dilation will fill it.

```python
dilated = cv2.dilate(image, kernel, iterations=1)
```

### Opening (Erosion → Dilation)

> [!IMPORTANT]
> The lecturer explicitly stated that **Opening vs Closing** will be on the exam. Remember the order:
> - **Opening = Erosion → Dilation** (removes small white noise)
> - **Closing = Dilation → Erosion** (fills small holes/gaps)

Now we have two more operations. One is **opening** and the other is **closing**. Opening is actually **erosion followed by dilation**. Make sure you remember this because it is important for the exam. So do not get confused with opening and closing.

> **Opening = Erosion → Dilation**

This is used for removing small white noises. In the image, you can see there are some white noises in the background. Those are removed in the output. How it does this: first we erode. When we shrink, all the small white noise just becomes black, right? Then we dilate. When we dilate, the edges of the foreground expand back. So this is a combination of techniques.

```python
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
```

### Closing (Dilation → Erosion)

**Closing** is the opposite. We **dilate first and then erode**. This is useful for **closing small holes or gaps**. You can see some missing information or gaps in the object. When we dilate, those gaps get filled with white. Then when we shrink or erode, we get back the original size but with the gaps filled. So it is like you get the original image but without those gaps.

> **Closing = Dilation → Erosion**

```python
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
```

### Applications

All these morphological operations are highly used in **medical imaging** where detail is very critical, and also in **industrial applications** and **document processing**. You can see how we can clearly identify each letter with these operations. So it is very common in document processing.

## Image Transformation Techniques

Next is image transformation techniques. Image transformations are another set of tools that can be used for various modifications. You have already learned about rotation and translation; those are examples of transformations.

### Affine Transformation

An **affine transformation** is a general term for transformations like **rotation**, **shearing**, **translation**, et cetera. If we have an image which has some parallel lines, even after applying an affine transformation, we **maintain that parallelism**. The affine transformation will maintain all the parallelism and structure as it is.

**Affine transformation equation:**

> **y = Ax + B**
>
> Where:
> - **x** = input coordinates
> - **y** = output coordinates
> - **A** = affine matrix (e.g. rotation matrix)
> - **B** = translation vector

It can shift the image to the left or right or top or bottom.

### Types of Transformations

**Translation:** Shifting the image in the x or y direction. The object can be displaced to the left, right, top, or bottom.

```python
# Translation matrix: shift by tx, ty
M = np.float32([[1, 0, tx],
                [0, 1, ty]])
translated = cv2.warpAffine(image, M, (width, height))
```

**Rotation:** Rotating the image around a specified point. You remember passing the angle within the function, then finding the centre of the image for rotation, right? So rotating the image around a specified point, you specify that point as the centre of the image and rotate based on that centre.

```python
centre = (width // 2, height // 2)
M = cv2.getRotationMatrix2D(centre, angle, scale)
rotated = cv2.warpAffine(image, M, (width, height))
```

**Scaling:** Changes the size by a scaling factor. You can say you need to double the height or width, so it is based on a scaling factor.

```python
scaled = cv2.resize(image, None, fx=2.0, fy=0.5)  # double width, half height
```

**Shearing:** Slanting the image along the x or y axis.

## Summary

So those are the different image processing techniques. That is what the processing stage does. This is the core stage and why we need this. We can enhance the image, we can manipulate it for segmentation and other activities. We have already discussed the techniques like filtering, sharpening, blurring, and all these are applications of the concepts. The libraries are doing all the mathematical computations. We just need to make use of those functions that are already available.

## Next Week

Next week we will dive into feature detection and description. That is one of the interesting topics. We will go into depth on object detection, like how we detect objects. Similar to edge detection, we will see how the feature detection system works.
