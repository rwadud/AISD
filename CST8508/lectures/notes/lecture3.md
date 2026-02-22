# Lecture 3: Feature Detection

## Topics Covered
1. Segmentation and Binary Images
2. Thresholding (Basic and Adaptive)
3. Contours
4. Feature Detection Introduction
5. Image Gradient and Gradient Formula
6. SIFT (Scale Invariant Feature Transform)
7. SURF (Speeded Up Robust Features)
8. ORB (Oriented FAST and Rotated BRIEF)
9. Feature Descriptors and HOG
10. Feature Matching (Euclidean, Hamming, Brute Force, FLANN)
11. Machine Learning in Feature Detection
12. Real-Time Feature Detection
13. Future Trends (Deep Learning, CNNs)

---

Today, we are going a little bit deeper into **feature detection**. This is one of the critical or crucial parts of a **machine vision** system. Without detecting features from the image, we cannot do any further processing. We need to identify features from the image. So today we will see how feature detection can be described in different ways.

Today we have a lot of topics to cover. The first one is **segmentation** and **binary images**. Last week, we learned about different stages. One of the stages was **thresholding**, and if you remember, I mentioned it is our first task, like out of those nine steps. So we will see how segmentation can be done using a threshold. You already did thresholding for your assignment, some of you at least. We will cover how it differs between **basic** and **adaptive thresholding**, and how we can apply this to the segmentation of an image. Then we will see what **contours** mean. It is a concept that helps you understand how to outline shapes, how to extract a shape from an object or shape from the image. Then we will see a feature detection introduction and then the basics of the algorithm. The main idea of all feature detection is the **image gradient**, and we will see the equation and all the stuff for computing the gradient. Then comes different types of feature detection techniques that we use, such as **SIFT** and **ORB**, which come under advanced feature detection. And then we talk about **feature descriptors**. Some of you already know about descriptors because you have seen them already. Then finally, towards the end, we will see **feature matching**. It is about matching the features we find between images. And then a bit about **deep learning** and feature detection.

## Segmentation and Binary Images

So the first topic is **segmentation** and **binary images**. Last week, we saw the thresholding. Thresholding can help us identify the proper features from the image. If you look at this image, you can see the input image is a coloured image which has a lot of shapes like circles, squares, connected squares, a bar, one is always a shape, etc. But if you see the output, this output represents the **segmented image** or **thresholded image**. If you look at the output, you can see, if you look carefully, not all the shapes are being transferred in the output image. Especially around this area, this blue colour shape, this object is missing here. And half of this red bag is missing here. So which means we don't need all the details from the input image for the processing. We need maybe a portion of it, or some of the objects from it. How we decide is by defining a **threshold** for choosing what we need for the processing. So that is what is being done here.

**Segmentation** extracts objects from an image for further processing. We are extracting particular objects from the image. And if you can clearly see, the output is a **binary image**. It is just a black and white image. It just contains zeros and ones. The output of segmentation is typically a binary image. Of course, it has values of zero and one. One indicates all the parts of the image that we want to keep, that we want to use for further processing. And zeros represent all the parts which we don't want, that are not necessary for the processing. So we can neglect those parts. By looking at this, you can get an idea, the output image or the thresholded image acts as a **mask** to the source image. If you put this mask, you can see what is retained from the image and the rest will be masked. So it acts like a mask to that source image.

> [!IMPORTANT]
> The lecturer stressed: the **output of segmentation is always a binary image**. Keep this in mind — binary images are a key input to many image processing algorithms.

As I mentioned, segmentation is one of the critical tasks because in image processing, we need input like binary images. In most of the cases in our application, we convert it into **grayscale** or we convert it into binary in order to make the processing much easier. That is why we say that segmentation is critical, because it generates a binary image. And we need this binary image for many of the image processing algorithms. So it acts as the mask, and then one of the typical ways to get a binary image is **thresholding**. When you apply a **basic or simple thresholding**, you get a binary image. Anything above the threshold we assign white, and anything below we assign black. So we get just a black and white image. Thresholding is a type of segmentation that looks at the value of the source image. It looks at each **pixel** in the source image, and it is compared with a **global value** or a central value to decide whether a single pixel or group of pixels should have a value of zero or one. Depending upon this comparison, we assign zero or one to a group of pixels or a single pixel. This is the basic concept of segmentation. Segmentation can be done by techniques like thresholding. The main thing you have to keep in mind is that the **output of segmentation is always a binary image**.

### Thresholding Matrix Example

So this is how we perform thresholding and how we get the binary image. The left side shows the input matrix and the right side shows the output matrix.

**Example 1: Threshold = 128**

In the input matrix, you can see the first column shows all zeros, the second one all 64, the third one 190, and 255.

| Input pixel | Comparison         | Output |
|------------|---------------------|--------|
| 0          | 0 < 128 → below     | 0      |
| 64         | 64 < 128 → below    | 0      |
| 190        | 190 > 128 → above   | 1      |
| 255        | 255 > 128 → above   | 1      |

In the first example, we take or we define our threshold as 128, a random number, and then we compare each pixel in our input matrix with that 128. Depending upon the decision, for example, the first pixel is zero. When you compare with 128, it is less than 128. So we keep that pixel at zero in the output. Similarly for the next pixel, 64, it is also less than 128. So we keep that pixel also zero in the output matrix. And when you go to the third, it is 190, and it is of course greater than 128, so we make it one. Similarly, 255 is also greater than 128, so we keep it one. Likewise, we process every pixel in the input matrix and we get a final matrix like this. From this matrix, you can get an idea of what it would be. This is a binary image, of course, and it represents an **edge**, because an edge is something like a part of the image which has a drastic contrast in the **intensity** value. From zero to one is a drastic change, black to white. So it indicates there is an edge. So just an additional idea, but this is how the matrix will look and stuff like that.

**Example 2: Threshold = 64**

That is the first example. When you come to the second example, we have the same input matrix. But this time we use a binary threshold which is 64, basically half of 128 plus one.

| Input pixel | Comparison        | Output |
|------------|---------------------|--------|
| 0          | 0 < 64 → below      | 0      |
| 64         | 64 = 64 → not above | 0      |
| 190        | 190 > 64 → above    | 1      |
| 255        | 255 > 64 → above    | 1      |

Here also we do the same comparison. Each pixel in the input matrix is compared with 64, and if it is less than 64, we keep zero, and if it is greater, we keep one. When you look at the output matrix, you can see the difference. Now we have more details and less unwanted stuff. Depending upon the application, for example, if you are doing medical imaging applications, we need more detail, so at that time we consider smaller thresholds because we need more detail from the image. But for any other application, maybe if you are developing a filter for your camera or something like a portrait mode or something, you can go for a much higher threshold. Depending upon your application, you can decide what kind of threshold you want to use, but the main idea is that depending on the threshold, the output will change.

### Basic Thresholding in OpenCV

So this is basic thresholding from OpenCV. We are reading in the image. And I have told you that one way to convert the colour space is you can specify from **BGR to grayscale**, you can do this. By reading your image, you can pass zero, which will make it read as grayscale. And then you are showing the image. Now I am computing the height and width of my input image. Since this is a coloured image, I use shape because we have a third dimension, which is the **number of channels**, as I explained last time. Then I define a binary array, which is basically the same height and width of my image, but just with zeros. And I define the threshold as 85, a random number, and I am traversing across the rows and columns through a for loop and I am looking at each pixel technically in the image, comparing it with the threshold. And if it is greater than the threshold, we assign the value 255. For graphics, **255 means white, zero means black**. But for the code, 255 works as one. So we use 255 here just to say we are making it white. This is a slow process because you take some time to traverse through the two for loops.

The alternative for this is, you already know, you can simply use `cv2.threshold` and pass the input image and the threshold. Here we have 85, and 255 is the value to be replaced if the pixel is passing the threshold. And this is the threshold type of thresholding. If you run this program, you can see it takes a lot of time with the manual approach. Basically the manual and the OpenCV method give the same output. There is no difference, but the OpenCV one runs much faster, so I recommend using that. I am just showing you what is going on behind the scenes. So this is my input image, which is converted into grayscale, and then this is the output for the manual one, and this is the output for the OpenCV one, and they are basically the same.

### Adaptive Thresholding

Now we have **adaptive thresholding**. You have seen that **binary thresholding** is not great for all scenarios. It is okay if you have a good image, where you can clearly separate your **foreground** and your **background**. But if you are having an image which does not have even lighting, some sort of shadow or something affecting your image, then it is very hard for binary thresholding to process that image. It is more likely that you will lose a lot of details from that. That is why we need another solution called **adaptive thresholding**. Adaptive thresholding, as I think I mentioned last week as well, instead of one thresholding value, we have **multiple threshold values** for the same image. We divide our image into **sub regions**, and for each sub region we will have a different threshold value. How this threshold value is calculated is by using two functions: **mean adaptive** or **Gaussian mean adaptive**. These are the two ways we can perform adaptive thresholding. Instead of taking a single global value for the comparison, the adaptive thresholding will use the **local neighbourhoods**. We specify how much will be the size of the neighbourhood, let's say four or five, so it takes that many neighbouring pixels and then performs a mean. It computes the **mean value** of those pixels and compares each pixel in that region with that mean value as the threshold. And if it is a **Gaussian mean**, then what you do is you apply the **Gaussian function**, and the **Gaussian weighted sum** of that local neighbourhood is used as the threshold for comparing each pixel in that region. By doing this, you can handle issues like being unable to set a single threshold value for the whole image. As I mentioned, it computes a threshold value for each sub region instead of for the whole image. And as I told you, there are two methods: **mean** and **Gaussian**. We can see the difference.

So I am showing you all three thresholding types. First is the binary one, where we have a threshold of 70, and we apply a simple thresholding with `cv2.threshold`. The function helps us generate the binary image based on this. For the adaptive thresholding, we use `cv2.adaptiveThreshold`. We specify the adaptive method, the threshold type, the **block size** which is the neighbourhood size, and we also specify a **constant C** which is subtracted from the mean. We subtract from the mean because otherwise it is more likely that more of the pixels will get whiter. So just to prevent that, we always use that constant C, which we subtract from the computed threshold. So this is how adaptive thresholding works. For adaptive thresholding, we specify the neighbourhood size (the block size) and the constant that is subtracted.

For the input image that had some issue with the lighting, you can see if we used binary thresholding, the output would be more darker because it has some shadow in the input. So it is not showing any important details. And you could see the bottom portions are not that great. When I go to mean thresholding, it gives a good result, but still you can see the text and the door, it is not that clear. In this scenario, the **Gaussian thresholding** is good because I can read the characters, even the numbers are clear. So that is why we refer to and recommend using **adaptive thresholding** if you have input images with uneven lighting.

## Contours

Now let's talk about **contours**. A contour is similar to **edge detection**. If you look at the input image, you can see we have a sword, and when you do the **contour detection**, you get an image like this which plots the shape of that weapon. You could easily get confused because edge detection would also have detected this particular shape. But the difference between edge detection and contour is that a **contour always forms a closed path**. It gives the exact shape of the object. But in edge detection, you have seen that there is no guarantee that you will get a closed path. Sometimes some part of the image's edge will be visible, but it is not always a closed path. For contours, it will always be a closed path. This will be useful for **shape analysis**. A **contour** is a curve that joins a set of points which are enclosing an area that has the **same intensity**. If you look at this image, you can see the same intensity for the sword in different portions. It covers all the points that enclose an object which has the same intensity or same colour. That is what is mentioned: the area of **uniform colour or intensity** forms the object that we are trying to detect. What we are trying to detect is the object which has uniform intensity. In this case, it is the sword, which has uniform intensity. Only the handle will have a more different colour. Even then, the handle has uniform colour by itself, so we were still able to identify the shape of the handle also. It always encloses a path which collects all the points enclosing that area. It is similar to edge detection but with the restriction that the edges detected must form a **closed path**. It defines the **boundary**, so we can say that it can be used to define the boundary of each object. This is useful for **shape analysis**, **object detection**, and **recognition**.

As I mentioned last time, segmentation is useful because the output of segmentation is a **binary image**, and this binary image is a key input to many of the image processing algorithms. One of the examples is contours. **Contours always need binary input**. So we need to have segmentation done before performing contour analysis. The output of segmentation, which is a binary image, is used as the input to contour detection for further processing.

### cv2.findContours

We have the function `cv2.findContours`. This is the OpenCV built in function for finding contours in any image. Basically, this function returns two outputs. The first one is **contours**. Contours are like a list of contours, and contours are represented as points, like x, y coordinates, at each location. So it represents a **vector of boundary coordinates**. Like anything in space which has a value and magnitude, the vector of boundary points may be something like x1, y1, x2, y2 values. For example, if we have a contour of a circle, for each point, we will have x1, y1 or x2, y2 values. So it represents a vector. That is what contours means.

The second output is **hierarchy**. In this image, you can see the image of the phone. You have a **child contour**. The "s" represents a child of that contour. So this is the **parent** and this is the **child** contour. It is within this image. If you use this function, it can even return the hierarchy. It can even give you the child contour and also the parent contour, etc. So that is why we have hierarchy and contours. Contours is a list of contours in the image, which is the first output of the `findContours` function. And **hierarchy** is the optional output vector with information about the **topology**, basically the **parent and child relationship**.

In this image, you can see that when you perform the contour analysis of the input image, we have contours around the shape of the phone and also the camera and the proximity sensor, and the flashlight. Then we have one on the pencil and also around the side. So it encloses the path and shows the shape of the object.

### cv2.drawContours

Once you identified all the contour points, you can draw them around the shape. How you can draw them is using the `drawContours` function. You can pass a **thickness** parameter. If you specify that the thickness is greater than or equal to zero, it draws an outline around your image. Otherwise, if the thickness is like minus one or anything less than zero, it just fills the region with whatever colour you supply. That is how you can draw the contours.

In this example, we have the input image, then we convert to grayscale, and then we apply thresholding and get a binary image with zeros and ones. This should be the input to the contour function. When we do the contour detection, we see that it clearly identifies the shape and highlights around the tree and the roots. So it encloses the path of the shape.

## Feature Detection

### Introduction

Next is the introduction to **feature detection**. It is the process of identifying and locating important features, significant structures, or patterns within the image. If you look at these two images, can you say that this is the image of the same building? Do you think this is the image of the same building or a different building? Same building, right? As a human being, how did you conclude that? What made you think that it is the same? You looked at maybe the towers, the arches, maybe the shape. You see an additional structure like a porch which was not present in the first image, but still you can say that it looks the same. And also, do you think the size is the same? From the images, there is a difference, because one image was like the camera was closer, so we feel like the object is much bigger. And in the other one, the camera viewpoint is much farther, so you can feel that it appears different, but it is the same.

That is the main principle. Even if it appears in a different **scale**, different **rotation**, different **lighting condition**, or different appearance, as a human being, we are able to recognize it by looking at the features. You looked at the shape, colour, edges, and stuff like that. But for machines, it is very hard to identify this. How we train them is by using some **feature detection algorithms** that can identify important features. For example, in the top portion you can see **corners**, and even the arch shape, and there are some structures which are highlighted in the image. They identify those features and they try to see if the same features are present in the second image. If they are exactly matching, the machine confirms that the images match. As a human being, we look at the colour, shape, and things like that to identify features. We call these interesting points "**features**."

**Feature detection** is the process of identifying and locating **significant structures or patterns**, which are the core features, within the image. These features are crucial for understanding and interpreting visual information. Tasks such as **object recognition**, of course, require identifying features from the image. It is also used for **motion tracking**. For example, for motion tracking used in sport analytics, if you are playing a sport, for example badminton, there are scenarios like there are close rallies together where the shuttlecock is hard to follow. They do tracking with the shuttlecock, they track the motion and make decisions. For this, they are tracking just the ball. We need to make sure that the ball is being tracked in every frame of that video. For this, we identify the feature of that ball, maybe the shape or some unique feature, and then we track it. That is one of the applications where we use feature detection.

### Types of Features

A **feature** is nothing but an interesting part of the image. Examples include **edges**. An edge is an interesting part. If you can identify the same edge in another image, you can say that the images are matching. That is just one example of a feature. As I mentioned earlier, anything like a sharp change in the **intensity**, for example even from zero to 20, is a sharp change. If you draw a line here, it is an edge because the white background then becomes black. So it represents an edge. That is what an edge means, a sudden change in the intensity. **Corners** of course mean the intersection of edges. From triangles, we get corners. **Blobs** are like a similar texture, sometimes a patch or something that appears in the image of the same intensity. We call them blobs. And **ridges** sometimes look like edges, but they are not connected to any other edges, they are just some kind of lines of high intensity. These are all examples of features from the image which help us with other processing.

Feature detection has been significant since the beginning of computer vision. Since the beginning, we use feature detection, but we were giving more focus on **edge detection**. Edge detection, like **Canny**, was a simple detection at that time. But now, as technology has grown, we have several complex algorithms, and there are also solutions from **deep learning** which help us have more efficient methods that provide more efficiency and accuracy than the traditional methods. The applications of feature detection span different fields: **autonomous vehicles**, healthcare for **medical imaging** and diagnosing diseases, and even entertainment for **augmented reality**, for enhancing the real world with digital overlays. In autonomous vehicles, we use it for **navigation**, **obstacle detection**, and **pedestrian detection**, etc.

### Image Gradient

The basic principle behind all feature detection is the **image gradient**. **Gradient** means it gives more importance to the direction in which the **intensity** of the image changes. The gradient measures the **directional changes in the intensity**. For example, if this is the image and this is a specific position, and we say the gradient value at this particular point, that will be the **magnitude**. The intensity value at this particular point is the x value. The intensity of the image can vary from left to right or from top to bottom, towards the centre or towards the outer edges. Depending on that, we will have an **orientation**. For each pixel there is a direction. There is a mathematical equation for computing this gradient. Each pixel will have a **magnitude** and a **direction or orientation**. That is the main concept of feature detection. If we are able to understand the orientation of the image, then even if the image appears in different rotation or different scaling, the system can still identify the features from any variation of the same image. Like in the previous image, as a human being, we were able to identify objects even if the scale was different, the camera viewpoint was different, and the size appeared to be different. But we were still able to identify. This is how the computer does it. The main idea or the trick for feature detection is the gradient. If the algorithm can find the gradient for each pixel, it can identify the appearance of the same feature in another image.

This is how a **histogram of oriented gradients** would look like. If you look closely, for each image, around that person, we clearly detected the shape, and it has a gradient drawn in different directions. The gradient is the directional changes in the intensity or colour of the image, and these are fundamental in identifying the features. This is the basic concept behind feature detection.

### Gradient Formula

This is the equation that we use for computing the **image gradient**. As I mentioned, this measures the change in the image function. For this particular pixel, we have an intensity and also a gradient showing towards which direction the colour or the intensity is changing. In this image, you can see towards the centre it is more dark, black colour, and as it goes outwards, it gets brighter. You can see towards the centre the arrows point towards the centre, which means that the intensity is changing, decreasing towards the centre. This is an example of how the gradient would look. The image intensity is converging towards the centre, it is getting darker, so we draw the orientation towards the centre. On the right hand side, you can see it is more darker towards the left side and the right side is more bright, so we represent the direction from right to left. The change in colour represents the **magnitude**, and the blue arrows represent the **direction** of the gradient.

This is the formula to find the image gradient. The gradient, as mentioned, is the change in intensity of the colour, or change in intensity of each pixel.

**Angle (orientation):**

$$\theta = \arctan\!\left(\frac{\partial F / \partial y}{\partial F / \partial x}\right)$$

Where:
- `∂F/∂y` = change in image function in the **y-direction**
- `∂F/∂x` = change in image function in the **x-direction**

**Magnitude:**

$$\text{Magnitude} = \sqrt{\left(\frac{\partial F}{\partial x}\right)^2 + \left(\frac{\partial F}{\partial y}\right)^2}$$

> [!IMPORTANT]
> The lecturer emphasized: you **don't have to worry about the gradient equations** — the algorithm handles them. But understand the **concept**: every pixel has a **magnitude** and **orientation**, and this is the foundation of all feature detection.

The equation computes the angle, which is **theta**, which is the **inverse tangent** of the change in the y direction divided by the change in the x direction. So F represents the change of the image function. **del F by del y** represents the change of the image function in the y direction, and **del F by del x** represents the change of the image function in the x direction. This is how the angle is calculated. The **magnitude** of a particular pixel is computed by taking the **square root** of the change in the x direction squared plus the change in the y direction squared. These are the equations. You don't have to worry about these equations at all. The algorithm is already taking care of it, but this is the concept behind feature detection. Once it can compute the magnitude and direction for each pixel, it will do a lot of sorting and some kind of cleanup to get the good **key points**. **Key points** is a term that we will use because key points mean **unique points** from an image. Those key points have a particular orientation which can be used for identifying the same image or the same object in different images. This is the equation for computing the gradient.

## SIFT (Scale Invariant Feature Transform)

So in the right side image, you can see the same book, but it is a little bit rotated, it is not the same size anymore, and also on top of the book another object is placed, so it is not clearly visible. This is called **occlusion**. There is a hindrance or it is overloaded. Our task for the computer is to find if the same book is present in the second image. And if it is present, draw **bounding boxes** to show that this particular item exists. For this, we can use different algorithms. One of them is **SIFT**.

**SIFT** stands for **Scale Invariant Feature Transform**. This is capable of handling scenarios with different **scale** or **rotation**. What it does is it identifies and describes **local features** in images, and it is **invariant to scaling, rotation**, and **partially invariant to changes in illumination**. It is not completely efficient in dealing with changes in illumination, but it can deal with it to some extent. It detects corners, circles, blobs, etc. So we extract the features which we need for the processing.

### Key Points

> [!IMPORTANT]
> The lecturer highlighted that **key points** is an important term you will use in your assignments. Each key point carries **unique information** via its **descriptor**.

The next concept is **key points**. Key points is a term that is very important. In your assignment, you will be using it to identify the key points from images. **Key points are special points in an image that carry unique information**. A key point is basically a feature, but it carries unique information. Each key point has data called a **descriptor**. Each descriptor for each key point is unique. It is not identical. From the image, we extract unique features for processing. That is what a key point is. These are special points in an image that carry unique information. The **SIFT algorithm** is a powerful method for detecting and describing local features in images.

### Three Main Steps of Feature Detection

There are different steps. The first one is **scale space extrema detection**. Don't get confused with these words. It is just giving you an idea of what is happening. The main idea of any feature detector is: the first step is to **detect the key points**, the second step is to **compute the descriptors**, and the third step is **feature matching**. These are the three stages of many feature detection algorithms.

### Step 1: Scale Space Extrema Detection

SIFT also does the same. The first step is to detect key points. How do we detect key points? In the first step, which is **scale space extrema detection**, what it does is if we have an input image, it creates replicas of this image in different sizes. Each of these replications is called an **octave**. **Octaves** mean **down sampled versions** of the input image. There can be multiple octaves. For example, the first octave may be half of the size of the image, octave two can be one quarter of the original size, octave three can be one eighth of the size, and so on. We make different scales of our input image. We apply **Gaussian blur** on each of these octaves. The blur smoothens the image at each level, and even if the image appears blurry or at a different scale, it can still be identified.

So first, what we do in scale space extrema detection is we downsample our input image into different octaves. Each octave will be different in scale from our input image. In each of these octaves, we apply Gaussian blur and we take a difference of these Gaussian blurred images. This is called the **Difference of Gaussian (DoG)**. For example, the difference would be Gaussian 2 minus Gaussian 1, and the next DoG would be Gaussian 3 minus Gaussian 2, and so on. We take a difference of the Gaussian blurred images in each octave and we get a sequence of **Difference of Gaussian** images. Then what we do is find our key points from these different images. We try to see what is the **maximum point** or **minimum point** from these Difference of Gaussian images. We consider those points as **potential key points**. We don't say that they are key points yet, but we consider them as candidates for potential key points.

### Step 2: Key Point Localization

In the second step, we identify the actual key points. So in the first step, we divide the image into octaves, apply Gaussian blur, take the Difference of Gaussian between those images, and from those Difference of Gaussians, we compute the **maxima and minima** by comparing each point with its **26 neighbours** across the adjacent scales and categorize them as potential key points. Then comes the second step where we find the actual key points. Here we eliminate **low contrast points**. If there is not much difference between adjacent points, we can just ignore them. We also **remove points that lie along edges**. I mentioned that we need only unique points. For edges, the points have similar intensity or something in common between the points along the edge. We just need maybe one point from one edge. We don't want to replicate the same information from the edge. So we can eliminate points that lie along edges because they have similar information. We just want **unique information**. All these things are taken into consideration, and then we have a final list of key points. This improves **stability and accuracy**.

### Step 3: Orientation Assignment

Finally, as I mentioned, for those key points we compute the **gradient**. The algorithm will compute the gradient, the **orientation** and **magnitude** of each point. This step is critical because it ensures **rotation invariance**. If you find the orientation of that point, then even if it appears in a different orientation or different scale, the algorithm is able to identify it.

### Step 4: Descriptor Computation

The fourth step is computing the **descriptors**. Here, around each key point, we take a region and divide it into some blocks, let's say **16 by 16 blocks**, and then we compute the **histogram** of each block. Finally, what we get as output will be a **one dimensional feature vector**. It can be a **128 dimensional vector**. So it will be a **128 dimensional descriptor**. It has a lot of key point information in it.

> **SIFT Descriptor Dimensions:**
> - Region around key point: **16 x 16** pixel block
> - Divided into **4 x 4** sub-blocks = 16 sub-blocks
> - Each sub-block produces an **8-bin** orientation histogram
> - Total descriptor size: 16 sub-blocks x 8 bins = **128-dimensional vector**

### Step 5: Feature Matching

The fourth and next step is **feature matching**. It can use different techniques. One of the techniques is **Euclidean distance**, where we compute the distance between points. Whichever distance is minimum, we take those points, the points which have the **smallest distance** between them, because that is the closest match. If they have greater distance, then they are not matching. This is how feature matching is done. This is the overall idea: we detect the key points, we compute the descriptors, and finally we match the features.

### SIFT Summary

For **SIFT**, what we do is we divide the image into **octaves**, we apply **Gaussian blur**, then we compute the **Difference of Gaussian**, and we select some key points. In the second step, we eliminate **low contrast values** and **points along edges**, so we get a finite set of key points. Then for those key points, we generate **descriptors** by considering some area around each key point and computing the **histogram**. The final output will look like a **one dimensional vector**. From this, if we want to do matching, we can do it. In this particular example, we don't do matching because we just have one image. We just identified the important features, like the corners, the top of the ear portion, and the eyes towards the mouth and the chin. All the important features are identified.

## SURF (Speeded Up Robust Features)

The next one is **SURF**, **Speeded Up Robust Features**. It is a faster alternative to SIFT, and it offers robustness to changes in **scale**, **rotation**, and **illumination**. SIFT gives partial efficiency for dealing with illumination, but SURF is much better in that regard. And SURF is faster than SIFT. Why it is faster is because it uses **integral images**. An **integral image** is an image in which each pixel's value is the **sum of the pixels above and to the left** of it.

### Integral Image Example

For example, if we have an image with intensity values like this, what the **integral image** does for each pixel is: take the pixel and sum all the pixels to its left and top. For the first pixel, there is nothing to its left or top, so we keep the same value. For the second pixel, we have a pixel to the left, so we add them. For the pixel below, there is nothing to the left, but there is a value on top, so we add that. And if a pixel has values both on top and to the left, it will add all of them. So each pixel becomes the sum of the pixels to its left and above it.

**Worked Example:**

Original image matrix:

|   |   |   |
|---|---|---|
| 1 | 2 | 3 |
| 4 | 5 | 6 |

Integral image computation (each cell = sum of all pixels above and to the left, inclusive):

| Cell     | Calculation                          | Result |
|----------|--------------------------------------|--------|
| (0,0)    | 1                                    | **1**  |
| (0,1)    | 1 + 2                                | **3**  |
| (0,2)    | 1 + 2 + 3                            | **6**  |
| (1,0)    | 1 + 4                                | **5**  |
| (1,1)    | 1 + 2 + 4 + 5                        | **12** |
| (1,2)    | 1 + 2 + 3 + 4 + 5 + 6                | **21** |

Resulting integral image:

|    |    |    |
|----|----|----|
| 1  | 3  | 6  |
| 5  | 12 | 21 |

This is what an **integral image** is. That is how the image is taken as input for the SURF algorithm. The advantage is that because we are using the integral image for **convolutions**, it uses **box features** for computing keys and is more suitable for **real time applications**. For example, for certain features, it can even detect a portion from a T shirt, like the flash logo from Sheldon's T shirt. So it is clearly matched. That is an example showing how SURF works. It detects fewer key points than SIFT, but they are very accurate.

### SURF Key Point Detection

Here also, the main steps are the same: first feature detection, second descriptor computation, third feature matching. The first step is key point detection. For that, SURF uses a **Hessian matrix** based detector, which is used to find key points. We don't need to go into much depth about that, but this is the technology that the algorithm uses for identifying the key points faster than SIFT due to the use of **integral images** and **box filters**.

### SURF Orientation Assignment

Again, like SIFT, we have different scales, as I mentioned earlier, different sizes like half, one quarter, etc. We represent the image at different scales. The third step is also similar to SIFT. We have **orientation assignment**, but instead of using the gradient computation, we use **Haar wavelets**. **Haar wavelet** is a method that helps us to identify changes in intensity. This is the method used for assigning the orientation. After identifying the key points, we assign the **dominant orientation**. From the **Haar wavelet response**, we get the dominant orientations in that key point. We take the dominant orientation, which again helps us identify the feature even if the image appears in different orientations in different images.

### SURF Descriptor Computation

Finally, the second main step is the computation of **descriptors**. Here, the fourth step does the descriptor generation. We take a smaller region around the key point and compute a descriptor. In SIFT, we had 16 by 16 blocks. Here, instead of 16 by 16, we take a **4 by 4 region**, and for each sub region, we apply the **Haar wavelet response** in the x and y directions. Instead of histograms, we use the **Haar wavelet** method. The output is **64 dimensional** instead of 128. So it is just half the size of SIFT, making it faster. It can detect fewer key points, but it is faster.

> **SURF Descriptor Dimensions:**
> - Region around key point: **4 x 4** sub-regions = 16 sub-regions
> - For each sub-region, compute Haar wavelet responses: `∑dx`, `∑dy`, `∑|dx|`, `∑|dy|` = **4 values**
> - Total descriptor size: 16 sub-regions x 4 values = **64-dimensional vector**
> - Compared to SIFT's 128 dimensions, SURF is **half the size** and faster to compute

## ORB (Oriented FAST and Rotated BRIEF)

> [!IMPORTANT]
> The lecturer repeatedly emphasized the **three main steps** of feature detection algorithms:
> 1. **Detect key points**
> 2. **Compute descriptors**
> 3. **Feature matching**
>
> SIFT, SURF, and ORB all follow this pattern.

Now for the advanced feature detection technique: **ORB**. **ORB** stands for **Oriented FAST and Rotated BRIEF**. It is a combination of different algorithms. It uses an algorithm called **FAST** and **BRIEF**. **FAST** is used for identifying key points, and **BRIEF** is a good algorithm for computing descriptors. I mentioned the three main steps: first is finding the key points, second is computing the descriptors, third is feature matching. FAST is a very good key point detector, and BRIEF is a good descriptor. ORB uses the combination of both. **FAST** stands for **Features from Accelerated Segment Test**, and **BRIEF** stands for **Binary Robust Independent Elementary Features**. These two are different algorithms, but we combine the best parts of both in ORB. ORB takes advantage of **FAST corner detection** to locate key points. In SIFT, we were computing the gradient for key point detection. In one algorithm we used the **Difference of Gaussians**, in SURF we used **Haar wavelets**. Instead of using the gradient, ORB uses **corner detection**, so it detects only the corners. It does not compute the gradient for all pixels. OpenCV has a function called `ORB_create` for creating the ORB detector. Lab 3 will be using this OpenCV function.

In this image, you can see ORB detected very few features, but all are prominent features. Especially you see the corners towards the ears and the corners which intersect the ears to the head. All the key points detected here are just the corner points. They don't detect a lot of points like SURF or SIFT, but still they detect good enough features. It focuses on **intensity changes**, making it robust and fast. It generates **binary descriptors**. In previous examples, we had 128 dimensional and 64 dimensional vectors. Here, instead of that, we have **binary descriptors** which have just **zero and one values**. It does not have anything else, just binary values. It is easy for the algorithm because binary is more understandable for the computer, so it is easier for the system to do the computations.

## Feature Descriptors

### Histogram of Oriented Gradients (HOG)

**Feature descriptors** provide, as mentioned earlier, a unique and robust representation of detected features. Each descriptor is unique. It has unique information for each key point. It is not identical information, but unique. This is very important for **feature matching** because it has all the unique information from one object, and we have unique information from the other image. So it is easy to compare and make a decision. One of the examples for feature descriptors is **histogram of oriented gradients (HOG)**. This is the one which we use in SIFT. SIFT uses **histograms of gradients**. The image shows absolute value of gradient in x and absolute value of gradient in y. If you take the absolute value of the x gradient, you get **vertical edges** or vertical features highlighted. And if you take the y gradient, you have all the **horizontal features** highlighted. It is a critical descriptive representation for identifying **human shape**. So this is one of the techniques we use in **autonomous vehicles** to identify **pedestrians**. If pedestrians are present, it can easily detect the human shape. That is why it is called particularly useful for **human detection** in computer vision. It plots the image of pixel orientation and gradients on a histogram and simplifies the representation of image. It works by analyzing gradients and their directions in localized portions of an image, creating a unique representation of human shape and motion. This methodology is specifically used for identifying the human shape. That is why it is used for identifying pedestrians.

### HOG Gradient Calculation: Worked Example

The descriptor has unique information, and if you look at this, this is how we can calculate the **gradient magnitude** and **angle** from a pixel area. If you look at the pixel from the top of the car, it is highlighted here. For that area, we have the pixel values around it:

**Pixel neighbourhood (3x3 around the target pixel):**

```
       171
  70  [pixel]  120
       205
```

**Computing gradient in x-direction (horizontal change):**

$$G_x = \text{right} - \text{left} = 120 - 70 = 50$$

**Computing gradient in y-direction (vertical change):**

$$G_y = \text{bottom} - \text{top} = 205 - 171 = 34$$

> Note: The lecturer approximated the y-direction as 50 as well for a simpler example. Using the lecturer's values:

**Using the lecturer's simplified values (Gx = 50, Gy = 50):**

The feature vector for this pixel is: **(50, 50)** showing the change in x-direction and change in y-direction.

**Gradient Angle:**

$$\theta = \arctan\!\left(\frac{G_y}{G_x}\right) = \arctan\!\left(\frac{50}{50}\right) = \arctan(1) = 45°$$

**Gradient Magnitude:**

$$\text{Magnitude} = \sqrt{G_x^2 + G_y^2} = \sqrt{50^2 + 50^2} = \sqrt{2500 + 2500} = \sqrt{5000} \approx 70.71$$

So for that particular area, the **gradient magnitude** (the value of the intensity change) is **70.71** and the angle is **45 degrees**. We got an orientation for that particular area. Similarly, we calculate for each sub-region and then we have a **histogram** which shows the orientation of each pixel, and we can define the orientation for each area. This helps with feature detection, even if the image appears in different scale or different orientation.

## Feature Matching

The last step of feature detection is **feature matching**. This is optional, but most applications use it. For example, **panoramic image stitching**. If you want to stitch two images, we need to find exactly matching objects in both images and stitch them together. These two images are not identical, but they have common points. This particular part is present here as well, and when we do feature matching, we can identify the similar points and finally stitch them together into a single image. So it was previously two different images and we stitched them by identifying the similar points and combining them. This is an example of how stitching is done. That is where we use feature matching. **Feature matching** involves identifying similar features, like corners and blobs, with the same textures, in different images. This is the key for tasks where the **correspondence** between features in multiple images is critical. We need a correspondence between two images. There should be a common point or common part of the image where we can stitch them together.

### Hamming Distance

For feature matching, we use different **distance computation** methods. One of them is **Euclidean distance**, and another one is **Hamming distance**. **Hamming distance** is used in **ORB** because, as mentioned, ORB has **binary descriptors**.

ORB is very efficient for matching. For matching, instead of Euclidean distance, it uses **Hamming distance**. In **Euclidean distance**, we compute the smallest distance between features, and if the distance is small, we say the features are matching. If the distance is high, we say they are not matching. Similarly, we use different techniques for feature matching. **Hamming distance** works as follows: if we have two **binary descriptors**, for each position we compare the values. If they match, the distance is zero. If they do not match, the distance is one. We count the total mismatches.

**Hamming Distance Worked Example:**

| Position | Descriptor A | Descriptor B | Match? | Distance |
|----------|-------------|-------------|--------|----------|
| 1        | 1           | 1           | Yes    | 0        |
| 2        | 0           | 1           | No     | 1        |
| 3        | 1           | 1           | Yes    | 0        |
| 4        | 0           | 0           | Yes    | 0        |
| 5        | 1           | 0           | No     | 1        |

**Total Hamming Distance = 0 + 1 + 0 + 0 + 1 = 2**

So we have two **binary descriptors**. **Hamming distance** is computed by comparing each value at the same index in both descriptors. If they are not matching, there is a distance of one. The total number of mismatches gives the Hamming distance. This is how Hamming distance is computed, and it is used in **ORB** because we have binary descriptors.

### Brute Force and FLANN Matching

Then another type of matching is **Brute Force** and **FLANN**. These are different ways to compute the distance between features. For your lab, you can either use Brute Force or FLANN. You don't need to do both, but you can choose between them. **Brute Force Matcher**, as its name indicates, is a brute force match. For each descriptor value in one image, it looks for a match in all the descriptors in the other image. So it takes every point and looks for a match in all the other points. It is more time consuming and it is not the best method if you have a large dataset. **FLANN** is quite useful because it is much faster to compute the distance. Applications involve **panoramic image stitching**, **motion tracking** (sport analytics is an example for motion tracking), **object recognition**, and **3D model reconstruction**. In all of these applications, we need feature matching. Feature matching is an optional step in feature detection, but for these applications, we need it.

## Algorithm Comparison

| Feature       | SIFT                        | SURF                        | ORB                              |
|--------------|-----------------------------|-----------------------------|----------------------------------|
| **Speed**     | Slowest                    | Faster than SIFT            | Fastest                          |
| **Key Point Method** | Difference of Gaussian (DoG) | Hessian matrix + integral images | FAST corner detection    |
| **Orientation** | Gradient histogram         | Haar wavelet response       | Intensity centroid               |
| **Descriptor** | 128-dimensional vector      | 64-dimensional vector       | Binary (0s and 1s)               |
| **Descriptor Method** | Histogram of gradients | Haar wavelet (x, y)        | BRIEF                            |
| **Matching Distance** | Euclidean distance    | Euclidean distance          | Hamming distance                 |
| **Scale Invariant** | Yes                   | Yes                         | Yes                              |
| **Rotation Invariant** | Yes                | Yes                         | Yes                              |
| **Illumination** | Partially invariant      | Better than SIFT            | Robust via intensity changes     |

## Machine Learning in Feature Detection

Now, about **machine learning** in feature detection. As I mentioned, feature detection has been around since the beginning of computer vision. Since the advancement in the field of **AI** and **deep learning**, we are also advancing the algorithms for feature detection. There are different techniques: **supervised learning**, **unsupervised learning**, and **semi supervised learning**. These are also applicable for feature detection. Here also we use these techniques. In **supervised learning**, we use **labeled data** for training. For example, email spam detection: we give the labels to identify the spam and train the model with the labeled data. For **unsupervised learning**, we don't give any labels, but the model learns from the **patterns**. An example is customer segmentation based on purchasing behaviour. If you are a customer who buys only branded products, or a customer who doesn't care about the brands, the system can categorize you based on your history of purchasing. Based on the pattern, it can predict your behaviour. **Semi supervised learning** combines both approaches. We train the model with labeled data, and once it reaches a certain maturity, there is no need for providing labels anymore. For example, Google Photos can learn to identify a person's face. And after a point in time, it can easily identify that person's photos from your gallery. These are examples of the different techniques used in machine learning for feature detection.

## Real-Time Feature Detection

**Real time feature detection** is one of the challenging tasks in feature detection. It is very hard to achieve because of many reasons: **computing power** and **efficient algorithms** all need to be there. But real time processing is really needed because, for example, with **autonomous vehicles**, we cannot rely on a system which is very slow. It has to make decisions quickly. If there is a pedestrian, it should apply brakes. You cannot wait. So real time feature detection, or real time processing, is very important in machine vision. It still faces significant challenges, particularly in balancing **computational cost** with the need for **accuracy**. The current scenario is that if you want to have a highly accurate system, the computational load will be very high. We are looking for something lightweight but which should have more accuracy and efficiency. In applications like **surveillance** and **autonomous driving**, where decisions must be made quickly and accurately, these challenges are amplified because it affects human life. We cannot risk lives while relying on a slow system. It has to be fast and reliable.

### Remedies

Common remedies include **algorithm optimization**, where you optimize the complexity of your algorithms. Another remedy is the use of **low level programming languages**. Python is easy, but if you need to build real time systems, usually you use **C or C++** or other low level languages because the overhead of interpreting takes time. Use of low level programming languages is another remedy, which is harder for humans to write, but it is necessary. Another remedy is utilizing **hardware acceleration** like **GPUs**, for more computing power. These are the remedies for the challenges we have in real time feature detection.

## Future Trends

For future trends, of course **deep learning** and **neural networks** continue to advance and help with feature detection. **Neural networks** help us to learn patterns automatically, so they can be used for faster feature detection. Slowly we are departing from traditional methods of handwritten algorithms and relying more on **deep learning algorithms**, which help us deal with larger datasets. Even if the data is not well curated, AI algorithms can manipulate it by doing image processing, and you can even broaden your dataset by using some techniques that you will learn in upcoming classes, how to expand your dataset and what key things you have to keep in mind when selecting a dataset. Future trends involve **deep learning** and **neural networks**, and we need to think about building smarter feature detection systems that can improve over time, learning from new data and experience. We are slowly moving away from traditional algorithms which took a lot of time and computational power. Now we have deep learning techniques like **CNNs** which help us do the job much better and faster.

## Next Week Preview

Next week we will be learning about one of the core architectures, which is **CNN**. We will have our introduction to **CNNs**, their architecture, and how CNNs have solved a lot of problems in computer vision.
