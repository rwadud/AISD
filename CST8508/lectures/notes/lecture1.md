# Lecture 1: Introduction to Machine Vision

## Topics Covered
1. Introduction and the Rise of Deep Learning
2. Key Technologies (Sensors, Image Processing, Machine Learning)
3. Applications of Machine Vision in Everyday Life
4. Case Study: Machine Vision During COVID-19
5. Machine Vision System Workflow (Acquisition, Processing, Interpretation)
6. Introduction to Image Processing
7. OpenCV Basics (Reading, Displaying, and Understanding Images)
8. Pixels and Image Representation
9. Advanced Techniques: Neural Networks and CNNs
10. Career Opportunities in Machine Vision

---

## Introduction and the Rise of Deep Learning

> [!IMPORTANT]
> The lecturer hinted: "Maybe you'll get this for your exam" — the rise of **deep learning** and its impact on machine vision.

The rise of **deep learning** techniques brought a drastic change in the field of **machine vision**. It is the rise of deep learning that brought in drastic change in the field of machine vision.

## Key Technologies

There are mainly **three steps** in machine vision technology.

### 1. Image Capturing

For capturing the image, we need some devices. We can use **cameras** or **sensors** for this. The popular or most commonly used image sensors are **CCD** and **CMOS**, which are widely used in our cameras. Smartphone cameras and digital cameras all use CCD and CMOS. The image sensor is one of the key technologies.

### 2. Image Processing Techniques

Once we have acquired the image using the sensors, the second step is processing that image. For that, we need some techniques. An example is **feature and edge detection**, which you are doing in your lab activity. You are using some filters, like **Canny edge detection**. All those things are technologies used in machine vision.

### 3. Machine Learning (Deep Learning)

The role of **machine learning**, particularly **deep learning**, is the key turning point. All these technologies work together to enable machines to understand visual data: **sensors**, **processing techniques**, and **machine learning**, especially deep learning. All these technologies make up machine vision.

## Applications of Machine Vision in Everyday Life

We see machine vision everywhere, every day.

### Facial Recognition

From your cell phone, using **facial recognition** to unlock your smartphone, you are dealing with machine vision technology daily.

### Retail (Barcodes and Inventory)

It is actually used for **barcodes**. When you go to Walmart or any other supermarket, you can pay by yourself by scanning the barcode. When you scan the barcode, you get all the information. That is one application of machine vision.

It is also used for **inventory management**. For employees working as inventory storekeepers, they can keep track of inventory, like how many packets of milk are there. If the shelves are empty, it will alert them to restock. These are some of the applications in retail.

### Manufacturing (Quality Control)

In the **manufacturing industry**, it can be used for **quality checks** on the assembly line. We have devices over **conveyor belts** that take images of the products and then process them to compare with expected values. If it matches, they can pass. If not, we can say the product is not good. That is how we use machine vision in manufacturing.

### Entertainment

In movies, we see a lot of **visual effects** these days. That is a practical application in entertainment. We also have **augmented reality** for games. These are simple examples in entertainment.

### Automobile Industry

In the automobile industry, there is a big change. Now we have **self-driving cars**. For example, **Tesla** and similar autonomous vehicles use machine vision. This is not possible without machine vision because the sensors analyze the image, process it, and make rapid, instant decisions. If an obstacle is there, it does not hesitate; it just applies the brake at that moment. Machine vision systems are not simple because they need quick decision-making capacity, especially for **life-threatening systems**. Depending upon the criticality of the system, you need to choose which algorithm to use.

It is also helping in **lane keeping**, **traffic sign detection**, and **adaptive cruise control**. All those are practical scenarios or applications of machine vision in automobiles. You can think about how these technologies are helping our everyday life, making our lives much easier.

### Healthcare

**Autonomous vehicles** use machine vision for navigation. In the **healthcare industry**, machine vision is used even for **diagnosing diseases**. From medical image scans, the system can detect whether there is a **tumour**, **fracture**, or other issue. It can suggest to doctors symptoms of certain diseases. It is used for **diagnostics** and **robotic surgeries**, which are very common these days. Those are very practical applications of machine vision in advanced fields.

### Security

**Facial recognition** systems enhance security. Even in **airport security systems**, machine vision can be used for security purposes. These applications highlight the transformative power of machine vision.

### Agriculture and E-Commerce

In **agriculture**, machine vision on **drones** can help farmers monitor their crops. It is also used to determine when crops are ready for harvest and to sort good produce from bad, such as separating good apples from damaged ones. The **e-commerce** industry also utilizes machine vision.

## Case Study: Machine Vision During COVID-19

During **COVID-19**, we had access control to supermarkets and public spaces based on machine vision systems. There was a limit on the number of people inside the supermarket, so they had a count of people entering and leaving.

There were also **fever detection cameras**, which are **thermal cameras** that can sense your temperature. If you have a temperature like **37 degrees Celsius**, they can identify a fever.

> **Temperature detection example:**
>
> Body temperature reading = 37°C → **Fever detected** (flagged by thermal camera)

Other applications include **social distancing detection**, **sanitizer prioritization**, and **face mask compliance**. If you were not wearing a face mask, you could be detected by the camera.

## Machine Vision System Workflow

The workflow of a machine vision system can be summarized in **three steps**:

### 1. Image Acquisition

Acquiring the image by capturing it using **sensors** or **cameras**.

### 2. Image Processing

This is where you do the analysis. We analyze and manipulate the image. Sometimes the captured image is not clear, so you can apply techniques to make it sharper and provide more detail for visualization. Image processing is very important for machine vision. This is where we prepare our data or polish our dataset to use as input.

### 3. Interpretation or Action

This is where your actual model works. We make a decision based on the data we get from step two. For example, in an **automated inspection system** in manufacturing, we capture the image, do some processing, compare with expectations, and then make a decision.

## Introduction to Image Processing

**Image processing** forms the core of machine vision. Without processing, the machine vision system cannot perform. We have different formats for images, like **RGB**, and different **colour spaces**. Colour spaces define how colour is represented. We deal with different types of images, which means different types of content for our dataset, so we need to normalize or prepare the dataset using image processing.

We can read the image, display the image, and change it into a different colour space. We can do **edge detection** or **filtering** by smoothing or sharpening.

## OpenCV Basics

As you know, we are using **OpenCV** for our lab activities because it is a tool that helps with image manipulation. It is not used for developing or training your model but specifically for processing images.

### Reading and Displaying Images in Google Colab

In our lab, we import `cv2`. Since we cannot use `cv2.imshow` in Google Colab because it does not have access to the system window, we use `cv2_imshow` from the package `google.colab.patches`. We read the image using `cv2.imread` and print the shape.

```python
import cv2
from google.colab.patches import cv2_imshow

# Read the image
image = cv2.imread('husky.jpg')

# Print the shape
print(image.shape)
```

### Understanding image.shape

`image.shape` returns a **tuple** giving the **height**, **width**, and **number of channels**.

**Example output:**

```
(555, 830, 3)
```

| Component    | Value | Meaning                           |
|-------------|-------|-----------------------------------|
| Height      | 555   | 555 pixels tall                   |
| Width       | 830   | 830 pixels wide                   |
| Channels    | 3     | Red, Green, Blue (RGB)            |

> **Total pixel count:**
>
> 555 × 830 = **460,650 pixels**
>
> Each pixel has 3 channel values, so total values = 555 × 830 × 3 = **1,381,950 values**

If it was a **grayscale image**, we would just have the height and width because there are no channels:

```
(555, 830)
```

### Printing Pixel Values

`print(image)` prints all the pixel array values.

**Example output:**

```
[[[33, 37, 42],
  ...]]
```

For example, `[33, 37, 42]` shows the **RGB values** for the first pixel:

| Channel | Value | Meaning                        |
|---------|-------|--------------------------------|
| Red     | 33    | Intensity of red (0–255)       |
| Green   | 37    | Intensity of green (0–255)     |
| Blue    | 42    | Intensity of blue (0–255)      |

This represents the intensity of red, green, and blue for that particular pixel.

## Pixels and Image Representation

An image comprises a large number of **pixels**, which is the basic unit of an image.

- In a **coloured image**, a pixel has **red, green, and blue** intensities (a tuple of 3 values). For example, a pixel might have values `[29, 49, 88]` meaning an intensity of 29 for red, 49 for green, and 88 for blue.
- In a **black and white (grayscale) image**, it has values from **0** (black) to **255** (white), representing a single intensity value.

> **Pixel intensity range:**
>
> - **0** = Black (darkest)
> - **255** = White (brightest)
> - Values between 0–255 = Shades of grey

## Advanced Techniques: Neural Networks

**Computer vision** has come a long way due to advancements in **AI**, especially **neural networks** or **deep learning** techniques. This evolution has revolutionized machine vision. Because of AI, we get more accurate systems that can recognize complex patterns from images. Even if objects are **occluded** or overlapping, neural networks can identify distinct objects.

### CNN (Convolutional Neural Network)

**CNN** (Convolutional Neural Network) is one of the backbones for this. A CNN can automatically do **feature extraction** from images. That is why we say CNNs are the backbone for a machine vision system. CNNs are powerful for **automated feature extraction** in images. If we put good, polished images into the system, we can get the results we want.

CNN stands for **Convolutional Neural Network** because it uses the mathematical operation called **convolution**. It is a class of neural networks specifically applied to analyze **visual imagery**. Unlike traditional neural networks that process text or other data, CNNs are specifically for **images** or **video**.

### How CNNs Work

CNNs are trained with a large amount of data (images). For example, if we train with a dataset of **cats and dogs**, and input an image of a cat, it goes through different **layers** and finally gives the output "**cat**". If we input a dog, it should say "dog." The input is being passed through different layers, and while it is being passed through each layer, the system extracts some useful information. Using all this information, it makes a decision. We will learn the CNN architecture and all those details in upcoming sessions.

**CNN Classification Flow:**

```
Input Image (Cat) → Layer 1 → Layer 2 → ... → Layer N → Output: "Cat"
                    ↓           ↓                ↓
              Feature      Feature          Final
              Extraction   Extraction       Decision
```

## Career Opportunities

There are several career opportunities in this field:

| Role                        | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| **Machine Vision Engineer** | Designs and implements machine vision systems                               |
| **Data Scientist**          | Analyzes datasets and converts data into meaningful information              |
| **AI Specialist**           | Innovates new algorithms or technologies                                    |

Industries are actively seeking those with expertise in machine vision technology because there are a large number of applications requiring sophisticated, reliable solutions.

## Recommended Resources

The lecturer mentioned several recommended resources for the course:

- A **CNN animation video** that gives a clear idea of how different classification problems or object detection can be achieved using CNNs.
- **ChatGPT** as a useful tool for asking questions related to the course.
- The course will be using **OpenCV**, **TensorFlow**, and **PyTorch** as the main libraries and frameworks.

## Summary

We have gone through the basics of machine vision. There are three main steps: **acquisition**, **processing**, and **making decisions**. The technology has evolved since the invention of imaging devices. One of the key technologies is sensors like **CCD** and **CMOS**. We have image processing techniques like filtering, smoothing, and other processing techniques. Finally, we have machine learning techniques, especially **neural networks** and **deep learning**.

We have also gone through various applications: in **retail** we use it for barcode scanning; in the **automobile industry** we use it for lane keeping and obstacle detection; in **entertainment** for everyday use; and in advanced fields like **healthcare** for diagnostics and robotic surgeries, and in the **automobile industry** again for navigation.

Machine vision is a broad field. It is not small — it is bigger than **computer vision**.

## Q&A Highlights

- **Biometrics:** Machine vision is used for biometrical applications, as it involves image processing.
- **Manufacturing monitoring:** Beyond quality control, machine vision can also be used to monitor machines for problems and assist with planning, rather than just checking final products.
- **Agriculture details:** Machine vision helps determine when crops are ready for picking and can sort good produce from bad (e.g., good apples from damaged ones).
- **Power and scale:** One challenge discussed is the computing power needed. The focus is on making systems that can work with limited power but still process large amounts of data.

## Next Week

Next week, we will be learning more about the second step, which is **image processing** — the fundamentals of image processing and what techniques can be used.
