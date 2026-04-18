# Object Detection Fundamentals

Instructor: Stephin Rachel Thomas
March 19, 2026

---

## Today's Topics

- Introduction to Object Detection
- Traditional Object Detection
- Deep Learning in Object Detection
- Detection Models Vs Classification models
- R CNN
- YOLO
- SSD

---

## Introduction to Object Detection

**Object detection** is a crucial aspect of computer vision, where the goal is to identify and locate objects in images or videos.

It's a step beyond image classification by not only categorizing the objects but also indicating **their location and scale within the scene**.

Common applications include surveillance, autonomous vehicles, and facial recognition.

Bounding boxes are represented by corner coordinates: `(x1, y1)` (top-left) and `(x2, y2)` (bottom-right), along with a class label (e.g., Person, Dog).

---

## The Evolution of Object Detection

A detector takes an input image and produces three outputs:
- **Bounding boxes**
- **Classes**
- **Score/Probability**

The field of object detection has transitioned from manual feature extraction and simple classifiers to sophisticated deep learning models. Early methods, like **template matching and feature-based approaches**, were limited by their rigidity and inability to handle variations in scale, viewpoint, and illumination. The advent of deep learning brought about a paradigm shift, leveraging neural networks to **automatically learn features** directly from data.

---

## Traditional Object Detection: Overview

In traditional object detection, feature extraction was a critical step. Techniques like **SIFT** (Scale-Invariant Feature Transform) and **HOG** (Histogram of Oriented Gradients) were widely used to describe local image appearances and shapes.

The **sliding window method** was then applied to systematically move across the image at various scales and extract these features. A classifier like **SVM** (Support Vector Machine) would then determine whether each window contains the object of interest.

*Technique: Sliding window with image pyramid*

---

## Traditional Object Detection: Key Concepts

**Descriptors** play a crucial role in defining the characteristics of objects. For example, SIFT identifies and describes local features in images, invariant to scaling and rotation, making it effective for matching different images of the same object.

The Histogram of Oriented Gradients (**HOG**) captures the structure of objects by aggregating local gradient directions or edge orientations.

The sliding window technique involves moving a window of various sizes across the image, extracting features at each position, and using a classifier like SVM to label each window as containing the object or not.

---

## Limitations of Traditional Object Detection

Traditional methods struggled with **variations in object scale, orientation, and lighting conditions**. The need for handcrafted features **limited their adaptability and effectiveness**, as these features might not generalize well across diverse scenarios. Additionally, the computational inefficiency of sliding windows, particularly in high-resolution images, posed significant challenges for real-time applications.

---

## Transition to Deep Learning

The advent of deep learning marked a transformative change in object detection. Neural networks, particularly Convolutional Neural Networks (CNNs), allowed for automatic feature extraction, learning complex patterns directly from data. This shift not only improved detection accuracy but also enabled the systems to adapt to a wide range of objects and scenes, overcoming many limitations of traditional methods.

### Comparison of Pipelines

**Traditional computer vision technology:**
- Image pre-processing → Human-designed features (Geometric, Texture, MFCCs) → Feature selection (SFFS, SBS, SFS) → Classifier (ANFIS, ANN, SVM) → Target

**Deep Learning:**
- Image pre-processing → Multi-layer neural network → Softmax → Target

---

## Deep Learning in Object Detection: Feature Extraction

In deep learning-based object detection, CNNs play a vital role in feature extraction. Layers of a CNN automatically learn to detect edges, textures, and eventually complex patterns as the network deepens. This hierarchical feature extraction process enables the model to learn a robust representation of objects, making it adept at handling variations in appearance and viewpoint.

---

## Deep Learning in Object Detection: Detection Head

The detection head of a deep learning model is responsible for predicting object classes and locations. Unlike classification models that output class probabilities, detection models also predict bounding boxes around objects. This involves not only identifying **'what'** is present in an image but also **'where'** it is.

Techniques like Region Proposal Networks (RPN) in Faster R-CNN and grid-based approaches in YOLO (You Only Look Once) exemplify different strategies for this task.

---

## Outputs of Detection Models vs. Classification Models

Detection models differ from classification models in their outputs. While classification models output a **probability distribution across different classes** for the whole image, detection models provide **class probabilities, bounding box coordinates, and sometimes confidence scores** for multiple objects within the image.

This distinction is crucial as it allows detection models to localize multiple objects and their scales within a single image, offering a more detailed understanding of the scene.

**Example outputs:**

Classification output:
```
"Cat"
```

Detection output:
```json
[
  { "label": "Cat", "bbox": [20, 30, 50, 100] },
  { "label": "Dog", "bbox": [100, 25, 40, 80] }
]
```

---

## Types of Detection Heads: Anchor-Based

Anchor-based detection heads use predefined bounding boxes (anchors) of various sizes and aspect ratios to detect objects. Techniques like Faster R-CNN generate region proposals based on these anchors, adjusting them to better fit the objects. This method is beneficial for detecting objects of different shapes and sizes but can be computationally intensive due to the large number of proposals.

For each anchor:
- **loc**: $\Delta(cx, cy, w, h)$
- **conf**: $(c_1, c_2, \ldots, c_p)$

Feature maps at multiple scales (e.g., 8×8, 4×4) are used for detection.

---

## Types of Detection Heads: Anchorless

Anchorless detection heads, seen in models like CornerNet and CenterNet, do away with predefined anchors. Instead, they directly predict the corners or centers of objects. This approach simplifies the detection pipeline and can reduce computational complexity. However, it might require more sophisticated training strategies to achieve the precision offered by anchor-based methods.

---

## Case Studies and Applications: Real-World Examples

Object detection has vast applications in today's world. In autonomous vehicles, it's used for pedestrian and vehicle detection to navigate safely. In retail, it assists in inventory management through product recognition. In healthcare, it aids in identifying anomalies in medical imaging. These real-world examples demonstrate the practical utility and transformative potential of object detection technology.

---

## Case Studies and Applications: Industry Use-Cases

In industries like security and surveillance, object detection plays a crucial role in monitoring and threat detection. In agriculture, it helps in crop analysis and yield prediction. In manufacturing, it's used for quality control by detecting defects. These use-cases highlight the versatility of object detection in providing solutions across various sectors.

---

## Advanced Topics in Object Detection: R-CNNs and Beyond

The development of R-CNN (Region-based CNN) and its successors, Fast R-CNN and Faster R-CNN, marked significant advancements in object detection. These models improved **accuracy** and **speed** by integrating region proposal networks with deep learning. Following these, methods like **SSD** (Single Shot MultiBox Detector) and **YOLO** (You Only Look Once) further optimized the process, enabling real-time detection by eliminating the need for separate region proposal stages.

---

## R-CNN (Region-based CNN)

R-CNN solves exhaustive search performed by sliding window, by proposing bounding boxes, and passing these extracted boxes to an image classifier (Eg: **ImageNet**).

**Selective search** algorithm is used for making bounding box proposals.

### R-CNN Pipeline: Regions with CNN features
1. **Input image**
2. **Extract region proposals** (~2k)
3. **Compute CNN features** (via warped region)
4. **Classify regions** (e.g., aeroplane? no. / person? yes. / tvmonitor? no.)

---

## Calculating Bounding Box in R-CNN

**Region proposal:** $(p_x, p_y, p_h, p_w)$
**Transform:** $(t_x, t_y, t_h, t_w)$
**Output:** $(b_x, b_y, b_h, b_w)$

### Translation:
$$b_x = p_x + p_w t_w \quad \text{(Horizontal translation)}$$
$$b_y = p_y + p_h t_h \quad \text{(Vertical translation)}$$

### Log-space scale transform:
$$b_w = p_w \exp(t_w) \quad \text{(Horizontal scale)}$$
$$b_h = p_h \exp(t_h) \quad \text{(Vertical scale)}$$

---

## Intersection Over Union (IOU) Metric

IOU metric is used to determine good bounding box.

$$\text{IoU} = \frac{\text{Size of Union}}{\text{Size of Prediction Box}}$$

> **Note:** The formula as shown on the slide appears incorrect. The standard IoU formula is:
> $$\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}$$

- Typically, an IoU over **0.5** is considered acceptable
- The higher the IoU, the better the prediction
- It is a measure of overlap

Legend:
- **Green** is our true bounding box
- **Red** is our predicted bounding box

---

## Advanced Topics: Single Shot Detectors (SSD) and YOLO

Single Shot Detectors (SSD) and YOLO (You Only Look Once) represent a leap in object detection, focusing on **speed** and **efficiency**. SSD discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales. YOLO, on the other hand, divides the image into a grid, and each grid cell predicts bounding boxes and class probabilities directly. These methods are renowned for their ability to detect objects in real-time.

---

## SSD vs YOLO

### SSD Architecture
- Input: 300×300 image
- Backbone: VGG-16 Network
- Extracted feature maps at multiple scales starting from 38×38
- SSD makes **8732 predictions** using 6 feature map layers
- Output: Object Detection → Non Max Suppression

### YOLO Architecture
- Input: 448×448 image
- YOLO Customized Architecture with 1024-depth feature maps (7×7 grid)
- Fully Connected layers → 7×7×30 output tensor
- **Detections: 98 per class** → Non-Maximum Suppression
- Performance: **63.4 mAP**, **45 FPS**

---

## Single Shot Detectors (SSD)

1. **Single Shot**: Unlike region-based methods that require multiple stages, SSD performs object detection in a single pass through the network, making it very fast.

2. **Multi-Scale Feature Maps**: SSD uses multiple feature maps at different scales to detect objects of various sizes. This allows the model to handle objects at different resolutions effectively.

3. **Default Boxes**: SSD introduces the concept of default boxes (also known as anchor boxes) with different aspect ratios and scales at each feature map cell. These default boxes act as reference points for predicting bounding boxes.

4. **Bounding Box Prediction**: For each default box, SSD predicts the offsets to the ground truth bounding box and the confidence scores for each class. This results in a set of bounding boxes with associated class probabilities.

5. **Non-Maximum Suppression (NMS)**: Similar to YOLO, SSD applies NMS to filter out overlapping bounding boxes and keep only the most confident detections.

---

## You Only Look Once (YOLO)

1. **Single Forward Pass**: Unlike traditional methods that apply the model to multiple regions of the image, YOLO processes the entire image in a single forward pass. This makes it extremely fast.

2. **Grid Division**: YOLO divides the input image into multiple grids. Each grid cell is responsible for predicting a fixed number of bounding boxes and their corresponding confidence scores.

3. **Bounding Box Prediction**: For each grid cell, YOLO predicts multiple bounding boxes, each with a confidence score that indicates the likelihood of an object being present and the accuracy of the bounding box.

4. **Class Prediction**: Along with bounding boxes, YOLO predicts class probabilities for each grid cell, indicating which object class (e.g., person, car, dog) is present.

5. **Non-Maximum Suppression (NMS)**: To reduce redundant detections, YOLO applies NMS to keep only the most confident bounding boxes for each detected object.

---

## Challenges in Object Detection

Despite advancements, object detection faces challenges like detecting small or occluded objects, handling diverse and complex backgrounds, and dealing with varying lighting conditions. Balancing precision and recall, especially in crowded scenes, remains a critical issue. There's also the challenge of computational resource requirements for training and deploying sophisticated models.

---

## Future Perspectives and Emerging Technologies

The future of object detection lies in integrating it with emerging technologies like augmented reality (AR) and the Internet of Things (IoT). The development of low-power, high-performance models is essential for edge computing applications. Furthermore, incorporating advances in artificial intelligence, such as explainable AI and reinforcement learning, can lead to more robust and intelligent object detection systems that understand context and interactions within a scene.

---

## Conclusion and Key Takeaways

In conclusion, object detection has evolved from traditional methods to advanced deep learning techniques, significantly enhancing its capabilities and applications. The field continues to grow, driven by ongoing research and technological advancements. Key takeaways include the importance of robust feature extraction, the efficiency gains from modern detection methods, and the challenges and opportunities that lie ahead in this dynamic and impactful area of computer vision.
