# CST8508 Potential Exam Questions Review

This review is based on the cleaned lecture transcripts. The strongest items are topics the lecturer directly connected to the exam or repeated as important.

## Priority Guide

- **Must Know:** Directly mentioned as exam-related, called most important, or repeated in lecture recaps.
- **Likely:** Strongly emphasized or central to multiple lectures.
- **Nice to Know:** Useful supporting detail, but less likely to be the main question.

## One-Page Cram Sheet

### Must Know Formulas

```text
CNN output size = (N - F + 2P) / S + 1
```

```text
IoU = Area of overlap / Area of union
```

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 x (Precision x Recall) / (Precision + Recall)
```

### Must Know Definitions

- **Opening:** Erosion then dilation. Removes small white noise.
- **Closing:** Dilation then erosion. Fills small holes or gaps.
- **CNN:** Neural network designed for image data using convolution, pooling, and fully connected layers.
- **Convolution:** Sliding a kernel/filter over an input and computing dot products.
- **Pooling:** Downsampling feature maps to reduce size and computation.
- **Segmentation:** Dividing an image into meaningful regions; often produces a binary mask.
- **Object detection:** Finds what object is present and where it is using bounding boxes.
- **Object tracking:** Follows objects across video frames while maintaining identity.
- **Sensor fusion:** Combines multiple sensor data sources to improve accuracy and reliability.

### Must Know Comparisons

| Comparison | Key Difference |
|---|---|
| Classification vs Detection | Classification says what; detection says what and where. |
| Detection vs Tracking | Detection works per frame; tracking follows objects across frames. |
| Opening vs Closing | Opening removes noise; closing fills gaps. |
| SIFT vs SURF vs ORB | SIFT accurate/slow, SURF faster, ORB fastest/binary. |
| SSD vs YOLO | SSD uses multi-scale feature maps; YOLO uses grid prediction in one pass. |
| Single-stage vs Two-stage tracking | Single-stage is faster; two-stage is usually more accurate. |
| CCD vs CMOS | CCD higher quality/global shutter; CMOS cheaper/power efficient/rolling shutter. |

### Must Know Diagrams To Practice

- CNN architecture: `Input -> Conv -> Pooling -> Fully Connected -> Output`
- Object detection architecture: `Backbone -> Neck -> Detection Head`
- Sensor fusion pipeline: `Raw data -> Preprocess -> Align/Synchronize -> Fuse -> Unified output`

## Worked Examples

### Example 1: CNN Output Size

Question:

```text
Input size N = 7
Filter size F = 3
Padding P = 1
Stride S = 2
```

Solution:

```text
Output = (N - F + 2P) / S + 1
       = (7 - 3 + 2(1)) / 2 + 1
       = (7 - 3 + 2) / 2 + 1
       = 6 / 2 + 1
       = 4
```

Final answer:

```text
4 x 4
```

### Example 2: No-Padding CNN Output Size

Question:

```text
Input size N = 6
Filter size F = 3
Padding P = 0
Stride S = 1
```

Solution:

```text
Output = (6 - 3 + 2(0)) / 1 + 1
       = 3 + 1
       = 4
```

Final answer:

```text
4 x 4
```

### Example 3: Max Pooling

Question:

```text
Input block:
6 2
5 5
```

Max pooling output:

```text
6
```

Reason: max pooling keeps the largest value in the region.

### Example 4: Average Pooling

Question:

```text
Input block:
6 2
5 5
```

Average pooling output:

```text
(6 + 2 + 5 + 5) / 4 = 18 / 4 = 4.5
```

### Example 5: Confusion Matrix Metrics

Question:

```text
TP = 40
TN = 50
FP = 10
FN = 20
```

Solution:

```text
Accuracy = (40 + 50) / (40 + 50 + 10 + 20)
         = 90 / 120
         = 0.75

Precision = 40 / (40 + 10)
          = 40 / 50
          = 0.80

Recall = 40 / (40 + 20)
       = 40 / 60
       = 0.67

F1 = 2 x (0.80 x 0.67) / (0.80 + 0.67)
   = 0.73 approximately
```

### Example 6: IoU

Question:

```text
Overlap area = 30
Union area = 60
```

Solution:

```text
IoU = 30 / 60 = 0.5
```

Interpretation:

An IoU of 0.5 is usually treated as the minimum threshold for a good bounding box in the lecture.

### Example 7: Opening vs Closing Scenario

Question:

```text
An image has small white noise in the background. Which operation should you use?
```

Answer:

```text
Opening, because opening removes small white noise.
```

Question:

```text
An object has small black holes or missing gaps. Which operation should you use?
```

Answer:

```text
Closing, because closing fills small holes or gaps.
```

## Highest Priority: Direct Exam Cues

### 1. What is the difference between opening and closing in morphological operations?

**Opening** is erosion followed by dilation.

It is used to remove small white noise from an image. Erosion removes small bright noise, then dilation restores the main foreground object.

**Closing** is dilation followed by erosion.

It is used to fill small holes or gaps. Dilation fills the gaps, then erosion brings the object closer to its original size.

Remember:

```text
Opening = Erosion -> Dilation
Closing = Dilation -> Erosion
```

### 2. Draw and explain the basic CNN architecture.

A basic CNN architecture is:

```text
Input Image -> Convolutional Layer -> Pooling Layer -> Fully Connected Layer -> Output Layer
```

The lecturer specifically said this is what you may be asked to draw.

Key parts:

- **Input layer:** Receives the image pixels.
- **Convolutional layer:** Applies filters/kernels to extract features such as edges, textures, and shapes.
- **Pooling layer:** Downsamples feature maps to reduce size and computation.
- **Fully connected layer:** Combines extracted features and performs final reasoning/classification.
- **Output layer:** Produces class scores or probabilities. The number of neurons usually equals the number of classes.

### 3. Perform a convolution operation on a matrix.

You may be given an input matrix and a kernel/filter, then asked to compute the output.

Process:

1. Place the kernel over the first region of the input.
2. Multiply matching values between the kernel and input region.
3. Add the products together.
4. Move the kernel by the stride.
5. Repeat until the full output matrix is computed.

Key point: **kernel** and **filter** mean the same thing in this context.

### 4. Compute CNN output size.

Use this formula:

```text
Output size = (N - F + 2P) / S + 1
```

Where:

- **N** = input size
- **F** = filter/kernel size
- **P** = padding
- **S** = stride

Example:

```text
Input N = 5
Filter F = 3
Padding P = 1
Stride S = 2

Output = (5 - 3 + 2(1)) / 2 + 1
       = 4 / 2 + 1
       = 3
```

Answer should be written as **3 x 3**, not just 3.

## Very High Priority: Strongly Hinted Topics

### 5. What are the steps in Canny edge detection?

Canny edge detection finds important edges in an image.

Steps:

1. **Noise reduction:** Apply Gaussian blur to reduce noise.
2. **Gradient calculation:** Compute edge strength and direction.
3. **Non-maximum suppression:** Thin the edges by keeping only local maximum values.
4. **Double thresholding:** Classify pixels as strong edges, weak edges, or non-edges.
5. **Edge tracking by hysteresis:** Keep weak edges only if connected to strong edges.

Important detail:

- Values below the lower threshold are discarded.
- Values above the upper threshold are strong edges.
- Values between thresholds are weak edges.
- Weak edges are kept only if connected to strong edges.

### 6. What is thresholding, and why is it used for segmentation?

Thresholding converts an image into a binary image by comparing pixel values to a threshold.

Binary image values are usually:

```text
0 = background / black
1 or 255 = foreground / white
```

Thresholding is used for segmentation because it separates the foreground object from the background.

Types:

- **Simple/global thresholding:** One threshold is applied to the whole image.
- **Adaptive thresholding:** Different thresholds are computed for different regions.
- **Otsu's thresholding:** Automatically chooses a threshold based on pixel distribution.

Use adaptive thresholding when lighting is uneven.

### 7. What is segmentation?

Segmentation divides an image into meaningful regions.

The lecturer emphasized that segmentation output is a **binary image**, which can act as a mask for further processing.

Many algorithms, such as contour detection, require binary input.

### 8. What are contours?

Contours are closed boundaries around objects or regions in a binary image.

They are useful for:

- Shape analysis
- Object boundary detection
- Object recognition

Important point: contour detection usually requires segmentation first, because it needs a binary image.

### 9. Compare SIFT, SURF, and ORB.

| Detector | Main Idea | Descriptor | Matching | Speed |
|---|---|---|---|---|
| SIFT | Uses Difference of Gaussian and gradients | 128-dimensional vector | Euclidean distance | Slowest |
| SURF | Uses integral images, Hessian matrix, Haar wavelets | 64-dimensional vector | Euclidean distance | Faster than SIFT |
| ORB | Combines FAST keypoints and BRIEF descriptors | Binary descriptor | Hamming distance | Fastest |

Key idea:

- SIFT is accurate but slower.
- SURF is faster and uses fewer descriptor dimensions.
- ORB is fastest and uses binary descriptors.

### 10. What are the three basic steps in feature detection?

The lecturer emphasized this as the common process behind many feature detectors:

1. Detect key points.
2. Compute descriptors.
3. Match features.

## CNN and Training Topics

### 11. Why are ANNs limited for image classification?

ANNs are inefficient for images because every pixel becomes an input neuron.

Example from lecture:

```text
1000 x 1000 RGB image = 1000 x 1000 x 3 = 3,000,000 input neurons
```

If connected to 1000 hidden neurons:

```text
3,000,000 x 1000 = 3,000,000,000 connections
```

Main limitations:

- Computationally heavy
- Too many parameters
- Longer training time
- Higher risk of overfitting

CNNs solve this by extracting only important spatial features using convolution and pooling.

### 12. What does each CNN layer do?

- **Convolutional layer:** Extracts important spatial features using filters.
- **Pooling layer:** Reduces feature map size and computation.
- **Flattening layer:** Converts multidimensional feature maps into a 1D vector.
- **Fully connected layer:** Combines features to make a classification decision.
- **Output layer:** Produces class scores/probabilities.

### 13. What are filter size, stride, and padding?

**Filter size** controls how much of the input the kernel sees at once.

- Small filters detect fine details.
- Large filters detect broader patterns.

**Stride** is how far the filter moves each step.

- Larger stride gives smaller output.
- Smaller stride keeps more spatial detail.

**Padding** adds zeros around the image border.

- Helps preserve edge information.
- Helps control output size.

### 14. What is pooling?

Pooling reduces the spatial size of feature maps.

Types:

- **Max pooling:** Takes the maximum value from each region.
- **Average pooling:** Takes the average value from each region.

Pooling reduces computation and helps the model generalize better.

### 15. What is an activation function?

An activation function introduces non-linearity so the network can learn complex patterns.

Common examples:

- **Sigmoid:** Often used for binary classification; can suffer from vanishing gradients.
- **Tanh:** Outputs values from -1 to +1; can also suffer from vanishing gradients.
- **ReLU:** Outputs `max(0, x)`; commonly used in hidden layers.
- **Softmax:** Used in multi-class output layers to produce class probabilities.

### 16. What is a loss function?

A loss function measures how far the model prediction is from the correct answer.

Common examples:

- **Mean Squared Error:** Often used for regression.
- **Binary Cross Entropy:** Used for binary classification.
- **Categorical Cross Entropy:** Used for multi-class classification.

The goal of training is to minimize loss.

### 17. What is backpropagation?

Backpropagation is the training algorithm used to update weights and biases.

Basic process:

1. Forward pass through the network.
2. Compute the loss.
3. Calculate error terms.
4. Propagate the error backward.
5. Update weights using an optimizer.
6. Repeat until loss is minimized.

Important point: backpropagation happens during training, not validation or testing.

### 18. What is overfitting, and how can it be reduced?

Overfitting happens when a model performs very well on training data but poorly on unseen data.

Signs:

- Training accuracy is high.
- Validation accuracy is much lower.
- Training loss decreases while validation loss increases.

Solutions:

- Data augmentation
- Dropout
- Regularization
- Early stopping
- Simpler model
- More diverse training data

### 19. What is data augmentation?

Data augmentation expands a dataset by creating transformed versions of existing images.

Examples:

- Rotation
- Scaling
- Flipping
- Cropping
- Zooming
- Shearing

It helps reduce overfitting and improves generalization.

### 20. What are training, validation, and test sets?

- **Training set:** Used to train the model.
- **Validation set:** Used during training to tune hyperparameters and detect overfitting.
- **Test set:** Used after training to evaluate final performance on unseen data.

## Evaluation Metrics

### 21. What is a confusion matrix?

A confusion matrix compares actual values with predicted values.

Terms:

- **True Positive:** Actual positive, predicted positive.
- **True Negative:** Actual negative, predicted negative.
- **False Positive:** Actual negative, predicted positive.
- **False Negative:** Actual positive, predicted negative.

### 22. Define accuracy, precision, recall, and F1 score.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 x (Precision x Recall) / (Precision + Recall)
```

Meaning:

- **Accuracy:** Overall correctness.
- **Precision:** How many predicted positives were actually positive.
- **Recall:** How many actual positives were found.
- **F1 score:** Balance between precision and recall.

### 23. What is ROC/AUC?

The ROC curve plots true positive rate against false positive rate.

Interpretation:

- Curve near top-left = good model.
- Curve on diagonal = random guessing.
- Curve below diagonal = poor model.

AUC summarizes the ROC curve as a single score.

## Object Detection

### 24. What is the difference between classification and object detection?

**Classification** answers:

```text
What object is in the image?
```

**Object detection** answers:

```text
What object is in the image, and where is it?
```

Object detection outputs:

- Class label
- Bounding box coordinates
- Confidence score

### 25. What are the limitations of traditional object detection?

Traditional methods such as sliding windows are limited because they:

- Scan many regions of the image
- Are computationally expensive
- Struggle with scale changes
- Struggle with lighting changes
- Require manually designed features/classifiers

This is why deep learning methods became important.

### 26. Explain backbone, neck, and detection head.

- **Backbone:** Extracts features from the input image.
- **Neck:** Combines features from the backbone.
- **Detection head:** Makes final predictions: class, bounding box, and confidence.

The lecturer emphasized that good feature extraction is critical, so choosing a good backbone matters.

### 27. Compare anchor-based and anchorless detection.

**Anchor-based detection** uses predefined bounding boxes with different sizes and aspect ratios. These boxes are adjusted during training to match ground truth.

Examples:

- Faster R-CNN
- SSD
- YOLO

**Anchorless detection** directly predicts object centers or corners without predefined boxes.

Examples:

- CornerNet
- CenterNet

Tradeoff:

- Anchor-based methods are usually more accurate.
- Anchorless methods are simpler and less computationally complex.

### 28. What is IoU?

IoU means **Intersection over Union**.

```text
IoU = Area of Overlap / Area of Union
```

It measures how much the predicted bounding box overlaps with the ground truth box.

Higher IoU means better prediction. The lecture mentioned that IoU above 0.5 is generally considered a good bounding box.

### 29. Compare R-CNN, SSD, and YOLO.

**R-CNN:**

- Uses selective search to generate around 2000 region proposals.
- Each region is cropped, resized, and passed through a CNN.
- Less exhaustive than sliding window, but still slow.

**SSD:**

- Single-stage detector.
- Uses multiple feature maps at different scales.
- Good for detecting objects of different sizes.

**YOLO:**

- Single-stage detector.
- Divides the image into a grid.
- Predicts boxes and classes in one forward pass.
- Very fast.

## Object Tracking

### 30. What is object tracking?

Object tracking follows objects across video frames.

Basic steps:

1. Detect the object.
2. Assign a unique ID.
3. Track the object across frames.

### 31. What is the difference between detection and tracking?

Detection works on an image or frame and identifies where objects are.

Tracking works across video frames and maintains object identity over time.

Tracking must handle:

- Movement
- Occlusion
- Similar-looking objects
- Identity switching
- Motion blur

### 32. Compare single object tracking and multiple object tracking.

**Single object tracking** tracks one object.

**Multiple object tracking** tracks many objects at the same time and maintains a unique ID for each.

MOT is harder because objects can overlap, disappear, reappear, or look similar.

### 33. Compare single-stage and two-stage trackers.

**Single-stage trackers:**

- Detection and tracking happen together.
- Faster.
- Better for real-time or edge devices.
- Can be less accurate with occlusion or crowded scenes.

**Two-stage trackers:**

- First detect objects.
- Then associate detections across frames.
- Slower but often more accurate.
- Better for crowded scenes or occlusion.

Examples:

- Single-stage: DeepSORT
- Two-stage: ByteTrack, OC-SORT

### 34. What is ByteTrack?

ByteTrack is a two-stage multi-object tracker.

Main idea:

- First match high-confidence detections.
- Then use low-confidence detections to recover objects that may have been missed.

This helps reduce missed objects and improves tracking accuracy.

## Sensors and Sensor Fusion

### 35. Compare CCD and CMOS sensors.

| Feature | CCD | CMOS |
|---|---|---|
| Image quality | Higher | Lower historically, improved now |
| Light sensitivity | Better in low light | Less sensitive |
| Power | Higher power use | More power efficient |
| Cost | More expensive | Cheaper |
| Shutter | Global shutter | Rolling shutter |

Important issue:

CMOS rolling shutter captures line by line, so moving objects can appear skewed or wobbly.

### 36. Compare camera, LiDAR, radar, and thermal sensors.

**Camera/optical sensor:**

- High resolution and color.
- Cost effective.
- Struggles with fog, rain, snow, and low light.

**LiDAR:**

- Builds accurate 3D point clouds.
- Useful for autonomous vehicles and mapping.
- Expensive and affected by weather.

**Radar:**

- Measures distance and speed.
- Works in darkness, fog, rain, and dust.
- Lower resolution than LiDAR.

**Thermal sensor:**

- Detects heat signatures.
- Works in darkness, smoke, and fog.
- Lower resolution than visible cameras.

### 37. What is sensor fusion?

Sensor fusion combines data from multiple sensors to improve accuracy and reliability.

The lecturer called this the **most important concept** in the sensor lecture.

Main idea:

One sensor may be limited, but multiple sensors together give a more complete understanding of the environment.

Example:

- Camera gives color and visual detail.
- LiDAR gives 3D distance/point cloud.
- Radar gives speed and distance in poor weather.
- Thermal gives heat signatures in darkness or smoke.

Together, they improve decision-making.

### 38. Why is sensor fusion important?

Sensor fusion provides:

- **Redundancy:** If one sensor fails or is wrong, another can compensate.
- **Complementary information:** Different sensors capture different types of data.
- **Resilience:** The system becomes more robust in bad conditions.
- **Better accuracy:** Combined data gives a more complete picture.

High-stakes systems such as autonomous vehicles should not rely on only one sensor.

### 39. What is the sensor fusion pipeline?

Basic pipeline:

1. Collect raw data from multiple sensors.
2. Preprocess each data stream.
3. Calibrate, reduce noise, and normalize.
4. Align and synchronize data in time and space.
5. Apply fusion algorithm.
6. Produce a unified dataset or decision.

Fusion algorithms can include:

- Weighted averaging
- Probabilistic fusion
- Kalman filters
- Neural networks

### 40. What are applications of sensor fusion?

Examples:

- **Autonomous vehicles:** Cameras, LiDAR, radar, and ultrasonic sensors provide 360-degree awareness.
- **Smartphones:** Accelerometer, gyroscope, and magnetometer improve orientation and location tracking.
- **Robotics:** Visual, tactile, and auditory data help robots interact with environments.
- **Drones:** GPS, inertial sensors, and cameras support navigation and stability.

## Quick Study Order

If time is limited, study in this order:

1. CNN architecture, convolution, output-size formula, pooling.
2. Morphological operations, especially opening vs closing.
3. Canny edge detection and thresholding.
4. Feature detection: SIFT, SURF, ORB, descriptors, matching.
5. Training concepts: loss, activation, backpropagation, overfitting.
6. Evaluation metrics: confusion matrix, accuracy, precision, recall, F1, ROC/AUC.
7. Object detection: classification vs detection, backbone/neck/head, IoU, SSD vs YOLO.
8. Object tracking: detection vs tracking, SOT vs MOT, single-stage vs two-stage, ByteTrack.
9. Sensors: CCD vs CMOS, sensor tradeoffs, sensor fusion.

## Memory Hooks

- **Opening opens/removes noise:** Opening removes small white specks.
- **Closing closes holes:** Closing fills small gaps or black holes.
- **Precision = predicted positives:** Of everything the model predicted positive, how much was correct?
- **Recall = real positives:** Of everything actually positive, how much did the model find?
- **Classification = what:** It gives the class.
- **Detection = what and where:** It gives the class and bounding box.
- **Tracking = where over time:** It follows object identity across frames.
- **Backbone = feature extractor:** It finds useful visual features.
- **Detection head = decision maker:** It predicts class, box, and confidence.
- **YOLO = one look:** It predicts in one forward pass.
- **Sensor fusion = compensate:** One sensor's weakness is covered by another sensor's strength.

## Exam-Style Practice Questions

### Short Answer

1. Explain why image processing is considered the core of machine vision.
2. List the five steps of Canny edge detection and explain double thresholding.
3. Compare simple thresholding and adaptive thresholding.
4. Explain why segmentation often produces a binary image.
5. Explain why contour detection needs binary input.
6. Compare SIFT, SURF, and ORB.
7. Explain why ANN is inefficient for image classification.
8. Draw and label the CNN architecture.
9. Explain the role of convolution, pooling, and fully connected layers.
10. Explain overfitting and give three ways to reduce it.
11. Define accuracy, precision, recall, and F1 score.
12. Explain the difference between classification and object detection.
13. Explain backbone, neck, and detection head.
14. Compare anchor-based and anchorless detection.
15. Explain IoU and why it is used in object detection.
16. Compare SSD and YOLO.
17. Explain the difference between object detection and object tracking.
18. Compare single object tracking and multiple object tracking.
19. Compare single-stage and two-stage trackers.
20. Explain sensor fusion and why it is important.

### Calculation Practice

1. Compute the output size for `N = 10`, `F = 3`, `P = 0`, `S = 1`.
2. Compute the output size for `N = 12`, `F = 5`, `P = 2`, `S = 1`.
3. Compute the output size for `N = 8`, `F = 3`, `P = 1`, `S = 2`.
4. Given a 2 x 2 block with values `4, 9, 2, 7`, compute max pooling.
5. Given a 2 x 2 block with values `4, 9, 2, 7`, compute average pooling.
6. Given `TP = 30`, `TN = 50`, `FP = 10`, `FN = 10`, compute accuracy.
7. Given `TP = 30`, `FP = 10`, compute precision.
8. Given `TP = 30`, `FN = 10`, compute recall.
9. Given overlap area `45` and union area `90`, compute IoU.
10. Given IoU `0.3`, explain whether the bounding box is good according to the lecture threshold.

### Scenario Practice

1. An image has uneven lighting. Which thresholding method should you choose and why?
2. A document scan has small white noise around letters. Which morphological operation should you use?
3. A binary object has small black holes inside it. Which morphological operation should you use?
4. A real-time edge device needs fast object tracking. Would you choose a single-stage or two-stage tracker?
5. A crowded scene has occlusion and similar-looking objects. Would you choose a single-stage or two-stage tracker?
6. An autonomous vehicle needs to work in fog, rain, and darkness. Why is one camera not enough?
7. A detector predicts many overlapping boxes for one object. What method or metric helps select the best one?
8. A model performs well on training data but poorly on validation data. What is happening and how do you fix it?

## Self-Test Checklist

Use this checklist before the exam. If you cannot explain an item without notes, review that section again.

- [ ] I can explain opening vs closing without mixing the order.
- [ ] I can draw the CNN architecture from memory.
- [ ] I can compute convolution output size using the formula.
- [ ] I can explain what stride, padding, and filter size do.
- [ ] I can compute simple max pooling and average pooling.
- [ ] I can list the Canny edge detection steps.
- [ ] I can explain double thresholding and hysteresis.
- [ ] I can compare global, adaptive, and Otsu thresholding.
- [ ] I can explain why segmentation gives a binary image.
- [ ] I can compare SIFT, SURF, and ORB.
- [ ] I can explain why CNNs are better than ANNs for image classification.
- [ ] I can define loss function, activation function, and backpropagation.
- [ ] I can explain overfitting and list fixes.
- [ ] I can compute accuracy, precision, recall, F1, and IoU.
- [ ] I can explain classification vs detection vs tracking.
- [ ] I can explain backbone, neck, and detection head.
- [ ] I can compare anchor-based and anchorless detection.
- [ ] I can compare R-CNN, SSD, and YOLO.
- [ ] I can compare SOT and MOT.
- [ ] I can compare single-stage and two-stage trackers.
- [ ] I can explain ByteTrack at a high level.
- [ ] I can compare CCD and CMOS.
- [ ] I can compare camera, LiDAR, radar, and thermal sensors.
- [ ] I can explain sensor fusion and give examples.

## Transcript Evidence

These are the strongest transcript signals used to build the review:

- Direct exam cue for opening and closing: `transcripts/lecture2_cleaned.txt:99`.
- Direct exam cue for drawing CNN architecture: `transcripts/lecture4_cleaned.txt:21` and `transcripts/lecture4_cleaned.txt:29`.
- Direct exam cue for convolution calculation: `transcripts/lecture4_cleaned.txt:41`.
- Direct cue for CNN output-size questions: `transcripts/lecture4_cleaned.txt:51`.
- Canny steps and threshold emphasis: `transcripts/lecture2_cleaned.txt:69`.
- Lecture 10 final recap of revision topics: `transcripts/lecture10_cleaned.txt:45` to `transcripts/lecture10_cleaned.txt:49`.
- Sensor fusion called the most important concept: `transcripts/lecture10_cleaned.txt:49`.
