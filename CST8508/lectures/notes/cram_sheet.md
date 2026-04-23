# CST8508 Last-Minute Exam Cram Sheet

Use this for a fast final review. For details and worked examples, use `notes/potential_exam_questions_review.md`.

## Direct Exam Cues

### Morphology

```text
Opening = Erosion -> Dilation
Closing = Dilation -> Erosion
```

- **Opening:** Removes small white noise.
- **Closing:** Fills small holes or gaps.

### CNN Architecture

```text
Input -> Convolution -> Pooling -> Fully Connected -> Output
```

- **Convolution:** Extracts features using filters/kernels.
- **Pooling:** Downsamples feature maps.
- **Fully connected:** Combines features for classification.
- **Output:** Produces class probabilities or scores.

### CNN Output Size

```text
Output size = (N - F + 2P) / S + 1
```

- **N:** Input size
- **F:** Filter size
- **P:** Padding
- **S:** Stride

Always answer as dimensions, for example `3 x 3`, not just `3`.

## Image Processing

### Canny Edge Detection

Steps:

1. Noise reduction using Gaussian blur.
2. Gradient calculation.
3. Non-maximum suppression.
4. Double thresholding.
5. Edge tracking by hysteresis.

Double thresholding:

- Below lower threshold = discard.
- Above upper threshold = strong edge.
- Between thresholds = weak edge.
- Keep weak edge only if connected to strong edge.

### Thresholding

- **Simple/global:** One threshold for whole image.
- **Adaptive:** Different thresholds for local regions; useful for uneven lighting.
- **Otsu:** Automatically chooses threshold based on pixel distribution.

### Segmentation

Segmentation divides an image into meaningful regions. Its output is often a binary image/mask.

### Contours

Contours are closed boundaries around objects. They require binary input.

## Feature Detection

Three basic steps:

1. Detect key points.
2. Compute descriptors.
3. Match features.

| Detector | Key Idea | Descriptor | Matching | Speed |
|---|---|---|---|---|
| SIFT | DoG + gradients | 128-D | Euclidean | Slowest |
| SURF | Hessian + Haar wavelets | 64-D | Euclidean | Faster |
| ORB | FAST + BRIEF | Binary | Hamming | Fastest |

## CNN Training

### ANN Limitations

ANNs are inefficient for images because every pixel becomes an input neuron. A `1000 x 1000 x 3` RGB image has 3 million input neurons.

Main issues:

- Computationally heavy
- Too many connections
- Long training time
- Overfitting risk

### Activation Functions

- **Sigmoid:** Binary classification, 0 to 1.
- **Tanh:** -1 to 1.
- **ReLU:** `max(0, x)`, common hidden-layer activation.
- **Softmax:** Multi-class probabilities.

### Loss Functions

- **MSE:** Regression.
- **Binary cross entropy:** Binary classification.
- **Categorical cross entropy:** Multi-class classification.

### Backpropagation

```text
Forward pass -> Loss -> Backward pass -> Update weights -> Repeat
```

Backpropagation happens during training, not validation/testing.

### Overfitting

Overfitting means high training performance but poor validation/test performance.

Fixes:

- Data augmentation
- Dropout
- Regularization
- Early stopping
- Simpler model
- More diverse data

## Metrics

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 x (Precision x Recall) / (Precision + Recall)
```

- **Precision:** How many predicted positives were correct?
- **Recall:** How many real positives were found?
- **ROC/AUC:** Top-left curve is better; diagonal is random.

## Object Detection

### Classification vs Detection

- **Classification:** What object is present?
- **Detection:** What object is present and where is it?

Detection outputs:

- Class label
- Bounding box
- Confidence score

### Detection Architecture

```text
Backbone -> Neck -> Detection Head
```

- **Backbone:** Extracts features.
- **Neck:** Combines features.
- **Detection head:** Predicts class, box, and confidence.

### Anchor-Based vs Anchorless

- **Anchor-based:** Uses predefined boxes adjusted to ground truth.
- **Anchorless:** Predicts centers or corners directly.

### IoU

```text
IoU = Area of overlap / Area of union
```

IoU above `0.5` was treated as a good bounding box in lecture.

### R-CNN vs SSD vs YOLO

- **R-CNN:** Region proposals, more accurate than sliding window but slow.
- **SSD:** Single-stage, multi-scale feature maps.
- **YOLO:** Single-stage, grid-based, very fast.

## Object Tracking

### Detection vs Tracking

- **Detection:** Finds objects in a frame.
- **Tracking:** Follows objects across frames and maintains ID.

### SOT vs MOT

- **Single Object Tracking:** Tracks one object.
- **Multiple Object Tracking:** Tracks many objects and preserves unique IDs.

### Single-Stage vs Two-Stage Trackers

- **Single-stage:** Detection and tracking together; faster, lower accuracy.
- **Two-stage:** Detect first, associate later; slower, higher accuracy.

### ByteTrack

ByteTrack is a two-stage tracker that uses both high-confidence and low-confidence detections to reduce missed objects.

## Sensors and Sensor Fusion

### CCD vs CMOS

| Feature | CCD | CMOS |
|---|---|---|
| Image quality | Higher | Lower historically, improved now |
| Low light | Better | Worse |
| Power | More | Less |
| Cost | Higher | Lower |
| Shutter | Global | Rolling |

### Sensor Tradeoffs

- **Camera:** High-resolution color, but weak in fog/rain/low light.
- **LiDAR:** Accurate 3D point cloud, but expensive and weather-sensitive.
- **Radar:** Distance and speed, works in bad weather, but lower resolution.
- **Thermal:** Heat signatures, works in darkness/smoke/fog, but lower resolution.

### Sensor Fusion

Sensor fusion combines multiple sensors to improve accuracy and reliability.

Benefits:

- Redundancy
- Complementary data
- Resilience
- Better decisions

Pipeline:

```text
Raw data -> Preprocess -> Align/Synchronize -> Fuse -> Unified output
```

Examples:

- Autonomous vehicles
- Smartphones
- Robotics
- Drones

## Final Self-Test

- Can I draw CNN architecture?
- Can I compute CNN output size?
- Can I explain opening vs closing?
- Can I list Canny steps?
- Can I compare SIFT, SURF, and ORB?
- Can I explain overfitting and fixes?
- Can I compute accuracy, precision, recall, F1, and IoU?
- Can I compare classification, detection, and tracking?
- Can I compare SSD and YOLO?
- Can I explain sensor fusion and why it matters?

