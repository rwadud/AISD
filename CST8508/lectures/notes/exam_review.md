# CST8508 Machine Vision - Exam Review

---

## Week 1: Introduction to Machine Vision

**What is Machine Vision?**
Teaching machines to interpret and understand the world through digital images/video. Powered by the rise of **deep learning**.

**Three Key Technologies:**
1. **Image Capturing** - cameras/sensors (CCD, CMOS)
2. **Image Processing** - feature extraction, edge detection, filtering
3. **Machine Learning** - especially deep learning / neural networks

**Applications:**
- Facial recognition (phones, security, airports)
- Retail: barcode scanning, inventory management
- Manufacturing: quality control on assembly lines
- Autonomous vehicles: lane keeping, obstacle detection, traffic sign detection
- Healthcare: tumor/fracture detection, robotic surgery
- Agriculture: crop monitoring (drones), harvest readiness, produce sorting
- Entertainment: visual effects, augmented reality
- COVID-19: people counting, fever detection (thermal cameras), mask compliance, social distancing

**Basic Workflow:**
1. **Image Acquisition** - capture via sensors/cameras
2. **Image Processing** - analyze, manipulate, enhance, prepare data
3. **Interpretation/Action** - make decisions based on processed data

---

## Week 2: Image Processing

**Why Image Processing in MV?**
It is the **core** of machine vision. Without processing, machines cannot extract meaningful information from raw images.

1. **Enhancement** - reduce noise, improve contrast/clarity
2. **Feature Extraction** - identify edges, corners, blobs
3. **Segmentation** - divide image into meaningful regions
4. **Object Recognition** - identify objects in images
5. **Measurement** - precise measurement of dimensions/distances (quality control)

**Key Stages (9 stages, use a combination as needed):**
1. Acquisition
2. Enhancement
3. Restoration
4. Morphological Processing
5. Segmentation
6. Object Recognition
7. Representation & Description
8. Image Compression
9. Colour Image Processing

**Canny Edge Detection - 5 Steps:**
1. **Noise Reduction** - apply Gaussian blur
2. **Gradient Calculation** - compute magnitude and direction using Sobel operators
   - Magnitude: G = sqrt(Gx^2 + Gy^2)
   - Direction: theta = arctan(Gy/Gx)
3. **Non-Maximum Suppression** - thin edges by keeping only local maxima
4. **Double Thresholding** - classify pixels:
   - \< T1 -> suppressed (not edge)
   - \> T2 -> strong edge
   - Between T1 and T2 -> weak edge
5. **Edge Tracking by Hysteresis** - keep weak edges only if connected to strong edges

```python
edges = cv2.Canny(image, threshold1=100, threshold2=200)
```

**Image Processing Techniques:**
- **Filtering/Convolution** - kernel slides over image, extracts features, reduces size
  - Output size = (Input - Kernel) + 1 (no padding, stride 1)
- **Blurring** - averaging pixels, reduces noise, smooths image (preprocessing step)
- **Sharpening** - enhances edges/details, uses contrasting kernel values
- **Resizing/Scaling** - resize to exact dimensions or by scale factor
- **Histograms** - graph of pixel brightness distribution (bins group intensity ranges)
- **Thresholding** - segment foreground from background:
  - **Simple**: single global threshold for all pixels
  - **Adaptive**: different thresholds per region (mean or Gaussian), handles uneven lighting
  - **Otsu's**: automatic threshold based on pixel distribution
- **Morphological Operations** (shape-based):
  - **Erosion** - shrinks foreground (pixel=1 only if ALL under kernel are 1)
  - **Dilation** - expands foreground (pixel=1 if ANY under kernel is 1)
  - **Opening = Erosion -> Dilation** (removes small white noise)
  - **Closing = Dilation -> Erosion** (fills small holes/gaps)
- **Transformations**: affine (preserves parallelism), translation, rotation, scaling, shearing

---

## Week 3: Segmentation & Feature Detection

**Segmentation:**
- Divides image into meaningful regions; output is always a **binary image** (0s and 1s)
- Binary image acts as a **mask** to the source image
- Thresholding is a key segmentation technique

**Global vs Adaptive Thresholding:**
- **Global (simple)**: one threshold for entire image; works for even lighting
- **Adaptive**: different thresholds per sub-region; handles varying illumination
  - Mean adaptive or Gaussian mean adaptive methods

**Contour Object Detection:**
- Similar to edge detection but contours always form a **closed path**
- Useful for shape analysis, object detection, recognition
- Requires **binary input** (segmentation first)
- `cv2.findContours` returns contours (boundary points) and hierarchy (parent/child)

**Feature Detectors - Three Main Steps:**
1. **Detect key points** (unique, important points)
2. **Compute descriptors** (unique representation per key point)
3. **Feature matching** (compare descriptors between images)

**SIFT (Scale Invariant Feature Transform):**
- Invariant to scale, rotation; partially invariant to illumination
- Uses Difference of Gaussian (DoG) for key point detection
- 128-dimensional descriptor (16x16 block -> 4x4 sub-blocks x 8-bin histograms)
- Matching via Euclidean distance
- Slowest of the three

**SURF (Speeded Up Robust Features):**
- Faster than SIFT, uses **integral images** and **Hessian matrix**
- Orientation via **Haar wavelets** (not gradient histograms)
- 64-dimensional descriptor (half of SIFT)
- Better illumination handling than SIFT

**ORB (Oriented FAST and Rotated BRIEF):**
- **Fastest**; combines FAST (key point detection via corners) + BRIEF (descriptors)
- **Binary descriptors** (0s and 1s) - efficient for computation
- Matching via **Hamming distance** (count mismatches between binary strings)
- Fewer but prominent key points

| | SIFT | SURF | ORB |
|---|---|---|---|
| Speed | Slowest | Faster | Fastest |
| Key Points | DoG | Hessian + integral images | FAST corners |
| Descriptor | 128-D vector | 64-D vector | Binary |
| Matching | Euclidean | Euclidean | Hamming |

**Feature Matching Methods:**
- **Brute Force** - compares every descriptor against all others (exhaustive, slower)
- **FLANN** - faster approximate matching

---

## Week 4: CNN

**Traditional Methods vs Neural Networks:**
- Decision trees require structured feature tables (ear shape, face shape, etc.)
- We don't have such structured data for images - we only have raw pixel data
- Neural networks can learn patterns directly from images

**Limitations of ANN for Image Classification:**
1. **Computationally heavy** - 1000x1000x3 image = 3M input neurons; with 1000 filters = 3 billion connections
2. **Overfitting** - too many input pixels leads to memorization
3. **Longer training time**

**CNN Architecture:**
```
Input -> [Conv Layer -> Pooling Layer] x N -> Fully Connected Layer -> Output
         |______ Feature Extraction ______|   |____ Classification ____|
```

**CNN Layers:**

**1. Convolutional Layer:**
- Applies filters/kernels to extract features (edges, textures, shapes)
- Each filter detects different features; hierarchical extraction (low -> mid -> high level)
- Output = **feature map**
- **Output Size Formula: (N - F + 2P) / S + 1**
  - N = input size, F = filter size, P = padding, S = stride
- Factors: filter size, stride (step size), padding (zeros around input to preserve edge info)

**2. Pooling Layer:**
- **Downsamples** feature maps, reduces spatial dimensions
- **Max Pooling**: takes maximum value in each block
- **Average Pooling**: takes average of each block
- Output = **pooled feature map**

**3. Fully Connected Layer:**
- **Flatten** multi-dimensional feature maps to 1D vector first
- Every neuron connected to every neuron in previous layer
- Trains via adjusting **weights** and **biases** (learnable parameters)
- Produces class scores -> highest score = prediction

**Activation Functions:**
- Introduce **non-linearity** (needed to learn complex patterns)
- **ReLU**: f(x) = max(0, x) - most common in hidden layers

**Output Layer:**
- Number of neurons = number of classes
- Uses **softmax** for multi-class classification
- Highest probability neuron = prediction

**Performance Evaluation Metrics - ROC Curve:**

| Metric | Formula | Meaning |
|---|---|---|
| **Accuracy** | (TP+TN) / (TP+TN+FP+FN) | Overall correctness |
| **Precision** | TP / (TP+FP) | Accuracy of positive predictions |
| **Recall** | TP / (TP+FN) | Ability to find all positives |
| **F1 Score** | 2 x (Prec x Rec) / (Prec + Rec) | Harmonic mean of precision & recall |

**ROC Curve**: plots True Positive Rate vs False Positive Rate
- Closer to top-left = better model
- On diagonal = random guessing
- **AUC** (Area Under Curve) summarizes overall performance

**Confusion Matrix**: Actual vs Predicted (TP, TN, FP, FN)

---

## Week 5: Deep Learning for Image Classification

**Data Augmentation:**
- Expands dataset from limited images using transformations: rotation, scaling, flipping, cropping
- Reduces overfitting by exposing model to varied features
- Improves generalization

**Designing CNN Architecture:**
- No perfect formula - trial and error
- Consider: number of layers, types, filter size, stride, activation function
- Simple tasks: 5-10 layers; Complex: up to 500
- Number of filters typically increases deeper in the network (16 -> 32 -> 64 -> ...)

**Activation Functions:**

| Function | Range | Use Case | Drawback |
|---|---|---|---|
| **Sigmoid** | (0,1) | Binary classification | Vanishing gradient |
| **Softmax** | (0,1) | Multi-class classification (output layer) | Computationally heavier |
| **Tanh** | (-1,1) | Negative inputs | Vanishing gradient |
| **ReLU** | [0, inf) | Hidden layers (most common) | Dead neurons |

**Loss Function:**
- Measures difference between predicted and actual output
- Goal: **minimize loss**
- **MSE** = (1/n) x sum of (expected - predicted)^2 (regression)
- **Cross Entropy** = used for classification (binary or categorical)

**Back Propagation:**
1. **Forward pass**: input -> layers -> output prediction
2. **Calculate error/loss**: compare prediction to expected
3. **Compute error terms**: for each output neuron
4. **Propagate error backward**: through hidden layers
5. **Apply delta rule**: adjust weights using learning rate x error x input
6. **Repeat** until minimum loss achieved

- Uses **gradient descent** to minimize loss: w_new = w_old - learning_rate x gradient
- Only happens during **training** (not validation/testing)

**Best Practices for Training CNN:**
- Use training (80%), validation (10%), and test (10%) splits
- Monitor both training and validation loss
- Apply **early stopping** when validation loss starts increasing
- Periodically save model state (checkpointing)
- Standard training step: forward pass -> compute loss -> backprop -> adjust weights (optimizer)
- **Hyperparameters** (set before training): learning rate, layers, batch size
- **Learnable parameters** (learned during training): weights, biases

**Overfitting Solutions:**
1. **Dropout layers** - randomly deactivate neurons during training
2. **Regularization** (L1/L2)
3. **Data augmentation** - expand dataset
4. **Simplify model** - reduce layers
5. **Early stopping** - halt when validation degrades

**Underfitting Solutions:**
- Increase model complexity (more layers)
- More diverse data
- Train longer
- Better feature extraction
- Re-examine preprocessing

---

## Week 7: PyTorch

**What is PyTorch?**
Open-source ML library for Python. Known for flexibility, ease of use, and **dynamic computation graphs**. Developed by Facebook (FAIR) in 2016. Most popular framework for AI research.

**Key Features:**
- **Dynamic computation graph** (define-by-run) - change network on the fly
- Strong **GPU acceleration** (CUDA support)
- Deep **Python integration** (seamless with NumPy, SciPy)
- Supports CPU, GPU, TPU, parallel processing
- Supported on AWS, GCP, Azure

**PyTorch vs TensorFlow:**

| Aspect | PyTorch | TensorFlow |
|---|---|---|
| Graphs | Dynamic (define-by-run) | Originally static (define-and-run) |
| Ease of use | More intuitive, Pythonic | Steeper learning curve |
| Debugging | Easier | More complex |
| Community | Dominant in research | Strong, less research-focused |
| Deployment | Growing (TorchScript) | Extensive (TFLite, mobile) |

**Core Components:**

| Component | Role |
|---|---|
| **Tensors** | Multi-dimensional arrays with GPU support (like NumPy arrays but track gradients) |
| **Autograd** | Automatic differentiation - computes gradients automatically |
| **Optimizers** | SGD, Adam, RMSProp - update weights |
| **nn.Module** | Base class for layers, losses, models |

**Tensor Ranks:**
- 0-D: scalar, 1-D: vector, 2-D: matrix, 3-D+: higher-dimensional

**Neural Network Module (nn.Module):**
- Base class for everything: layers, losses, models
- Custom layers/models define `__init__` (setup layers) and `forward` (define computation)
- No need to define `backward` - autograd handles it

**Building a Simple Neural Network:**
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*..., 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, ...)  # flatten
        x = self.fc1(x)
        return x
```

**Training Loop:**
```python
for epoch in range(num_epochs):
    optimizer.zero_grad()     # clear old gradients
    outputs = model(inputs)   # forward pass
    loss = criterion(outputs, labels)
    loss.backward()           # compute gradients
    optimizer.step()          # update weights
```

**Why zero gradients?** PyTorch accumulates gradients by default; must manually flush.

**Evaluation:** wrap in `torch.no_grad()` to skip gradient tracking (saves memory).

**Best Practices:**
1. Use GPU acceleration where possible
2. Properly split data (train/val/test)
3. Use DataLoader for batching
4. Regularly save/load models
5. Keep code modular
6. Debug by overfitting on small dataset first
7. Never initialize all weights to zeros

**Key Customization Patterns:**

| What | Methods to Define |
|---|---|
| Dataset | `__init__`, `__len__`, `__getitem__` |
| Layer | `__init__`, `forward` |
| Loss | `__init__`, `forward` |

---

## Week 8: Object Detection

**Limitation of Traditional Object Detection (Sliding Window):**
- Computationally very expensive (nested loops over scales, aspect ratios, positions)
- Poor handling of scale variations, lighting conditions
- Requires manual classifier development (SIFT/HOG + SVM)
- Poor adaptability

**Object Detection vs Classification:**

| | Classification | Detection |
|---|---|---|
| Output | Class label + probability | Class label + bounding box + confidence (per object) |
| Localization | No | Yes (where is the object) |
| Multiple objects | No | Yes |

**Detection Head Types:**

| | Anchor-Based | Anchorless |
|---|---|---|
| Approach | Predefined boxes adjusted to match ground truth | Directly predict corners/centers |
| Accuracy | Higher | Lower |
| Complexity | Higher | Lower |
| Examples | Faster R-CNN, SSD, YOLO | CornerNet, CenterNet |

**Architecture: Backbone -> Neck -> Detection Head**
- **Backbone**: feature extraction (ResNet, VGG)
- **Neck**: combines features from multiple scales
- **Detection Head**: predicts class + bounding box + confidence

**SSD (Single Shot MultiBox Detector):**
- Single-stage, anchor-based
- Uses **multiple feature maps at different scales** (6 layers, starting 38x38)
- VGG-16 backbone, 300x300 input
- Default boxes (anchors) at different aspect ratios
- Non-Maximum Suppression for final selection
- Detects objects at different scales (early layers = small objects, deeper = larger)

**YOLO (You Only Look Once):**
- Single-stage, grid-based
- Divides image into **grid cells** (7x7)
- Each cell predicts bounding boxes + confidence + class probabilities
- 448x448 input, custom architecture
- Single forward pass, 45 FPS
- Non-Maximum Suppression for final selection

**SSD vs YOLO:**

| | SSD | YOLO |
|---|---|---|
| Feature maps | Multiple scales | Single grid (7x7) |
| Input | 300x300 | 448x448 |
| Backbone | VGG-16 | Custom |
| Predictions | 8732 | 98 per class |

**IoU (Intersection over Union):**
- IoU = Area of Intersection / Area of Union
- \> 0.5 = good prediction; higher = better

**Challenges in Object Detection:**
- Real-time processing requirements
- Small object detection
- Occluded objects
- Varying lighting
- Balancing precision and recall
- Computational resource demands

---

## Week 9: Object Tracking

**Object Tracking vs Object Detection:**

| | Detection | Tracking |
|---|---|---|
| Input | Single image/frame | Consecutive video frames |
| Output | What + where | What + where + identity over time |
| Occlusion | Cannot handle | Can predict through occlusion |
| Prediction | No | Predicts next position |

**Three Steps of Tracking:** Detect -> Assign unique ID -> Track across frames

**Single Object Tracking (SOT):**
- Tracks one object only
- Challenges: changing appearance, scale, occlusion
- Lower complexity

**Multiple Object Tracking (MOT):**
- Tracks many objects simultaneously
- Must manage identities, handle interactions
- Challenges: similar appearances, crowded scenes, dynamic object count
- **Re-Identification (ReID)**: assigns same ID when object reappears after leaving scene
- Higher complexity, requires sophisticated algorithms

**Single-Stage vs Multi-Stage Trackers:**

| | Single-Stage | Two-Stage |
|---|---|---|
| Workflow | Detection + tracking simultaneously | Detection first, then association |
| Speed | Fast (real-time) | Slower |
| Accuracy | Lower | Higher |
| Best for | Edge devices, real-time apps | Crowded/complex scenes |
| Examples | DeepSORT | ByteTrack, OC-SORT |

**Applications of MOT:**
- Urban traffic management / autonomous vehicles
- Retail (customer tracking, people counting)
- Sports analytics (player/ball tracking)
- Surveillance and security

**ByteTrack:**
- Two-stage tracker with high accuracy
- **Unique approach**: associates ALL detection boxes, including low-confidence ones
- Step 1: Object detection (YOLO/Faster R-CNN)
- Step 2: Two-stage association:
  - Stage 1: Match **high-confidence** detections with tracklets
  - Stage 2: Recover missed objects from **low-confidence** detections (using IoU + cosine similarity)
- Step 3: Gating mechanism filters redundant detections
- Achieves superior tracking accuracy on standard benchmarks

**Tools for MOT Development:**
- **Frameworks**: PyTorch, TensorFlow
- **Tracking toolkits**: DeepSORT, FairMOT
- **Annotation**: CVAT, LabelBox
- **Benchmarking**: MOTChallenge, VOT
- **Other**: NVIDIA DeepStream, OpenCV

---

## Week 10: Sensors and Sensor Fusion

**Single Sensor vs Multi-Sensor Analysis:**

| | Single Sensor | Multi-Sensor (Fusion) |
|---|---|---|
| Complexity | Low | High |
| Cost | Low | Higher |
| Accuracy | Narrow, focused | Comprehensive |
| Reliability | Single point of failure | Redundant, resilient |
| Use case | Non-critical systems | Safety-critical (autonomous vehicles) |

**Sensor Fusion:**
The process of **combining data from multiple sensors** to improve accuracy and reliability of decision making. Not redundancy - it's a **union** of capabilities, not an intersection.

**Applications of Sensor Fusion:**
- **Autonomous vehicles**: cameras + LiDAR + radar + ultrasonic for 360-degree coverage
- **Smartphones**: accelerometer + gyroscope + magnetometer for location/orientation
- **Robotics**: tactile + visual + auditory data
- **Drones**: GPS + inertial sensors + cameras for navigation

**Types of Sensors and Trade-offs:**

| Sensor | Strengths | Weaknesses |
|---|---|---|
| **Camera (Optical)** | High resolution, color, cost-effective | Sensitive to rain/fog/lighting, poor in low light |
| **LiDAR** | 360-degree 3D point cloud, highly accurate, long range | Expensive, sparse data, affected by weather |
| **Radar** | All-weather, speed measurement, small/lightweight | Low resolution, interference, not 360-degree |
| **Thermal** | Works in darkness/fog/smoke, accurate long range | Lower resolution, struggles with low temp variance |
| **Depth** | 3D representation, distance measurement | Limited range/resolution depending on tech |

**CCD vs CMOS:**

| | CCD | CMOS |
|---|---|---|
| Image quality | Higher, less noise | Improved greatly, now comparable |
| Light sensitivity | More sensitive (better low light) | Less sensitive |
| Power consumption | Higher | More efficient |
| Cost | More expensive | Cheaper |
| Shutter | **Global** (all at once, no motion artifacts) | **Rolling** (line-by-line, can cause skew/wobble) |

**Sensor Fusion Pipeline:**
1. **Data Collection** from multiple sensors
2. **Preprocessing** - calibration, noise reduction, normalization
3. **Alignment & Synchronization** - match in time and space
4. **Fusion Algorithm** - weighted averaging, Kalman filters, neural networks
5. **Unified Dataset** - comprehensive environment understanding

**Challenges of Sensor Fusion:**
- Data variation (different formats, rates, resolutions)
- Timing and synchronization
- High computational demands
- Security and privacy concerns
