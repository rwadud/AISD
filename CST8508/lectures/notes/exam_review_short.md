# CST8508 Exam Review

## Week 1 - Introduction to Machine Vision

### What is Machine Vision? / Applications

Machine vision is the field of teaching machines to interpret and understand visual data using sensors, image processing, and machine learning (especially deep learning). The rise of deep learning was the key turning point that drastically changed the field. Applications are everywhere: facial recognition on smartphones, barcode scanning and inventory management in retail, quality control on manufacturing assembly lines, self-driving cars (lane keeping, traffic sign detection, obstacle detection), healthcare diagnostics and robotic surgery, agricultural crop monitoring via drones, airport security, augmented reality in entertainment, and thermal fever detection during COVID-19.

### Basic Workflow

The basic workflow follows three steps: image acquisition (capturing images using sensors like CCD or CMOS), image processing (manipulating and enhancing images to extract meaningful information), and interpretation/action (making decisions based on the processed data, such as pass/fail on an assembly line).

## Week 2 - Image Processing

### Why Image Processing in MV

Image processing is the core of machine vision because without it, raw captured images are often too noisy, inconsistent, or complex for a system to analyze effectively. We need image processing for enhancement (reducing noise, improving contrast), feature extraction (identifying edges, corners, blobs), segmentation (dividing images into meaningful regions), object recognition, and measurement (precise dimensions for quality control).

### Key Stages

The nine key stages of image processing are acquisition, enhancement, restoration, morphological processing, segmentation, object recognition, representation and description, image compression, and colour image processing. Not all stages are needed for every application; you select a combination based on your use case.

### Canny Edge Detection

Canny edge detection is a multi-step algorithm known for its precision. First, noise is reduced using Gaussian blur. Then, gradients (magnitude and direction) are computed at each pixel using Sobel operators. Non-maximum suppression thins edges by keeping only local maxima along the gradient direction. Double thresholding classifies pixels as strong edges (above T2), weak edges (between T1 and T2), or suppressed (below T1). Finally, edge tracking by hysteresis keeps weak edges only if they are connected to strong edges.

### Different Types of Image Processing Techniques

Key image processing techniques include filtering/convolution (sliding a kernel across the image to alter pixels), blurring (averaging neighbouring pixels to reduce noise and smooth the image), sharpening (emphasizing contrast between adjacent pixels to highlight edges and details, using kernels with contrasting values), resizing/scaling (changing dimensions either by specifying exact size or by a scale factor), histograms (graphing pixel intensity distribution from 0-255, with bins grouping ranges together), thresholding (simple thresholding applies one global value to create a binary image; adaptive thresholding computes different thresholds for sub-regions, better for uneven lighting; Otsu's uses pixel distribution to find the optimal threshold), and morphological operations (erosion shrinks foreground by requiring all pixels under the kernel to be 1; dilation expands by requiring at least one pixel to be 1; opening is erosion then dilation and removes small white noise; closing is dilation then erosion and fills small holes/gaps).

## Week 3 - Segmentation, Feature Detection, and Matching

### Segmentation

Segmentation extracts objects from an image for further processing, and the output is always a binary image (zeros and ones). The binary image acts as a mask to the source image, retaining only what is needed. Segmentation is critical because many image processing algorithms require binary input.

### Global and Adaptive Thresholding

Global (simple) thresholding applies a single threshold to every pixel: values above become 1 (white), below become 0 (black). Adaptive thresholding divides the image into sub-regions, each with its own threshold computed from local neighbourhoods (mean or Gaussian weighted), making it far better for images with uneven lighting or shadows.

### Contour Object Detection

Contour detection is similar to edge detection but always produces a closed path enclosing an area of uniform intensity, making it useful for shape analysis and object boundary detection. Contours require binary input, so segmentation must be done first. OpenCV's findContours returns contour boundary points and an optional hierarchy (parent-child relationships).

### SURF, SIFT, ORB Detectors

SIFT (Scale Invariant Feature Transform) detects and describes local features invariant to scale and rotation, partially invariant to illumination changes. It works by creating octaves (downsampled versions of the image), applying Gaussian blur and computing Difference of Gaussians to find potential keypoints, eliminating low-contrast points and edge points, assigning orientation via gradient histograms, and producing a 128-dimensional descriptor vector per keypoint. SURF (Speeded Up Robust Features) is a faster alternative using integral images, Hessian matrix-based detection, and Haar wavelets for orientation, producing a 64-dimensional descriptor. ORB (Oriented FAST and Rotated BRIEF) is the fastest, using FAST corner detection for keypoints and BRIEF for binary descriptors (just 0s and 1s), matched using Hamming distance instead of Euclidean distance.

### Feature Matching

Feature matching identifies corresponding features between images. Euclidean distance is used with SIFT/SURF (smallest distance = best match), Hamming distance is used with ORB (count of mismatching bits between binary descriptors). Brute Force matching compares every descriptor against all others (exhaustive). FLANN is a faster approximate matching method. Applications include panoramic image stitching, motion tracking, and 3D reconstruction.

## Week 4 - CNNs and Performance Metrics

### Traditional Methods vs Neural Network / Limitation of ANN

Traditional methods like decision trees require pre-defined tabular features (ear shape, face shape, whiskers), which we do not have for raw image data. ANNs can process images but are computationally impractical: a 1000x1000 RGB image creates 3 million input neurons, and connecting each to 1000 hidden neurons yields 3 billion connections. ANN limitations for image classification are that they are computationally heavy (too many connections), prone to overfitting (too many input pixels), and require very long training times.

### CNN Architecture / CNN Layers

CNN architecture solves these problems by replacing the single hidden layer with specialized layers. The architecture is: Input Layer, then repeated Convolutional Layer and Pooling Layer blocks (feature extraction), then Fully Connected Layer (classification), then Output Layer. The convolutional layer applies learnable filters that slide across the input, extracting only important features (edges, textures, shapes) hierarchically (low-level edges early, high-level objects deeper), producing feature maps. The output size formula is: Output = (N - F + 2P) / S + 1, where N is input size, F is filter size, P is padding, and S is stride. The pooling layer downsamples feature maps to reduce spatial dimensions: max pooling takes the maximum value in each block, average pooling takes the mean. The fully connected layer flattens pooled feature maps into a 1D vector, then uses weight matrices and biases (the learnable parameters) to compute class scores. The output layer has neurons equal to the number of classes, using softmax activation to produce a probability distribution, with the highest probability being the prediction.

### Performance Evaluation Metrics / ROC Curve

The ROC curve plots True Positive Rate against False Positive Rate across all classification thresholds. The Area Under the Curve (AUC) summarizes overall model performance: a curve hugging the top-left corner is excellent, sitting on the diagonal is random guessing, and below the diagonal is worse than random. The confusion matrix shows actual vs. predicted values with True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN). Accuracy = (TP+TN)/(TP+TN+FP+FN). Precision = TP/(TP+FP). Recall = TP/(TP+FN). F1 Score = 2 x (Precision x Recall)/(Precision + Recall).

## Week 5 - Training CNNs

### Data Augmentation

Data augmentation artificially expands a dataset by applying transformations (rotations, scaling, flipping, cropping) to existing images, generating multiple variants from each original. This reduces overfitting by exposing the model to more variation and improves generalization to unseen data.

### Designing CNN Architecture

When designing a CNN architecture, consider the number and type of layers, filter sizes, stride, and activation functions. There is no perfect formula; it depends on task complexity (simple tasks need 5-10 layers, complex ones up to 500). The number of filters typically increases deeper in the network (e.g., 16, 32, 64) because deeper layers extract more complex features.

### Activation Functions

Activation functions introduce non-linearity so the network can learn complex patterns. Sigmoid outputs between 0 and 1, used for binary classification, but suffers from the vanishing gradient problem. Softmax outputs between 0 and 1 across all classes (probabilities sum to 1), used for multi-class classification. Tanh outputs between -1 and +1, useful for negative values, but also has vanishing gradient issues. ReLU outputs max(0, x), is computationally efficient, and is the standard for hidden layers.

### Loss Function

The loss function measures how far the model's predictions are from the true values. Mean Squared Error (MSE) computes the average of squared differences between predicted and expected outputs, used in regression. Cross Entropy Loss measures the difference between predicted probability distributions and actual labels: binary cross entropy for two classes, categorical cross entropy for multiple classes.

### Back Propagation

Back propagation is the supervised learning algorithm that optimizes weights and biases by minimizing prediction error. The process is: perform a forward pass to get predictions, calculate the loss (error between predicted and expected), compute the error term for each output neuron, propagate the error backward through hidden layers, apply the delta rule to adjust weights (delta W = learning rate x error term x input), and repeat until minimum loss is achieved. Back propagation only happens during training, not during validation or testing.

### Best Practices for Training CNN

Best practices for training include splitting data into training (80%), validation (10%), and test (10%) sets. Use validation to tune hyperparameters and detect overfitting. Apply early stopping when validation loss starts increasing while training loss continues decreasing. Periodically save model checkpoints. Monitor both training and validation loss/accuracy curves.

### Overfitting Solution

Overfitting occurs when the model learns training data too well (including noise) and performs poorly on unseen data. Solutions include dropout layers (randomly deactivating neurons during training), regularization (L1/L2), data augmentation, simplifying the model (fewer layers), and early stopping. Underfitting means the model cannot learn meaningful patterns; solutions include increasing model complexity, providing more diverse data, training longer, and improving preprocessing.

## Week 7 - PyTorch

### What is PyTorch?

PyTorch is an open-source machine learning library for Python developed by Facebook's AI Research Lab (FAIR), released in 2016. It is known for its flexibility, ease of use, and dynamic computation graph (define-by-run), which allows network changes on the fly without recompilation. It is extensively used in both academia and industry for computer vision, NLP, and more.

### Key Features

Key features include dynamic computation graphs (unlike TensorFlow's originally static graphs), strong GPU acceleration, deep Python integration with seamless compatibility with NumPy and SciPy, automatic differentiation (autograd), and support across all major cloud platforms (AWS, GCP, Azure).

### PyTorch vs TensorFlow

PyTorch vs TensorFlow: PyTorch uses dynamic graphs (define-by-run), is more user-friendly, easier to debug, and dominates in research. TensorFlow originally used static graphs (now supports eager execution in 2.0), has a steeper learning curve, offers extensive deployment options including TFLite for mobile/edge devices, and is stronger in production deployment. By 2021, PyTorch surpassed TensorFlow in research usage.

### Core Components

Core components are tensors (multidimensional arrays like NumPy's ndarray but with GPU support and gradient tracking), autograd (automatic differentiation engine that builds computational graphs and computes gradients via the chain rule), nn.Module (base class for defining layers, losses, and full models), and optimizers (SGD, Adam, RMSProp that abstract optimization algorithms).

### Tensor

Tensors are the fundamental data structure, generalizing scalars (rank 0), vectors (rank 1), matrices (rank 2), and higher-dimensional arrays. They can be moved to GPU with .cuda(), and gradient tracking is controlled via requires_grad. Setting requires_grad=True enables autograd to track operations for backpropagation.

### Neural Network Module / Building a Simple Neural Network

Building a neural network in PyTorch follows the Lego block analogy: define layers in __init__ (grabbing blocks), compose them in the forward method (assembling the function). The nn.Module class provides pre-built layers (Conv2d, Linear, MaxPool2d). Custom layers and losses also inherit from nn.Module, defining __init__ and forward (backward is handled automatically by autograd).

### Training Loop

The training loop iterates through a DataLoader, performs a forward pass, computes the loss, calls loss.backward() for gradients, and updates weights with optimizer.step(). Critically, optimizer.zero_grad() must be called each iteration to flush accumulated gradients. Evaluation uses torch.no_grad() to disable gradient tracking, saving memory during inference.

### Best Practices in PyTorch

Best practices include using GPU acceleration, properly splitting data, utilizing DataLoader for batching and shuffling, regularly saving model state (torch.save/load), keeping code modular, never initializing weights to all zeros, and debugging by first overfitting to a small dataset.

## Week 8 - Object Detection

### Limitation of Traditional Object Detection

Traditional object detection (sliding window) was limited by scale variations, lighting sensitivity, manual classifier development, poor adaptability, and heavy computational cost. It required scanning every position at multiple scales and aspect ratios, passing each patch through feature extraction (SIFT/HOG) and an SVM classifier, making it exhaustively expensive.

### Object Detection vs Classification

Object detection extends classification by predicting both what an object is (class label) and where it is (bounding box coordinates plus confidence score). A classification model outputs just a class label, while a detection model outputs a list of objects each with a label, bounding box, and confidence score.

### Detection Head - Types

The detection head determines how predictions are made. Anchor-based detection (e.g., Faster R-CNN, SSD, YOLO) uses predefined bounding boxes at various scales and aspect ratios that are adjusted during training to match ground truth; it provides higher accuracy. Anchorless detection (e.g., CornerNet, CenterNet) directly predicts object centers or corners without predefined boxes; it is simpler and less computationally complex but typically less accurate.

### SSD vs YOLO

SSD (Single Shot MultiBox Detector) uses a VGG-16 backbone with multiple feature maps at different scales (6 layers starting from 38x38), enabling detection of objects at various sizes. Initial layers detect smaller objects, deeper layers detect larger ones. It uses predefined default boxes (anchors) at each feature map cell and produces 8732 total predictions per image, filtered by non-maximum suppression.

YOLO (You Only Look Once) divides the input image into a grid (7x7 in v1) and predicts bounding boxes, confidence scores, and class probabilities for each cell in a single forward pass. It uses a 448x448 input and produces a 7x7x30 output tensor. YOLO is extremely fast (45 FPS) but early versions struggled with small objects since the final feature map has a large receptive field.

### Challenges in Object Detection

Key challenges in object detection include real-time processing demands, detecting small and occluded objects, varying lighting conditions, balancing precision and recall in crowded scenes, and significant computational resource requirements.

## Week 9 - Object Tracking

### Object Tracking vs Object Detection

Object tracking estimates and predicts the positions of moving objects across consecutive video frames, extending beyond detection by maintaining identity over time. The three main steps are: detect the object and draw a bounding box, assign a unique ID, and track the object across frames. Unlike detection (which works on single frames), tracking handles occlusion by predicting positions even when objects are temporarily hidden, and it can predict where an object will be in the next frame.

### Single vs Multiple Object Tracking

Single object tracking (SOT) monitors one object, facing challenges of changing appearance, scale variation, and occlusion. Multiple object tracking (MOT) simultaneously tracks many objects, managing their identities and interactions. MOT is more complex, requiring sophisticated algorithms to handle dynamic object counts, non-linear motion, similar appearances, identity consistency, and re-identification (ReID) when objects leave and re-enter the scene.

### Single Stage vs Multi Stage Object Trackers

Single-stage trackers (e.g., DeepSORT) perform detection and tracking simultaneously in one network pass, offering high speed suitable for real-time and edge device applications, but with lower accuracy. Two-stage trackers (e.g., ByteTrack, OC-SORT) separate detection and association into distinct phases: first detect all objects, then use association algorithms (Kalman filter for position prediction, Hungarian algorithm for matching, IoU for overlap measurement) to link detections across frames. Two-stage trackers achieve higher accuracy, especially in crowded or complex scenes, but are computationally heavier.

### Application of Multiple Object Tracking

Applications of MOT include urban traffic management and autonomous vehicles, retail customer tracking (including COVID-era occupancy counting), sports analytics for player and ball movement, and surveillance systems for security monitoring.

### ByteTrack

ByteTrack is a two-stage tracker that achieves high accuracy by associating every detection box, including low-confidence ones. Stage 1 matches high-confidence detections (above threshold) with existing tracklets. Stage 2 recovers objects missed in stage 1 by matching low-confidence detections using IoU and cosine similarity. A gating mechanism then filters redundant detections. This two-stage association ensures important objects are not missed.

### Tools for MOT Development

Tools for MOT development include deep learning frameworks (PyTorch, TensorFlow) for detection backbones, tracking toolkits (DeepSORT, FairMOT), annotation tools (CVAT, LabelBox), and benchmarking platforms (MOTChallenge, VOT).

## Week 10 - Sensors and Sensor Fusion

### Single Sensor vs Multi Sensor Analysis

Single sensor analysis uses one sensor type focused on specific data points. It is straightforward, simple, and cost-effective, best for non-critical systems. Multi-sensor analysis (sensor fusion) combines data from multiple sensors for a comprehensive understanding, providing redundancy (overlapping data reduces errors), complementary data (filling gaps), and resilience (handling individual sensor failures). The choice depends on system criticality.

### Sensor Fusion

Sensor fusion is the process of combining data from multiple sensors to improve accuracy and reliability of decision-making. It is not system redundancy but a union of capabilities: each sensor's strengths compensate for others' weaknesses. The fusion data pipeline involves raw data collection, preprocessing (calibration, noise reduction, normalization), data alignment and synchronization in time and space, fusion algorithms (weighted averaging, Kalman filters, neural networks), and finally a unified dataset.

### Application of Sensor Fusion

Applications of sensor fusion include autonomous driving (cameras, radar, LiDAR, ultrasonic sensors together providing 360-degree coverage for safe navigation in all conditions), smartphones (accelerometer, gyroscope, magnetometer fusion for GPS accuracy and orientation), robotics (combining tactile, visual, and auditory data), and drones (GPS, inertial, and camera fusion for precise navigation).

### Types of Sensors and Trade-offs / CCD vs CMOS

The main sensor types and their trade-offs are as follows. Cameras (optical sensors using CCD or CMOS) provide high-resolution colour images and are cost-effective, but are sensitive to rain, fog, and low light. CCD sensors produce higher quality images with less noise, are more light-sensitive (better in low light), use a global shutter (no motion artifacts), but consume more power and cost more. CMOS sensors are cheaper, more power-efficient, have improved greatly in quality, but use a rolling shutter (can cause skew/wobble in motion) and are less ideal for low light. LiDAR emits pulsed lasers to build high-resolution 3D point clouds, is highly accurate at long range, but is expensive, sparse on small objects, and affected by weather. Radar uses radio waves to measure distance and speed, works in all weather and darkness, is small and affordable, but has lower accuracy and resolution than LiDAR. Thermal sensors detect infrared radiation to visualize heat signatures, work in darkness, smoke, and fog, but have lower resolution and struggle when foreground and background temperatures are similar.
