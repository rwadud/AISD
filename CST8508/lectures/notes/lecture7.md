# Lecture 7: Transfer Learning, Deep Learning Frameworks, and Object Detection Preview

## Model Architecture: Backbone, Neck, and Head

A neural network model for vision tasks is typically broken into three conceptual parts:

1. **Backbone**: the main feature extractor (e.g., ResNet-50, MobileNet).
2. **Neck**: an intermediate module that takes the features from the backbone, processes them, and passes them to the task head.
3. **Head**: the final module that performs the actual task, such as classification, object detection, or segmentation.

> **Key takeaway**: A neural network is just a stack of layers. You can conceptually place a layer as part of the backbone, neck, or head. The separation is useful for transfer learning and because libraries like OpenMMLab explicitly use these names.

### Why separate the neck from the backbone?

- In transfer learning, you can freeze the backbone layers but still train the neck and head.
- Different transfer learning strategies benefit from this separation: you could freeze all of the backbone and only train the head, then slowly start unfreezing layers.
- OpenMMLab uses this explicit naming convention (backbone, neck, head).
- In many libraries that load a classification model, the final layer is already called the **head**, so this terminology is ubiquitous.
- You could consider the neck as part of the head itself. It is a conceptual separation, not a strict architectural rule.

### How many layers can the neck have?

There is no hard limit. You can put as many layers as you want.

> **Analogy**: Think of neural networks as Lego blocks. Each layer is a Lego block, and you are just building. As long as it works, it is up to you how you build it.

**Historical note from the lecturer**: Models like ResNet-50 and MobileNet came about as graduate students sitting in a lab adding one more layer at a time. The process was informally known as **"graduate student descent"** because researchers would just sit and add one layer, then another, and publish whatever worked.

---

## Transfer Learning

### What is transfer learning?

**Transfer learning**: the practice of taking a model pre-trained on one dataset (usually a large, general dataset like ImageNet) and adapting its learned weights to a new task or dataset.

### Main steps in transfer learning

1. **Define your model architecture**. Ideally define it in terms of backbone, neck, and head. Your new model can be exactly the same as the pre-trained model, or different, with only some parts initialized from the pre-trained checkpoint.
2. **Take a pre-trained checkpoint** and extract the weights.
3. **Load the weights into your model** for each of the matching layers.
4. **Optionally freeze** some layers so they are not updated during training.
5. **Train the model** on your new task.

### The model as a dictionary

You can think of your saved model as a big dictionary, where the keys are the names of the layers.

```python
# Printing model parameters in PyTorch (reconstructed example)
import torch
import torchvision.models as models

model = models.resnet50(weights="IMAGENET1K_V2")
for name, param in model.named_parameters():
    print(name, param.shape)
# Example output:
# conv1.weight torch.Size([64, 3, 7, 7])
# bn1.weight torch.Size([64])
# bn1.bias torch.Size([64])
# layer1.0.conv1.weight torch.Size([64, 64, 1, 1])
# ...
```

You name your layers (e.g., `conv1`, `batch_norm1`, `relu1`), and with each layer you have all of its different parameters. Loading a checkpoint means loading this big dictionary into memory and matching keys between your model definition and the checkpoint.

> **Key takeaway**: If you see errors like **"keys do not match"**, it means some layer names in your model definition do not match the names saved in the checkpoint.

### Choosing how much to fine-tune

Your choice depends on three practical constraints:

1. **Data available**: small dataset? Fine-tune all at once, e.g., the cats and dogs example.
2. **Compute available**: low compute? Freeze all pre-trained layers, train only the head, then slowly unfreeze.
3. **Time available**: only five minutes? Just load the pre-trained checkpoint and run it as is.

### Continual learning vs. transfer learning

**Continual learning** and restarting training are almost similar to transfer learning.

**Scenario (from lecture)**: You trained your network for 10 epochs and decide you should have trained it for 50. You restart your Colab session, load the saved checkpoint, and restart training. You are essentially doing transfer learning, except you do not have to worry about key mismatches because you are loading the same model architecture.

**Important detail**: When restarting training you also need to load the **optimizer settings**. The optimizer settings include your learning rate, which is on a schedule, so you do not want to start again with a high learning rate at, say, iteration 420.

```python
# Saving and restoring optimizer state (reconstructed example)
# Save
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "scheduler_state_dict": scheduler.state_dict(),
}, "checkpoint.pth")

# Restore
ckpt = torch.load("checkpoint.pth")
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
scheduler.load_state_dict(ckpt["scheduler_state_dict"])
```

### Freezing layers

**Freezing a layer**: setting its weights as fixed so they are not updated during backpropagation.

**Does freezing skip the layer?** No. You still perform the **forward pass** through the frozen layer. What changes is the **backward pass**: you do not track gradients and you do not update those weights.

In PyTorch:

```python
# Freeze a layer
for param in model.layer1.parameters():
    param.requires_grad = False
```

### When do you need to fine-tune at all?

- **Similar data distributions**: If you trained on ImageNet and your new task is cats and dogs, the data distributions are very similar (ImageNet is basically houses, cars, animals, and so on). You can just use the pre-trained model as is and may not need to fine-tune.
- **Different data distributions**: If your task is x-ray images, ImageNet does not have many healthcare images. In this case, you want to further fine-tune the feature extractor.

### Summary of transfer learning

1. Define your new model.
2. Load your checkpoint.
3. Copy the weights from the checkpoint into your model definition.
4. Optionally freeze some layers.
5. Train.

---

## ResNet Refresher

### Why ResNet matters

ResNet stands for **Residual Network**. The main innovation is the **residual connection** (also called a skip connection).

> **Lecturer's anecdote**: When the ResNet paper came out, many people initially reacted with "this is so dumb, just one extra connection and they get a paper at NeurIPS." But it actually worked, so it became standard. NeurIPS is the main conference for deep learning.

> **Course note**: Based on the assignment, it seemed like many students did not really understand the architecture of ResNet. This section is a refresher. It is also important because when the lecture talks about dimensions later, knowing the architecture makes it clear.

### ResNet-50 forward pass, step by step

**Input**: RGB image resized to 224x224 with 3 channels.

**Layer 1: First convolution**
- Filter: 7x7, stride 2, 64 filters.
- Each of the 64 filters is 7x7x3 because there are 3 input channels.
- Convolution process: multiply the filter by the patch on each channel, then sum across channels to get one pixel value.
- That is the amount of compute per output pixel, which is why GPUs (with highly parallelizable processing) are used.
- Output size formula: take your input, pad it (the input size increases), then:

$$\text{Output size} = \frac{\text{Padded input size} - \text{Filter size}}{\text{Stride}} + 1$$

  Equivalently, $\text{Output} = \lfloor (\text{Input} + 2 \cdot \text{Padding} - \text{Filter}) / \text{Stride} \rfloor + 1$. For ResNet's first conv with padding of 3: $\lfloor (224 + 6 - 7)/2 \rfloor + 1 = 112$.

- Output: 112x112x64.

> **Practical note**: If someone told you to deploy ResNet-50, you would need to know the size of GPU you need, which means knowing how many parameters the model has so it fits into VRAM.

**Layer 2: Max pooling**
- No learnable parameters, just takes the max.
- Generally used to decrease spatial size by the stride.
- Here: 112 decreases by 2 to give 56.
- Output: 56x56x64.

**Stages of residual blocks**

Each stage contains multiple residual blocks, and a stage groups blocks at the **same spatial dimension**. For example, one stage has everything at 56x56, the next at 28x28, the next at 14x14, the next at 7x7.

**ResNet-50 stage dimensions** *(added, standard architectural values for the bottleneck version)*:

| Stage | Spatial Size | # Filters (output channels) |
|-------|-------------|-----------------------------|
| 1     | 56 x 56     | 256                         |
| 2     | 28 x 28     | 512                         |
| 3     | 14 x 14     | 1024                        |
| 4     | 7 x 7       | 2048                        |

As spatial dimension halves, the number of filters doubles. This keeps the total "information content" approximately constant.

> **Empirical justification**: Halving spatial resolution risks losing information, so they double the number of filters at each step. This is not strictly necessary. You could halve both and maybe get no difference in results. It is what one particular graduate student found and published, so everybody uses it.

### Anatomy of a ResNet bottleneck block

Each block performs:

1. **1x1 convolution (squeeze)**: reduces the number of channels.
2. **3x3 convolution**: the main spatial feature extractor.
3. **1x1 convolution (expand)**: increases the channels back up so the output matches the input dimensions.
4. **Skip connection (residual)**: add the block input to the block output.

```
Input ----------------------------------------+
  |                                           |
  v                                           |
[1x1 conv: squeeze channels]                  |
  |                                           |
  v                                           |
[3x3 conv: main feature extraction]           |
  |                                           |
  v                                           |
[1x1 conv: expand channels back]              |
  |                                           |
  +------ skip connection (add input) --------+
  |
  v
Output
```

### Why the 1x1 convolutions?

**What does a 1x1 convolution do?**
- A 1x1 filter is not taking into account any context around your pixels. It is squashing all your channels.
- You multiply the filter by each channel at a single pixel location, add them up, and get one value.
- A 1x1 conv combines the channels through a **sum**, not an average, producing one value per spatial location per output channel.

**Why sandwich a 3x3 between two 1x1 convolutions?**
- The **first 1x1** squeezes the channels (reduces compute load).
- The **3x3** then operates on the squeezed representation, taking spatial context into account.
- The **second 1x1** expands the channels back up so the block output can be added to the input through the skip connection. Tensors with different dimensions cannot be added, so the dimensions must match.

**Why squeeze first?**
- The 3x3 runs over a smaller, squeezed version of the feature map.
- The squeezing concentrates information so that the 3x3 (the main convolution that finds new features) operates on more information in less space.
- It is also easier for computation.

> **Student question**: When we squeeze, are we forcing the model to learn small, fine-grained details?
>
> **Answer**: Yes. Usually convolutions learn basic edges, but the model also learns fine-grained style details through the squeeze.

### Why the skip connection? The vanishing gradient problem

- In a very deep network (imagine a thousand layers), backpropagation computes gradients using the chain rule:

$$\frac{dY}{dX} = \frac{dY}{dL_n} \cdot \frac{dL_n}{dL_{n-1}} \cdots \frac{dL_1}{dX}$$

- If each gradient is, for example, 0.1, then multiplying many of them gives $0.1 \times 0.1 \times 0.1 = 10^{-3}$, and over many layers you get an extremely small number.
- This is the **vanishing gradient problem**, which creates very unstable learning.
- **Solution**: Pass the gradient backward through the block, but also give it a direct connection from the input via the skip. That way, the gradient does not vanish as much.

With the skip connection, the gradient path looks like:
- Direct path: gradient flows straight through the skip.
- Block path: gradient flows through the block's layers.
- The layer receives both a small gradient through the block and a direct gradient through the skip.

### Why the conv receptive field grows deeper in the network

**Purpose of stacking convolutions**: convolution detects features. As you stack more convolution layers:
- **Early layers**: detect edges.
- **Middle layers**: detect objects.
- **Deep layers**: specialized filters. Some detect eyes, some detect ears. They become **specialized feature detectors**.

> **Empirical rule**: The deeper you go, the more advanced the feature detectors become.

- Start with a large receptive field to find global features like big edges.
- Reduce spatial resolution as you go deeper to capture smaller, more detailed features.
- The halving of spatial resolution and doubling of filters at each stage keeps the total information approximately the same.

### When is the skip connection applied, forward or backward?

**Both**. The skip connection exists in the model architecture, so every forward pass routes input directly to the block output, and every backward pass routes gradient directly back. Skip connections are present at every stage.

---

## Deep Learning Framework Landscape

### Why are there so many libraries?

Each graduate student in each lab created their own libraries because at the time there was not a good environment. Some took off, some did not.

### Timeline of major frameworks

| Year | Framework | Origin | Notes |
|------|-----------|--------|-------|
| ~2012 | TensorFlow (originally another name) | Google | C++ backend, fast but hard to use |
| ~2016-2017 | PyTorch | Facebook (originally one developer over a few days) | Dynamic graphs, Pythonic |
| later | JAX | Google (partially out of Boston offices) | XLA backend, functional programming style |
| 2019 | TensorFlow 2.0 (TensorFlow Eager) | Google | Dynamic graphs, catching up to PyTorch |

### Static vs. dynamic graphs

- **Static graph (original TensorFlow)**: you create the graph, compile it, and it cannot be changed. Every forward pass reuses the compiled graph.
- **Dynamic graph (PyTorch, JAX, TensorFlow Eager)**: also called **eager execution**. The graph is recomputed on the fly as inputs change, which is much easier to use and debug.

> **Note**: Older course slides may describe TensorFlow as static. That refers to the original TensorFlow. Modern TensorFlow 2.0 supports dynamic graphs.

### Why was TensorFlow popular despite being hard to use?

- At the time, Python was slow.
- TensorFlow was written with a **C++ backend** and compiled Python into C++ under the hood, making it very fast.
- It had both Python and SQL APIs.
- Still useful for **edge devices** that need speed.

### JAX

- Follows similar principles to PyTorch but with slight differences.
- Written on **XLA** (Accelerated Linear Algebra).
- Based on **functional programming**.
- Used heavily at DeepMind.

### Where each framework dominates

| Company | Framework |
|---------|-----------|
| DeepMind | JAX |
| Google | TensorFlow |
| Facebook (Meta) | PyTorch |
| Open source | Mostly PyTorch |

> **Recommendation**: Mostly, it is good to just learn **PyTorch**.

### What is PyTorch, conceptually?

PyTorch is basically **autograd over a computational graph**. Whatever operations you set up form a big computational graph, and PyTorch automatically calculates the backward pass and gradients for you.

> **Fun fact**: You can write PyTorch on your own out of NumPy if you want, because it is basically a computational graph plus autograd.

---

## Higher-Level Libraries Built on Top of the Frameworks

These libraries emerged because people realized they were repeatedly writing training loops, checkpointing logic, and visualization code. So they built abstractions on top of the core frameworks.

### OpenMMLab

- Developed at the **Chinese University of Hong Kong (CUHK)**.
- Specific to **computer vision tasks**.
- Main thesis: **modularity**. You control everything through **config files** instead of writing code directly. The configs call the underlying code in the background.
- Has some support for TensorFlow and JAX, but mostly PyTorch.

**What is OpenMMLab (from the slides)**:

1. An open-source tool system for computer vision.
2. A big collection of state-of-the-art algorithms and datasets.
3. A unified programming framework for efficient model development.
4. A complete toolchain from model production to model deployment.

**Scale of the ecosystem**:
- 2000+ pre-trained models.
- 250+ algorithms.
- 20+ tasks.
- 1 framework.

**OpenMMLab design philosophy**: Provide **modular, reusable, and extendable** components for various computer vision tasks, from object detection to action recognition. This simplifies the learning curve so researchers and developers can focus on innovation rather than implementation details.

**Motivations**:
- **Unified interface**: reduces learning curve and development time. You apply similar methodologies across segmentation, detection, tracking, etc.
- **Modular composition**: users can **select and combine components** as needed, enhancing flexibility and efficiency. Toolboxes share a common framework, making it easy to **switch between tasks** or **integrate multiple functionalities** into a cohesive workflow.
- **Standardized codebases** solve the challenge of diverse source codes. Researchers often face difficulty integrating and comparing algorithms due to inconsistencies in implementation and documentation. OpenMMLab addresses this with well-documented, consistent codebases.
- **Community**: vibrant developer and researcher community. Extensive documentation, tutorials, and GitHub repositories for each toolbox. Discussion on forums like **Stack Overflow**. **Regular updates and contributions from users around the world** keep the toolboxes state-of-the-art and user-friendly.

**Architecture stack**:

| Layer | Components |
|-------|-----------|
| **Deployment** | MMDeploy |
| **Computer Vision Libraries** | MMPreTrain, MMDetection, MMDetection3D, MMRotate, MMSegmentation, MMPose, MMAction2, MMOCR, MMagic, MMYOLO, MMFlow, MMTracking, MMHuman3D, MMFewshot, 30+ others |
| **Foundational Libraries** | MMCV (neural network operators, data transforms), MMEngine (training engine, evaluation engine, module management) |
| **Deep Learning Framework** | PyTorch |

**Brief tour of the computer vision toolboxes**:

- **MMPretrain**: pre-trained models and **image classification**. Absorbed the earlier MMClassification functionality. Main entry point for transfer learning backbones.
- **MMDetection**: **object detection**, with Faster R-CNN, YOLO, SSD, and many others. Also includes segmentation models. Modular design enables easy customization.
- **MMDetection3D**: **3D object detection** for autonomous driving and robotics. Supports point cloud processing, multi-modality fusion, 3D bounding box detection, point cloud segmentation, and LiDAR-camera fusion.
- **MMRotate**: **rotation detection**. Detects objects at arbitrary orientations, useful for aerial imagery and scene text detection.
- **MMSegmentation**: **semantic segmentation** with U-Net, DeepLab, PSPNet, and others. Used for medical image analysis, autonomous driving, and geographic information systems.
- **MMAction2**: **action recognition and temporal action detection**. Supports 3D CNNs and temporal segment networks. Used for surveillance, human-computer interaction, and sports analysis.
- **MMTracking**: **video object tracking** for single and multiple objects. Tools for real-time tracking, motion analysis, and trajectory prediction. Used for sports analytics and surveillance.
- **MMDeploy**: deployment toolkit (see below).

### Hugging Face

- Initially started as an open-source **NLP library**. Originally just **tokenizers** and **Transformers**.
- Has grown into a big ecosystem:
  - **Hugging Face Hub**: upload models.
  - **Hugging Face Spaces**: upload apps.
  - **Hugging Face Datasets**: get datasets.
- Right now the ecosystem is moving toward Hugging Face for most things.
- Initially supported PyTorch, TensorFlow, and JAX, but **recently dropped TensorFlow and JAX**, so now only PyTorch is supported (some legacy support remains).
- Has a really good **diffusers** library if you are working on generative modeling like diffusion models.

> **Recommendation**: If you know Python, you kind of know Hugging Face. Learn the **Transformers library**, and get familiar with Hugging Face **Spaces**.

### Keras

- Started by **François Chollet** as a graduate student.
- Initially supported mainly TensorFlow. After Chollet got hired at Google, Keras became the **official abstraction on top of TensorFlow**.
- Recently started supporting PyTorch and JAX, but not everything is supported in those backends, so it is not fully usable for PyTorch or JAX yet.
- Use case: a better abstraction on top of TensorFlow, similar to how Hugging Face is on top of PyTorch.

### Relationship diagram *(reconstructed)*

```
             PyTorch              TensorFlow            JAX
                |                      |                 |
      +---------+---------+            |                 |
      |         |         |            |                 |
   OpenMMLab  HF        timm         Keras            (DeepMind)
   (vision)  (NLP,    (HF's         (abstraction
              vision)  torchvision  on TF, also
                       equivalent)  some PyTorch/JAX)
```

### Summary of the main libraries in deep learning right now

- **Core**: PyTorch, TensorFlow, JAX.
- **On top**:
  - OpenMMLab (PyTorch based).
  - Hugging Face (PyTorch based).
  - Keras (mostly TensorFlow, some PyTorch).

> **Student question**: You said Hugging Face is the most important. In other classes we did Keras and OpenMM. Why use each one?
>
> **Answer**: Keras was the new kid on the block around 2017 because it packaged everything together nicely. Since then, the landscape has shifted. Syllabi may not be updated because things move fast. The important thing is to have a good conception of how things work so that you can compare a new library when it comes out.

---

## Tech Stack You Should Know for Industry

> **Context**: The previous course slides had a theme on tech stack and Conda use. The lecturer found this lacking and put together additional material for anyone planning to work in startups in this field. This is the tech stack mostly used in industry.

### Environments and packaging

- **Conda / venv**: always create an **isolated environment** for downloading specific versions of Python, NumPy, SciPy, etc.
- **Why it matters**: as seen with OpenMM, it depends on old versions of Transformers and Python. It has not been updated because the professor who was maintaining it passed away and it became unmaintained. So every time you start with a new library, always create a virtual environment.
- **Why Conda specifically matters in CV**: allows isolation with specific versions of Python and libraries like TensorFlow and PyTorch, ensuring consistency and compatibility. Conda's environment management is especially crucial when working with complex frameworks like OpenMMLab, where dependency conflicts are common.

**Common Conda commands**:

```bash
# Install standard data libraries
conda install numpy pandas matplotlib

# Install OpenCV for image processing (conda-forge channel)
conda install -c conda-forge opencv

# Install TensorFlow via conda-forge
conda install -c conda-forge tensorflow

# PyTorch: visit the PyTorch website for the exact install command for your system
# (the install command depends on your CUDA version and OS)

# You can still use pip inside a conda environment
pip install some-package

# Export your environment for sharing or reproducibility
conda env export > environment.yml

# Recreate an environment from a YAML file
conda env create -f environment.yml

# List installed packages in the active environment
conda list

# List all conda environments on the machine
conda env list
```

**Advanced Conda notes**:
- Conda excels at managing complex dependency trees.
- **Channels** (like `conda-forge`) extend package availability beyond defaults.
- Resolve conflicts by **specifying explicit package versions**.

**Docker**: in industry, most projects are dockerized and put on cloud machines. Learn how to write Docker configs. Dockerize something and put it on an EC2 machine or any cloud machine.

**Kubernetes**: if applying for machine learning roles, know **Google Kubernetes** and **Docker**. Know how to architect different Docker images, set up multi-machine and multi-GPU configurations.

### Docker in depth

**Why Docker**: Docker provides portable, isolated environments. It uses **containers**, lightweight and standalone executable packages that run consistently across environments. This eliminates "works on my machine" problems and simplifies collaboration and deployment.

**Docker architecture**:

| Component | Pieces |
|-----------|--------|
| **Docker Client** | `docker build`, `docker pull`, `docker run` |
| **Docker Host** | Docker daemon, containers, images |
| **Registry** | Container repositories (e.g., Docker Hub) |

**Workflow**:
1. Write a **Dockerfile** defining the environment.
2. Build the Dockerfile into an **image** with `docker build`.
3. Run the image as a **container** with `docker run`.
4. This guarantees development, testing, and production environments are identical.

**Minimal Dockerfile** *(reconstructed example)*:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install uv && uv pip install --system -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

**Full MMDetection Dockerfile example** *(from slides)*:

```dockerfile
ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMEngine and MMCV
RUN pip install openmim && \
    mim install "mmengine>=0.7.1" "mmcv>=2.0.0rc4"

# Install MMDetection
RUN conda clean --all \
    && git clone https://github.com/open-mmlab/mmdetection.git /mmdetection \
    && cd /mmdetection \
    && pip install --no-cache-dir -e .

WORKDIR /mmdetection
```

**Docker Hub**:
- Cloud-based repository for managing Docker images.
- Hosts open-source images that you can use as the base for custom containers.
- `docker pull <image>` to fetch an image.
- `docker push <image>` to publish your own.
- Supports private repositories for confidential projects.

### Cloud platforms

Do not only learn AWS. Industry uses all three major clouds.

| Concept | AWS | GCP | Azure |
|---------|-----|-----|-------|
| Virtual machine | EC2 | Virtual Machines (Compute Engine) | Virtual Machines |
| Blob storage | S3 | Buckets (Cloud Storage) | Blob Storage |
| SQL | RDS | Cloud SQL | Azure SQL |
| NoSQL | DynamoDB | Firestore / Bigtable | Cosmos DB |

> **Recommendation**: Get a basic understanding of what each service does in each cloud.

### AWS EC2 for compute-intensive CV tasks

**Why EC2**: AWS EC2 provides scalable compute capacity in the cloud, ideal for compute-intensive computer vision tasks.

**Instance selection**:
- For **deep learning**, use **GPU-enabled instances** like the **P3 or G4 series** to significantly speed up model training.
- Scale compute resources up or down based on demand, making EC2 flexible and cost-effective for training, large dataset processing, and deployment.

**Storage**:
- **Elastic Block Store (EBS)**: fast block storage attached to instances.
- **Amazon S3**: blob storage for datasets and model artifacts.
- Efficient use of EBS and S3 is crucial for CV data pipelines.

**Networking**: AWS's networking features improve data transfer speeds and reduce latency, enhancing CV application performance.

**Scalability and cost management**:
- **Auto Scaling**: adjusts capacity automatically to maintain performance under varying workloads.
- **AWS Budgets** and **Cost Explorer**: monitor and optimize expenses.
- **Spot Instances**: offer significant cost savings for flexible workloads that can tolerate interruption.

### Code editors: move toward AI-assisted editors

- **VS Code** is fine, but everything is moving toward **AI code editors**.
- **Cursor** has similar functionality to VS Code, same key bindings, and strong AI features.
- Learn how to:
  - Set up your AI assistant (main rules file, instructions, when to use the composer).
  - Know when to take a suggestion and when not to.
  - Put in guardrails so the AI does not make changes across files when you only asked for a local change.
  - Set up **agentic computer use**, like OpenAI's **Operator**, giving instructions in a sandbox because these tools can go off the rails really badly.

> **Interview note**: Companies like Facebook have started moving toward **interviews that observe how you use code editors and work with AI**. This trend is important for job prospects.

**VS Code features worth knowing**:
- **IntelliSense**: code completion.
- **Python extension**: linting, testing, environment management.
- **Docker extension**: manage containers and images from the editor.
- **Git integration**: built-in version control.
- **Live Share**: real-time collaborative coding.
- **Remote Development plugins**: code on remote systems, Docker containers, or cloud servers.
- **Debugging tools**: breakpoints, call stack inspection, variable exploration.

**Debugging Python inside a Docker container with VS Code**:
- Use the **Remote - Containers** extension to attach VS Code to a running container.
- You get all of VS Code's debugging tools (breakpoints, call stack, variable inspection) inside the container.
- Lets you debug in an environment identical to production, which is especially valuable for CV workflows where the environment includes CUDA and complex native dependencies.

### Other tools to know

| Tool | Purpose |
|------|---------|
| **PDB** | Python debugger, learn how to put in checkpoints and debug Python code |
| **Poetry** | Standard dependency manager for Python apps |
| **uv** | Newer, faster alternative to Poetry and pip |
| **Terraform** | Infrastructure as code for one-click deployment |
| **FastAPI** | Newest Python web framework, used by most companies |
| **LangGraph / LangChain** | For building AI agents (these days everyone wants AI agent engineers) |

> **Key takeaway**: If you know all of these and name-drop them in interviews, you will most likely get the job because the interviewer thinks **"this person knows stuff."**

### What sets candidates apart in interviews

- Building an object detector is easy; anyone can download a model and claim success.
- Interviewers probe: **Where did you get stuck? How did you scale this?**
- Most AI code editors can write tutorial-level boilerplate. You need to know how to **debug, deploy, and scale**.
- **Setting up a code editor that is comfortable for you is a game changer** and really increases efficiency.

**Suggested practice project** *(additional example)*: take an object detector you built in an assignment, dockerize it, deploy it on a GCP virtual machine, expose it through FastAPI, and benchmark its latency. Document where you got stuck and how you improved speed. This kind of narrative beats "I trained a classifier."

---

## Transfer Learning in Each Framework

### PyTorch

**Loading a pre-trained model from torchvision:**

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Load a pre-trained ResNet-50
model = models.resnet50(weights="IMAGENET1K_V2")

# Replace the final FC layer for a new number of classes
num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Freeze all backbone parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last block (layer4) and the new FC
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True
```

**Loading a custom checkpoint that is not in torchvision:**

```python
# Define the architecture with weights=None, uninitialized
model = models.resnet50(weights=None)

# Load checkpoint file (.pth or .h5)
checkpoint = torch.load("custom_checkpoint.pth", map_location="cpu")

# Sometimes the checkpoint is a dict with extra keys, sometimes only a state dict
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Load, allowing mismatched keys
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
```

**What is strict=False?**
- Does not error on missing or unexpected keys.
- Useful for classification transfer, where the final FC layer is named differently or has a different shape.

### Three ways to load weights for non-trivial transfer

1. Use `load_state_dict(..., strict=False)`.
2. Go through each parameter and set specific ones' `requires_grad` to `False`, checking that it is not the final layer.
3. Go through each parameter manually and choose which layers to load, then set `requires_grad` appropriately.

> **Note**: When converting from a classification model to an object detection model, copying weights is not trivial. But in most cases you will not be doing this.

### Checking what is trainable

```python
# Print trainable parameters (those with requires_grad=True)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)
```

> **Student question**: Would this show layers requiring grad, or check missing keys?
>
> **Answer**: Trainable parameters only show those with `requires_grad=True`. For missing or unexpected keys, those come from the return value of `load_state_dict`. If you have a ResNet-50 model and load a ResNet-50 checkpoint, those lists should be empty. When transferring from classification to object detection, the `unexpected` list will contain the FC layer weights and biases, because the classification head is unexpected for the object detection model.

### OpenMM (OpenMMLab)

**OpenMM architecture**:
- **MMEngine**: the main engine. Training engine plus evaluation engine. Also handles module management.
- **MMCV**: foundational library providing neural network operators and data transforms.
- Specific task-oriented modules (see the full list earlier): **MMPretrain** (classification), **MMDetection** (object detection), **MMSegmentation** (segmentation), and many more like MMDetection3D, MMRotate, MMTracking, MMAction2.
- Each module has its own config. These configs reuse each other. MMDetection reuses checkpoints from MMPretrain.

> **Clarification from lecture**: MMPretrain is only for classification. For detection, use MMDetection. For segmentation, use MMSegmentation. You cannot use MMPretrain to do object detection.

**The Runner and Hooks**:
- The **runner** is the main entry point. It runs your entire training.
- Within training, there are **events**: `before_run`, `before_train`, `before_train_iter`, `after_train_iter`, `after_val_iter`, etc.
- Events are handled via **hooks**. You can write your own hook for each event.
- Each hook has a **priority**: very high, high, normal, or low (about four levels).
- In your log file, you see `before_run` with hooks defined at very high, normal, and low priority.

**How the runner uses hooks**:
1. The training engine calls the runner.
2. The runner looks at all hooks defined in the config and their priorities.
3. For each event (e.g., `before_run`), it collects hooks with that event and runs them in priority order.

**Writing your own custom hook** *(reconstructed example)*:

```python
# Custom hook that collects training and validation losses
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class LossCollectorHook(Hook):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.train_losses.append(outputs["loss"].item())

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.val_losses.append(outputs["loss"].item())
```

Then plug it into your config at normal priority, and you have your own hook for visualizing losses as training happens.

**Config highlights for transfer learning**:

```python
# Example MMDetection transfer learning config (reconstructed)
model = dict(
    backbone=dict(
        type="ResNet",
        depth=50,
        frozen_stages=1,  # Freeze the first stage
        init_cfg=dict(
            type="Pretrained",
            checkpoint="path/to/pretrained.pth",
            prefix="backbone.",  # Match only backbone keys
        ),
    ),
    neck=dict(type="FPN", ...),
    head=dict(type="RetinaHead", ...),
)
```

- **Prefix**: because the model has a backbone, neck, and head, you give the prefix accordingly when loading a checkpoint.
- **Frozen stages**: for ResNet (4 stages), you can set `frozen_stages=4` to freeze all stages, or a smaller number for more granular control.

> **Student question**: For the frozen stages, are there specific parameters I should keep, or is it my choice?
>
> **Answer**: It is your choice. For ResNet (4 stages), maybe freeze the first, then go more granular. Depends on the model and its conception of a stage. There may be better tools for finer control, but this is the standard knob.

### Hugging Face

Hugging Face is very similar to PyTorch. Hugging Face has **batteries-included tools** for specific things.

**Main pieces**:
- **Tokenizers** library.
- **Transformers** library (the main engine).
- **timm**: image processing, the Hugging Face equivalent of torchvision.
- **Trainer**: part of the Transformers library. You do not have to write the training loop, just call `trainer.train()`.

**Loading a pre-trained vision model**:

```python
import timm

# Load an existing pre-trained model
model = timm.create_model("resnet50", pretrained=True)
```

**Loading from a custom checkpoint**:

```python
import torch
import timm

# Create the architecture with weights not initialized
model = timm.create_model("resnet50", pretrained=False)

# Load the custom checkpoint and populate the weights
state_dict = torch.load("custom_checkpoint.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=False)

# Freeze layers as needed
for param in model.parameters():
    param.requires_grad = False
```

> **Student question**: `pretrained=False` vs. `pretrained=True`?
>
> **Answer**: `pretrained=True` loads an existing pre-trained model. `pretrained=False` creates the model with uninitialized weights, then you load your custom checkpoint into the uninitialized architecture.

**Why use PyTorch directly if Hugging Face exists?**
- **Custom losses**: you need to write your own loss function.
- **Non-trivial gradient operations**: taking gradients and using them in unusual ways.
- **Researching a new method** not yet in the Hugging Face library.
- Otherwise, just use Hugging Face.

**Hugging Face also has**:
- An **extensive optimizers library**, which matters more if you are doing NLP.

### Keras

Keras has two memorable APIs: **Sequential** and **Functional**.

- Every layer is a Lego block.
- You define layer 1 attached to layer 2 attached to layer 3, etc.
- The output of layer 1 feeds into layer 2.
- You can draw out your diagram and write the code in a very similar shape.

**Training and prediction in Keras**:

```python
# Keras Sequential API (reconstructed example)
from tensorflow import keras
from tensorflow.keras import layers

base_model = keras.applications.ResNet50(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False  # Shortcut: freeze all layers at once

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(2, activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
predictions = model.predict(x_test)
```

**Callbacks** exist for early stopping, checkpointing, etc.

**Keras shortcut for freezing**:

```python
base_model.trainable = False  # Freezes all previously trainable layers
```

You can do non-trivial things with Keras because there is not much abstraction between layers. Everything is fungible in all these libraries; you can write custom losses, custom data loaders, etc.

### Comparing all libraries side by side

| Task | PyTorch | Hugging Face | Keras | OpenMMLab |
|------|---------|-------------|-------|-----------|
| Pre-trained model library | `torchvision.models` | `timm` | TF Hub | MMPretrain |
| Load a pre-trained model | `models.<model_name>(weights=...)` | `timm.create_model(..., pretrained=True)` | Define model, set weights | Fetch a config |
| Load custom checkpoint | `load_state_dict` | `load_state_dict` (same as PyTorch) | `load_weights` | `load_weights` |
| Freeze layers | `requires_grad=False` per param | `requires_grad=False` per param | `base_model.trainable=False` (shortcut) | `frozen_stages=N` in config |
| Training loop | Write your own | `Trainer.train()` | `model.fit()` | `runner.train()` |
| Remove classification head | Shortcut exists (e.g., `include_top=False`) | Shortcut exists | Shortcut exists | Shortcut exists |
| Deployment | External | Push to Hugging Face Hub | TF model files (used by Google) | MMDeploy |

> **Conceptual takeaway**: Each library does the same thing through different routes. The whole point is that the ideas transfer between libraries.

### When to use which

- **PyTorch**: custom code and research, things not defined in the libraries.
- **Hugging Face**: easy for **fast iteration** when the model you need exists in the hub.
- **TensorFlow**: mostly dropped by industry. Use only for use cases requiring **really fast serving**, mobile, or edge devices, or small models. TensorFlow shines here because of **TFLite**.
- **OpenMM**: when you need a **computer vision specific framework**.

### MMDeploy: OpenMMLab's deployment toolkit

**MMDeploy** is an open-source toolset for deploying deep learning models from the OpenMMLab ecosystem to various platforms and devices.

**Components**:

1. **Model Converter**: converts training models from OpenMMLab into backend models that can run on target devices. Supports output formats like **ONNX** and **TorchScript**.
2. **MMDeploy Model**: the result package exported by the Model Converter. Includes backend models and model meta information used by the Inference SDK.
3. **Inference SDK**: developed in C/C++ with bindings for **Python, C#, and Java**. Wraps preprocessing, model inference, and postprocessing modules.

**Supported platforms**: Linux, Windows, macOS, Android.

**Supported inference backends**: ONNX Runtime, TensorRT, OpenVINO, CANN, Qualcomm, Rockchip.

**Accepted inputs** (the upstream OpenMMLab modules MMDeploy can consume): MMPretrain, MMDet, MMSeg, MMagic, MMOCR, MMDet3D, MMPose, MMRotate, MMAction2, MMYOLO.

**When to use MMDeploy**: deploying CV models in real-world applications where you need efficient inference across different hardware setups, especially mobile, embedded, or edge devices.

---

## Preview: Object Detection

### What is object detection?

**Object detection**: for each object in an image, predict **both** what it is (the label) **and** where it is (a bounding box).

- **Classification**: only tells you what is in the image.
- **Object detection**: tells you what is in the image and where it is.

Example: input is an image containing a cat, dog, bicycle, and bucket. The output for the cat is the label "cat" and a bounding box around it, giving XY coordinates.

**Bounding box representation**:
- Some models predict the four corner coordinates (e.g., $x_1, y_1, x_2, y_2$).
- Others predict a center point plus change in height and width (e.g., $x, y, w, h$).
- It depends on how the images were labeled and how the model is set up.

### Converting a classification model to an object detector

- For **classification**, using a ResNet-50 backbone: add an average pooling layer, flatten, then softmax across the 1000 classes.
- For **object detection**: you want both the label **and** the bounding box coordinates. You need non-trivial processing to produce both.

This is **why the neck matters**. The neck does non-trivial processing that can produce:
- The **label** of the bounding box.
- The **coordinates** of the bounding box.

> **Student question**: So besides the backbone, you have two heads, the classification head and the bounding box head?
>
> **Answer**: Yes, you need the classification head and the bounding box regression head.

### Three main approaches to object detection

Three approaches, developed historically in order of increasing sophistication:

1. **YOLO (You Only Look Once)** (*the lecturer noted not knowing exactly what YOLO stands for, and that the name might be a meme; YOLO is now on version 12 or so*).
2. **SSD (Single Shot Detector)**.
3. **Feature Pyramid Network (FPN)**.

### YOLO

**Thesis**: The convolutional backbone has learned nice feature detectors. Use the **very last feature map** (the 7x7 final layer in a ResNet-like backbone) as input to the object detection head.

**Architecture**:
- Take the whole backbone network.
- Get the output from the last feature map.
- Put it into your neck (defined as an object detector).
- Add a classification head.
- The classification gives you what is in each bounding box, producing labels.

**Problem with YOLO v1**:
- The final feature map is 7x7, so each cell has a **large receptive field**. It captures big features but misses tiny far-away objects.
- A tiny far-away dog or person is totally missed by YOLO v1.
- **Version 1 of YOLO detected everything from one feature map but missed all the tiny objects**.

### SSD (Single Shot Detector)

**Thesis**: Capture objects at different scales by using feature maps from multiple stages.

**Architecture**:
- Instead of using only the final layer, take the output from **each stage** of the backbone.
- Feed each of these feature maps into the object detector.

**Benefit**: You get feature maps at different resolutions, capturing both big and small objects.

**Problem with SSD**:
- Earlier layers have only learned **rudimentary features** (edges, basic patterns). They have good spatial resolution but do not see the rich semantic context.
- You do not get the complex context around your objects at high resolution.

### Feature Pyramid Network (FPN)

**Thesis**: Combine the **best of both worlds**. Get the rich semantic context of the final layer and the spatial resolution of earlier layers.

**Architecture**:
- For each layer, take the rudimentary feature map (earlier stage) and the final feature map.
- **Add them together** in a top-down fashion.

**Result**: You get both the **semantic detail** from the last layer and the **spatial resolution** of the early layers.

FPN is the standard for object detection right now. Many modern detectors use it, including **Faster R-CNN** and **Mask R-CNN**.

**Visual representation of the three approaches** *(reconstructed)*:

```
YOLO (v1)                 SSD                       FPN
----------------          ----------------          ----------------
                          Stage 4 -> detect         Stage 4 ----+
                          Stage 3 -> detect                     |
                          Stage 2 -> detect         Stage 3 <---+ (add)
                          Stage 1 -> detect                     |
Stage 4 -> detect                                   Stage 2 <---+ (add)
                                                                |
                                                    Stage 1 <---+ (add)
                                                     (all feed into detector)
```

### Comparison of the three approaches

| Approach | Feature maps used | Strength | Weakness |
|----------|------------------|----------|----------|
| YOLO v1 | Last feature map only | Simple, fast | Misses small/far-away objects |
| SSD | Feature maps at every stage | Captures multi-scale objects | Early-stage features lack semantic richness |
| FPN | Combines early and late features top-down | Both semantic richness and spatial resolution | More complex neck |

> **Course note**: The next lecture may cover only YOLO and SSD. Whether it covers Feature Pyramid Network is uncertain. FPN is the standard used in modern systems, so it is worth knowing regardless.

### Implementing an object detection model in MMDetection

**MMDetection workflow**:
- Reuses a classification checkpoint from MMPretrain.
- Defines a new **neck** (the Feature Pyramid Network) before the detection head.
- The FPN in MMDetection implements the underlying components of Faster R-CNN, YOLO, or similar detectors.

```python
# Simplified MMDetection config for transfer learning (reconstructed)
model = dict(
    type="FasterRCNN",
    backbone=dict(
        type="ResNet",
        depth=50,
        frozen_stages=1,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="pretrain/resnet50.pth",
            prefix="backbone.",
        ),
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),
    rpn_head=dict(...),  # Region proposal network
    roi_head=dict(...),  # Region of interest processing
)
```

### Alternative: PyTorch's torchvision detection

If you do not want to use MMDetection for the assignment, you can directly call a Faster R-CNN or other detector from **`torchvision.models.detection`**.

```python
# Use torchvision's Faster R-CNN (reconstructed example)
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()
# ready to run inference
```

> **Course note**: For the object detection assignment, you could take a pre-trained YOLO or Faster R-CNN and use it directly.

### Alternative: TensorFlow Object Detection API

The **TensorFlow Object Detection API** is another toolkit for building object detection models.

- Provides **pre-trained models** and multiple architectures.
- Lets you train **custom detectors** on your own data.
- Supports SSD, Faster R-CNN, and Mask R-CNN among others.
- Historically instrumental in advancing object detection research and applications.
- Usable as an alternative to MMDetection or torchvision's detection models if you are working in the TensorFlow ecosystem.

### The broader point

- You can take a **classification model** and convert it to an **object detection model** with non-trivial additions to the neck.
- You can even take a **text model** and convert it to an **image model**. It might not do worse because it has learned **semantics**, which is better than starting from scratch.

---

## Environment Troubleshooting for MMPretrain

> **Student question**: Is there a specific environment setup for MMPretrain? It has dependencies, and I want to match a standard configuration.
>
> **Answer**: Create a **separate environment** for MMPretrain with an **older version of Python** and an **older version of Transformers**. The library has not been updated since the maintainer passed away.

```bash
# Example (reconstructed) conda environment for MMPretrain
conda create -n mmpretrain python=3.8 -y
conda activate mmpretrain
pip install torch==1.13 torchvision==0.14
pip install transformers==4.25
pip install -U openmim
mim install mmengine
mim install mmpretrain
```

---

## Final Wrap-Up

> **Key takeaway**: As long as you understand **how to load the checkpoint**, **how to train it**, and **what the training loop looks like**, you have most of what is needed across all frameworks.

The basic building blocks of a neural network are the same across libraries. The differences are in the API surface, the level of abstraction, and the specific use cases each library targets. Mastering PyTorch gives you the foundation to understand all the other libraries, because each one either builds on PyTorch or uses the same conceptual operations.

**Putting the whole stack together**: OpenMMLab provides a comprehensive suite of tools covering many aspects of computer vision. Its modular design, combined with a supportive community, makes it a strong choice for both academic research and industrial applications. By combining **OpenMMLab** with **Conda** (environment isolation), **Docker** (reproducible containers), **VS Code** or an AI-assisted editor (development and debugging), and **AWS EC2** or another cloud (scalable compute), you can build, deploy, and scale advanced computer vision models more efficiently and effectively.
