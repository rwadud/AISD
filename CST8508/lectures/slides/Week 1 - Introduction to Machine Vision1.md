# Week 1 - Introduction to Machine Vision

---

## Course Overview

This course will take you on an exciting journey through the world of Machine Vision, combining theoretical knowledge with practical applications. Get ready to explore the path from pixels to perception!

> We are here right now. The field of CV is huge, we will hope to scratch the surface in this class.

---

## What is Machine Vision?

Machine vision is a technology that enables machines to **interpret** and **understand visual information** from the surrounding environment.

It involves **capturing images or videos** using cameras or sensors, **processing** these images to extract useful information, and then using this information to **make decisions** or perform specific tasks.

---

## History and Evolution of Machine Vision

The journey of Machine Vision began with simple image capturing devices, evolving through the digital revolution to sophisticated AI algorithms. Key milestones include the development of **digital cameras**, the rise of **neural networks**, and significant increases in **computing power**.

> What was the significant turning point? The rise of deep learning techniques.

---

## Key Technologies in Machine Vision

Essential technologies in Machine Vision include **image sensors** (like CCD and CMOS), **image processing techniques** (such as filtering and edge detection), and the role of Machine Learning, particularly **deep learning**, in modern Machine Vision systems.

---

## Applications of Machine Vision in Everyday Life

Machine Vision is all around us!

- In **retail**, it's used for barcode scanning and inventory management.
- In the **manufacturing** industry, it ensures quality checks on the assembly line.
- In **entertainment** it allows the detection and tracking of objects of interest, enhancing visual effects in movies and games.
- Lane keeping, blind spot checking, adaptive cruise, etc. in **autonomous vehicles**.
- Think about how these technologies affect our daily lives and the efficiency of businesses.

---

## Machine Vision in Advanced Applications

Machine Vision is pivotal in advanced fields.

- Autonomous vehicles use it for navigation
- Medical imaging uses it for diagnostics and surgical assistance
- Facial recognition systems enhance security and personalized experiences.

These applications highlight the versatility and transformative power of Machine Vision.

> Refer to Week 1 Asynchronous material

---

## Basic Workflow of a Machine Vision System

The workflow of a Machine Vision system can be summarized in three steps:

1. **Image Acquisition** (Capturing the image)
2. **Image Processing** (Analyzing and Manipulating the image)
3. **Interpretation/Action** (Making decisions based on the processed image).

For example, in an automated inspection system in manufacturing, this workflow ensures quality control.

---

## Introduction to Image Processing

**Image processing forms the core of Machine Vision.**

We deal with different Image Formats like JPEG, PNG, and RAW, and Color Spaces such as RGB and HSV.

Let's see a quick example of how to read an image using OpenCV.

---

## What is a Pixel?

### The Basic Unit of an Image

Think of a pixel as a numerical representation at location (x,y) in an image.

It is either a **single value** (in grayscale images representing the black intensity), or a **tuple of 3 values** representing the Red, Green, and Blue intensities.

Example pixel values at specific coordinates:

| Location (X,Y) | R   | G   | B   |
|----------------|-----|-----|-----|
| [1, 1]         | 221 | 135 | 129 |
| [200, 200]     | 148 | 48  | 72  |
| [77, 423]      | 222 | 128 | 112 |
| [446, 390]     | 163 | 102 | 101 |

---

## Transition to AI in Machine Vision

The evolution of AI, especially **Deep Learning**, has revolutionized Machine Vision. AI enhances its **accuracy** and enables the recognition of **complex patterns**.

In upcoming sessions, we'll dive deeper into how **CNNs** work and how they're applied in real-world scenarios.

---

## Convolutional Neural Networks - A Sneak Peek

Convolutional Neural Networks (CNNs) are a **class of deep neural networks**, most commonly applied to analyzing visual imagery. They are powerful for **automated feature extraction** in images, which we'll delve into in later lectures.

---

## Summary of Today's Lecture

- The basics of Machine Vision.
- Its history and key technologies.
- Real-world applications, from everyday uses to advanced fields.

Remember, the world of Machine Vision is as expansive as it is exciting, and you're just getting started!
