# Multimodal Deepfake Detection

A Python-based academic prototype for analyzing the authenticity of facial images and voice samples.

The project contains two main detection components:

- A **Convolutional Neural Network (CNN)** for binary classification of facial images as real or fake.
- An **audio similarity pipeline** based on MFCC feature extraction and cosine similarity for comparing a test voice against a reference voice.

A **Tkinter-based graphical interface** was also developed to provide a simple way to interact with the image and audio analysis components.

| Project Type | Academic |
|---|---|
| Development Language | Python |
| Institution | University of Tabuk |
| Date | May 2025 |

---

## Project Overview

The increasing availability of AI-generated and manipulated media creates new challenges for digital identity, biometric verification, and cybersecurity.

This project explores a simple multimodal approach to media authenticity analysis by combining:

- **Facial Image Analysis**
- **Voice Similarity Analysis**

These components are used as a prototype for media authenticity verification.

The image component uses a custom CNN implemented with **PyTorch**.

The audio component extracts **13 MFCC features** from a reference and test voice, averages the features over time, and calculates their **cosine similarity**.

The project also includes a **Tkinter graphical interface** for selecting image and audio files and displaying the resulting analysis.

---

## Objectives

The main objectives of the project were to:

- Develop a CNN-based model capable of classifying facial images into real and fake categories.
- Prepare a dataset containing real and manipulated facial images.
- Train and evaluate the image classification model.
- Implement an image prediction script using the trained CNN.
- Develop an audio comparison method using MFCC features.
- Calculate similarity between a reference voice and a test voice.
- Provide a configurable similarity threshold for audio classification.
- Develop a simple desktop GUI using Tkinter.
- Explore the application of machine learning and biometric analysis to deepfake-related security problems.

---

## System Architecture

The project consists of two main processing pipelines.

### Image Pipeline

```text
Real / Fake Image Dataset
          |
          v
Image Loading
          |
          v
Resize to 128 x 128
          |
          v
Normalization
          |
          v
Train / Test Split
          |
          v
CNN Training
          |
          v
Saved Model
          |
          v
Image Prediction
          |
          v
Real / Fake
```

### Audio Pipeline

```text
Reference Voice + Test Voice
             |
             v
          Librosa
             |
             v
      MFCC Extraction
             |
             v
       13 MFCC Features
             |
             v
     Mean Feature Vector
             |
             v
      Cosine Similarity
             |
             v
    Threshold Comparison
             |
             v
Same Speaker / Likely Fake or Different Speaker
```

---

# Image Deepfake Detection

## Dataset Organization

The image training data is organized into two classes:

```text
data/
├── real/
└── fake/
```

The preprocessing code assigns:

- **Real → Label 0**
- **Fake → Label 1**

Images are loaded using **OpenCV** and resized to:

**128 × 128 pixels**

The resulting image arrays are combined into a single dataset before training.

---

## Image Preprocessing

The preprocessing implementation is located in:

```text
utils/preprocess.py
```

The preprocessing pipeline performs the following operations:

1. Reads images from the `real` directory.
2. Reads images from the `fake` directory.
3. Resizes each image to `128 × 128`.
4. Assigns the appropriate class label.
5. Combines both classes into the final dataset.

The training script then converts pixel values from:

```text
0–255
```

to:

```text
0–1
```

and rearranges the dimensions into PyTorch's expected:

```text
N × C × H × W
```

format.

---

## CNN Architecture

The image classification model is implemented in:

```text
model/cnn_model.py
```

The architecture consists of two convolutional blocks followed by fully connected layers.

```text
Input
  |
  v
Conv2D
3 → 16 filters
  |
  v
ReLU
  |
  v
MaxPooling
  |
  v
Conv2D
16 → 32 filters
  |
  v
ReLU
  |
  v
MaxPooling
  |
  v
Flatten
  |
  v
Linear
32 × 32 × 32 → 128
  |
  v
ReLU
  |
  v
Linear
128 → 2
  |
  v
Real / Fake
```

### Layer Summary

| Layer | Configuration |
|---|---|
| Conv2D | 3 → 16 filters |
| Activation | ReLU |
| Pooling | MaxPool2D |
| Conv2D | 16 → 32 filters |
| Activation | ReLU |
| Pooling | MaxPool2D |
| Flatten | Feature vector |
| Fully Connected | 32 × 32 × 32 → 128 |
| Activation | ReLU |
| Output | 128 → 2 classes |

The final layer contains two outputs corresponding to the two classes:

- **0 → Real**
- **1 → Fake**

---

## Model Training

The training pipeline is implemented in:

```text
main.py
```

The training process uses:

- **PyTorch**
- **Adam optimizer**
- **Cross-Entropy Loss**
- **Batch size of 32**
- **5 training epochs**
- **80/20 train-test split**

The optimizer is configured with:

```text
Learning Rate = 0.001
```

The implementation automatically selects:

```text
CUDA GPU
```

when available; otherwise it uses:

```text
CPU
```

During training, the loss is printed for each epoch.

After training, the model is evaluated on the test split and the calculated accuracy is printed.

The trained model parameters are saved as:

```text
saved_model.pth
```

---

## Image Prediction

Image inference is implemented in:

```text
predict.py
```

The prediction pipeline:

1. Loads the saved CNN weights.
2. Opens the input image using Pillow.
3. Converts the image to RGB.
4. Resizes it to `128 × 128`.
5. Converts it to a PyTorch tensor.
6. Runs the model in evaluation mode.
7. Selects the predicted class.

The output is:

```text
Real ✅
```

or:

```text
Fake ❌
```

### Example

```bash
python predict.py path/to/image.jpg
```

---

# Audio Analysis

The project also includes an audio comparison component.

The implementation is located in:

```text
voice_spoof_detector/
└── voice_spoof_detector.py
```

The audio pipeline uses:

- **Librosa**
- **NumPy**
- **SciPy**

The input audio is loaded at:

```text
16 kHz
```

The system extracts:

```text
13 MFCC coefficients
```

The MFCC values are averaged over time to produce a fixed-length feature vector.

The reference and test feature vectors are then compared using **cosine similarity**.

---

## Cosine Similarity

The similarity score is calculated as:

```text
similarity = 1 - cosine(reference_embedding, test_embedding)
```

The resulting value is then compared against a configurable threshold.

For example:

```bash
python voice_spoof_detector/voice_spoof_detector.py real.wav test.wav 0.3
```

The threshold can be supplied as the third command-line argument.

If the similarity is greater than or equal to the threshold, the program reports:

```text
Same Speaker (or very close)
```

Otherwise:

```text
Likely Fake or Different Speaker
```

> **Important:** This audio component is a similarity-based baseline. It is not a trained anti-spoofing classifier and does not by itself prove that a voice is AI-generated.

---

# Graphical User Interface

The project includes a **Tkinter-based interface** implemented in:

```text
inter.py
```

The interface provides:

- Username input
- Image file selection
- Image preview
- Voice file selection
- Audio similarity calculation
- Display of image/audio results

The application window is titled:

```text
AI Deepfake Verifier
```

The GUI provides a simple desktop interface for interacting with the project's media-analysis components.

---

# Evaluation

The academic project included testing using real and manipulated image and voice samples.

For image testing, the trained CNN was evaluated using real and AI-modified images.

For audio testing, reference and modified voice samples were compared using the implemented audio similarity pipeline.

The documented project examples include:

```text
Real image → Real
Fake image → Fake
```

and voice examples with different similarity scores.

The reported examples should be understood as project test cases, not as a comprehensive benchmark of real-world deepfake detection performance.

---

# Example Results

The project report documents the following example voice results:

| Test Case | Reported Result |
|---|---|
| Real voice | Same Speaker (Real) |
| Modified / spoofed voice | Different Speaker or Spoofed |

### Reported Example Similarity Scores

| Test | Similarity Score |
|---|---:|
| Real voice | 0.99 |
| Modified / spoofed voice | 0.1423 |

These values are examples from the project's documented tests and should not be interpreted as a universal classification threshold or production accuracy metric.

---

# Cybersecurity Relevance

This project is related to several cybersecurity and digital identity challenges.

Potential areas of application include:

- Biometric verification
- Identity protection
- Voice spoofing analysis
- Deepfake detection
- Digital media authenticity
- Fraud prevention
- AI-assisted impersonation defense

The project demonstrates how machine-learning and signal-processing techniques can be applied to a security-oriented media verification problem.

---

# Technologies

### Programming

- Python 3.13

### Machine Learning

- PyTorch
- Convolutional Neural Networks
- Scikit-learn

### Computer Vision

- OpenCV
- Pillow
- NumPy

### Audio Processing

- Librosa
- SciPy
- MFCC feature extraction
- Cosine similarity

### GUI

- Tkinter

---

# Project Structure

```text
multimodal-deepfake-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── main.py
├── predict.py
├── inter.py
│
├── model/
│   └── cnn_model.py
│
├── utils/
│   └── preprocess.py
│
├── voice_spoof_detector/
│   └── voice_spoof_detector.py
│
├── data/
│   └── README.md
│
├── screenshots/
│
└── docs/
    └── academic-project-report.pdf
```

---

# Limitations

Several improvements would be required before treating this as a production security solution.

### Dataset Size

The dataset used for the academic experiment is limited compared with large-scale real-world deepfake datasets.

### Generalization

Testing on a small set of examples cannot establish how the model performs against unseen manipulation techniques.

### Audio Analysis

MFCC similarity is a baseline signal-processing approach and is not sufficient by itself to reliably detect modern AI-generated speech.

### Image GUI

The current GUI image result is not connected to the trained CNN and requires integration with the prediction function.

### Reproducibility

The training script does not currently define a fixed random seed, so train/test splits and results may vary between executions.

---

# Future Improvements

Potential improvements include:

- Connect the GUI directly to the trained CNN inference function.
- Replace or extend the MFCC baseline with a dedicated speaker-verification / anti-spoofing model.
- Investigate ECAPA-TDNN integration.
- Increase dataset size and diversity.
- Add precision, recall, F1-score, and confusion matrix evaluation.
- Add reproducible random seeds.
- Evaluate on independent test datasets.
- Add real-time webcam input.
- Add microphone input.
- Improve GUI design and accessibility.
- Deploy the system as a web or mobile application.

---

# 📸 Screenshots

Selected screenshots from the academic project report are provided in the `screenshots/` directory.

The screenshots focus on:

- CNN architecture
- Model evaluation
- Dataset expansion
- Image prediction
- Voice interface
- Voice results

---

# Academic Documentation

The original academic report is available at:

```text
docs/academic-project-report.pdf
```

It provides additional background, project objectives, development documentation, screenshots, testing examples, and proposed future improvements.

---

# Disclaimer

This project was developed for academic and educational purposes.

It is a prototype and has not been validated as a production-grade deepfake detection or biometric security system.

The reported test results represent the examples evaluated during the project and should not be interpreted as a guarantee of performance on unseen real-world media.
