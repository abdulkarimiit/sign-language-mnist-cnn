# 🤟 Sign Language MNIST - Real-Time ASL Alphabet Classifier

An end-to-end Computer Vision pipeline and Deep Learning model trained to recognize static American Sign Language (ASL) fingerspelling gestures ($28 \times 28$ grayscale images) across 24 distinct alphabet classes (A–Y, excluding 'J' and 'Z').

Built using **TensorFlow/Keras**, **OpenCV**, and **MediaPipe**, this repository includes the full training workflow, evaluation metrics, and a real-time webcam inference script.

---

## 📌 Project Overview

* **Dataset:** [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) (27,455 training samples, 7,172 testing samples).
* **Architecture:** Custom CNN with 3 Convolutional Blocks, Batch Normalization, Max Pooling, and Dropout regularization.
* **Preprocessing:** Dynamic data augmentation using Keras `ImageDataGenerator`.
* **Model Performance:**
  * **Test Accuracy:** `99.89%`
  * **Test Loss:** `0.0019`
  * **Macro F1-Score:** `0.9991`

---

## 📊 Model Architecture & Performance

### Convolutional Neural Network Structure
```text
Input (28, 28, 1)
 ├── Conv2D (32 filters, 3x3, ReLU) -> BatchNormalization -> MaxPooling2D -> Dropout (0.2)
 ├── Conv2D (64 filters, 3x3, ReLU) -> BatchNormalization -> MaxPooling2D -> Dropout (0.2)
 ├── Conv2D (128 filters, 3x3, ReLU) -> BatchNormalization -> Dropout (0.2)
 ├── Flatten
 ├── Dense (128 units, ReLU) -> BatchNormalization -> Dropout (0.4)
 └── Dense (25 units, Softmax)
