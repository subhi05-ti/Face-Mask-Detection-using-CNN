# Face-Mask-Detection-using-CNN

This project is a deep learning-based face mask detection system using Convolutional Neural Networks (CNN).
It can detect whether a person is wearing a face mask or not in both images and real-time video streams.

---


## Overview

Due to the COVID-19 pandemic, wearing masks has become a necessity in many public spaces. 
This project aims to automate the detection of face masks using a CNN model built with TensorFlow/Keras and OpenCV for video processing.

---

##  Dataset

We used the [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
which contains images of people **with** and **without** masks.

### Dataset Split:
- Training Set
- Validation Set
- Testing Set

Images were resized to **128x128** for faster processing and better performance.

---

##  Model Architecture

- Input Shape: (128, 128, 3)
- 3 Convolutional Layers with ReLU activation
- MaxPooling Layers after each Conv Layer
- Dropout Layers for regularization
- Flatten Layer
- Dense Layers
- Output Layer: Softmax (2 classes: Mask, No Mask)

The model was compiled using:
- **Loss Function**: `sparse_categorical_crossentropy`
- **Optimizer**: `Adam`
- **Metrics**: `acc`

---

## Result
-  98.09% accuracy on training data.
-  93.25% accuracy on test data.
