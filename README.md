# 🐱🐶 Image Classification Using CNN

## Project Overview

This project demonstrates Image Classification using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained to classify images into two categories: **Cats** and **Dogs**.

The project introduces the fundamentals of Computer Vision and Deep Learning by teaching how CNNs automatically learn image features and perform classification.

---

## Project Objectives

* Load and preprocess image datasets
* Build a CNN architecture
* Train a deep learning model
* Evaluate model performance
* Save and reuse the trained model
* Predict the class of new images

---

## Technologies Used

* Python 3.12
* TensorFlow
* Keras
* NumPy
* Matplotlib
* VS Code

---

## Project Structure

Image_Classification_CNN/

├── dataset/

│ ├── cats/

│ └── dogs/

├── cnn_classifier.py

├── predict.py

├── cats_dogs_model.keras

├── test.jpg

└── README.md

---

## Dataset

The dataset contains images of:

* Cats
* Dogs

The images are organized into separate folders inside the `dataset` directory.

Example:

dataset/

├── cats/

│ ├── cat1.jpg

│ ├── cat2.jpg

│ └── cat3.jpg

└── dogs/

├── dog1.jpg

├── dog2.jpg

└── dog3.jpg

---

## CNN Architecture

The model consists of:

1. Rescaling Layer
2. Convolution Layer (Conv2D)
3. Max Pooling Layer
4. Flatten Layer
5. Dense Layer
6. Output Layer (Softmax)

---

## Model Training

Run the training script:

```bash
python cnn_classifier.py
```

The model is trained for 10 epochs and saved as:

```text
cats_dogs_model.keras
```

---

## Making Predictions

Place a test image in the project folder and name it:

```text
test.jpg
```

Run:

```bash
python predict.py
```

Example Output:

```text
Prediction: cats
```

or

```text
Prediction: dogs
```

---

## Learning Outcomes

Through this project, I learned:

* Image preprocessing techniques
* CNN architecture design
* Feature extraction using convolution layers
* Model training and evaluation
* Saving and loading trained models
* Image classification using TensorFlow

---

## Future Improvements

* Increase dataset size
* Add data augmentation
* Improve CNN architecture
* Use transfer learning models
* Support multiple image categories

---

## Author

**DHEEPAN P**

Mini Project – Image Classification Using Convolutional Neural Networks (CNNs)

