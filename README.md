# Emotion Detection Model Readme

## Overview

This repository contains code for building, training, and testing an emotion detection model using facial expression images. The model is built using TensorFlow and Keras and is designed to recognize seven different emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

## Files

1. **emotion_detection.ipynb:**
   - Jupyter Notebook containing the code for data preprocessing, model building, training, and testing.

2. **emotiondetector.json:**
   - JSON file containing the architecture of the trained emotion detection model.

3. **emotiondetector.h5:**
   - HDF5 file containing the trained weights of the emotion detection model.

## Dataset

The model is trained on the [Face Expression Recognition Dataset](https://www.kaggle.com/ananthu017/emotion-detection-fer). The dataset consists of facial expression images categorized into different emotions.

## Instructions

1. **Dependencies:**
   - Install the required dependencies by running `pip install -r requirements.txt`.

2. **Dataset:**
   - Download the Face Expression Recognition Dataset from Kaggle and extract the images into appropriate directories.
   - Set the `TRAIN_DIR` and `VALIDATION_DIR` variables in the notebook to the paths of the training and validation datasets.

3. **Run the Notebook:**
   - Execute the code in the `emotion_detection.ipynb` notebook to preprocess the data, build the model, and train it.

4. **Model Save:**
   - The trained model architecture is saved in `emotiondetector.json`, and the weights are saved in `emotiondetector.h5`.

5. **Test the Model:**
   - Use the sample images provided in the notebook to test the trained model's predictions.

## Model Architecture

The emotion detection model is a convolutional neural network (CNN) with the following architecture:

- Input Layer
- Convolutional Layer with 128 filters, kernel size (3, 3), and ReLU activation
- MaxPooling Layer (2, 2)
- Dropout Layer (0.4)
- Convolutional Layer with 256 filters, kernel size (3, 3), and ReLU activation
- MaxPooling Layer (2, 2)
- Dropout Layer (0.4)
- Convolutional Layer with 512 filters, kernel size (3, 3), and ReLU activation
- MaxPooling Layer (2, 2)
- Dropout Layer (0.4)
- Convolutional Layer with 512 filters, kernel size (3, 3), and ReLU activation
- MaxPooling Layer (2, 2)
- Dropout Layer (0.4)
- Flatten Layer
- Dense Layer with 512 units and ReLU activation
- Dropout Layer (0.4)
- Dense Layer with 256 units and ReLU activation
- Dropout Layer (0.3)
- Output Layer with 7 units and softmax activation (for the 7 emotion classes)

