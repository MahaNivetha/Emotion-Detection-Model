#!/usr/bin/env python
# coding: utf-8

# Import modules
import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# Dataset directories
TRAIN_DIR = '/kaggle/input/face-expression-recognition-dataset/images/train/'
VALIDATION_DIR = '/kaggle/input/face-expression-recognition-dataset/images/validation'

# Create DataFrame for training data
def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)

# Create DataFrame for validation data
validation = pd.DataFrame()
validation['image'], validation['label'] = createdataframe(VALIDATION_DIR)

# Function to extract features from images
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode="grayscale")
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

# Extract features for training and validation data
train_features = extract_features(train['image'])
validation_features = extract_features(validation['image'])

# Normalize features
x_train = train_features / 255.0
x_validation = validation_features / 255.0

# Label encoding
le = LabelEncoder()
le.fit(train['label'])
y_train = to_categorical(le.transform(train['label']), num_classes=7)
y_validation = to_categorical(le.transform(validation['label']), num_classes=7)

# Build the model
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, validation_data=(x_validation, y_validation))

# Save the model
model_json = model.to_json()
with open("emotiondetector.json", 'w') as json_file:
    json_file.write(model_json)
model.save("emotiondetector.h5")

# Test the model on sample images
label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def ef(image):
    img = load_img(image, color_mode='grayscale')
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Test on sample images
sample_images = [
    '/kaggle/input/face-expression-recognition-dataset/images/train/happy/10039.jpg',
    '/kaggle/input/face-expression-recognition-dataset/images/train/sad/1001.jpg',
    '/kaggle/input/face-expression-recognition-dataset/images/train/angry/22.jpg',
    '/kaggle/input/face-expression-recognition-dataset/images/train/fear/10067.jpg',
    '/kaggle/input/face-expression-recognition-dataset/images/train/disgust/10646.jpg',
    '/kaggle/input/face-expression-recognition-dataset/images/train/neutral/10051.jpg'
]

for image_path in sample_images:
    print("Original image is of:", image_path.split('/')[-2])
    img = ef(image_path)
    pred = model.predict(img)
    pred_label = label[pred.argmax()]
    print("Model prediction is:", pred_label)
    plt.imshow(img.reshape(48, 48), cmap='gray')
    plt.show()
