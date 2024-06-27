# Bird Call Identification Using Deep Learning

This project aims to develop an accurate deep learning model to identify three bird species by their sounds: Brown Tinamou, Great Tinamou, and Cinereous Tinamou. The model processes audio files, extracts relevant features, and classifies the bird species present in the recordings.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features](#features)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Usage](#usage)
- [Streamlit Web Application](#streamlit)

## Introduction
Birds are indicators of environmental quality, and monitoring their presence and diversity is crucial for conservation efforts. This project involves creating a deep learning model to identify bird species from audio recordings, which can aid in biodiversity monitoring and conservation.

## Dataset
The dataset comprises audio recordings of three bird species: Brown Tinamou, Great Tinamou, and Cinereous Tinamou. The recordings were downloaded from the [Xeno-canto](https://xeno-canto.org/collection/species/all) website.

## Features
For audio signal classification, the following features are used:
- Spectrograms
- Mel Spectrograms
- Mel-Frequency Cepstral Coefficients (MFCC)

These features are extracted from the audio files to train the deep learning model.

## Model
A Convolutional Neural Network (CNN) is used for this classification task. The model architecture includes:
- Convolutional layers
- MaxPooling layers
- Dense layers
- Dropout layers

The model is trained to classify audio features into one of the three bird species.

## Training
1. Download recordings for each bird species.
2. Preprocess and augment data if necessary.
3. Convert raw audio files into features.
4. Split the dataset into training, validation, and test sets.
5. Train the model on the training set.
6. Validate and evaluate the model on the validation and test sets.

## Evaluation
The model's performance is evaluated using precision, recall, and accuracy metrics. A confusion matrix is also generated to analyze misclassifications.

## Prediction
The prediction function processes a user-input audio file, extracts features, and predicts the bird species using the trained model.

## Usage
To use the model for predicting bird species from an audio file:
1. Place the audio file in the appropriate directory.
2. Run the prediction script with the audio file path as an argument.
3. The model will output the predicted bird species.

## Streamlit Web Application
To use the pre-trained model and get the predictions:
1. Open the terminal in the same directory.
2. Run the command 'streamlit run app.py'.
