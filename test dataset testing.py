#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import Xception


# # Configuration Parameters 

# In[2]:


# Setting the height and width of the images in the dataset
IMAGE_HEIGHT , IMAGE_WIDTH = 128,128

# Setting the length of sequences (e.g., for video frames or time series data)
SEQUENCE_LENGTH = 30

# Specifying the path where the dataset is stored
DATASET_PATH = "dataset"

# Defining the class labels that represent different actions in the dataset
CLASS_LABELS = ["running","walking","handwaving"]


# # Video Frames Extraction

# In[3]:


# Function to extract frames from a video file
def frames_extraction(video_path):
    
    # List to store normalized frames
    frames_list  = []
    
    # Open the video file using OpenCV's VideoCapture
    video_reader = cv2.VideoCapture(video_path)
    
    # Get the total number of frames in the video
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the number of frames to skip for extracting SEQUENCE_LENGTH frames
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    # Iterate through each frame based on the skip_frames_window
    for frame_counter in range(SEQUENCE_LENGTH):
        
        # Set the position of the video reader to the current frame
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        
        # Read the current frame
        success, frame = video_reader.read()
        
        # Break the loop if there is an issue reading the frame
        if not success:
            break
            
        # Resize the frame to the specified dimensions    
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the pixel values to be between 0 and 1
        normalized_frame = resized_frame / 255
        
        # Append the normalized frame to the frames_list
        frames_list.append(normalized_frame)
    
    # Release the video reader
    video_reader.release()
    
    # Return the list of normalized frames
    return frames_list


# # Function to Create a Dataset from Video Files

# In[4]:


# Function to create a dataset from video files
def create_dataset():
    
    # Declared Empty Lists to store the features, labels and video file path values.
    features = []
    labels = []
    video_files_paths = []
    
    # Iterate through each class in CLASS_LABELS
    for class_index, class_name in enumerate(CLASS_LABELS):
        print(f'Extracting Data of Class: {class_name}')
        
        # List all files in the class directory
        files_list = os.listdir(os.path.join(DATASET_PATH, class_name))
        
        # Iterate through each file in the class directory
        for file_name in files_list:
            
            # Construct the full path to the video file
            video_file_path = os.path.join(DATASET_PATH, class_name, file_name)

             # Extract frames from the video using the frames_extraction function
            frames = frames_extraction(video_file_path)

             # Check if the number of frames extracted matches the specified sequence length
            if len(frames) == SEQUENCE_LENGTH:
                 # Append the frames, class index, and video file path to the respective lists
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    # Convert lists to NumPy arrays for better compatibility
    features = np.asarray(features)
    labels = np.array(labels)  
    
    # Return the features, labels, and video file paths
    return features, labels, video_files_paths


# In[5]:


# Call the create_dataset function and store the results in variables
features, labels, video_files_paths = create_dataset()


# In[6]:


def process_video_file(video_path, model):
    frames_list = frames_extraction(video_path)

    if len(frames_list) == SEQUENCE_LENGTH:
        input_sequence = np.expand_dims(frames_list, axis=0)  # Add batch dimension
        predictions = model.predict(input_sequence)
        predicted_label_index = np.argmax(predictions)
        predicted_label = CLASS_LABELS[predicted_label_index]

        print(f"Predicted activity for video {video_path}: {predicted_label}")

        # Display the video
        display_video(video_path)
    else:
        print(f"Skipping video {video_path} due to insufficient frames.")

def display_video(video_path):
    cap = cv2.VideoCapture(video_path)

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Display the frame
            cv2.imshow('Video', frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
# Load the saved model for testing
loaded_model = tf.keras.models.load_model("VGG19_model2.keras")


# In[7]:


# Convert integer labels to one-hot encoded labels
one_hot_encoded_labels = to_categorical(labels)
# Split the Data into Train ( 75% ) and Test Set ( 25% ).
features_train, features_test, labels_train, labels_test = train_test_split(features, 
                                                                            one_hot_encoded_labels, 
                                                                            test_size = 0.25, 
                                                                            shuffle = True, 
                                                                            random_state = 106)
# Set features and labels to None to free up memory if needed
features = None
labels = None


# In[8]:


# Initialize dictionaries to store counts of correct predictions and total samples for each label
label_correct_counts = {label: 0 for label in CLASS_LABELS}
label_total_counts = {label: 0 for label in CLASS_LABELS}

# Iterate through each sample in the test dataset
for i in range(len(features_test)):
    
    # Make predictions using the trained model for the current sample
    predicted_label = np.argmax(loaded_model.predict(np.expand_dims(features_test[i], axis=0))[0])
    
    # Get the actual label from the test labels
    actual_label = np.argmax(labels_test[i])
    
    # Increment the total count for the actual label
    label_total_counts[CLASS_LABELS[actual_label]] += 1
    
    # Check if the predicted label matches the actual label
    if predicted_label == actual_label:
        # Increment the correct count for the actual label
        label_correct_counts[CLASS_LABELS[actual_label]] += 1

# Calculate and print the accuracy for each label
print("Accuracy for each label:")
for label in CLASS_LABELS:
    label_accuracy = (label_correct_counts[label] / label_total_counts[label]) * 100
    print(f"{label}: {label_accuracy:.2f}%")


# In[9]:


from sklearn.metrics import precision_score, recall_score

# Make predictions for all test samples
predicted_labels = np.argmax(loaded_model.predict(features_test), axis=1)
actual_labels = np.argmax(labels_test, axis=1)

# Calculate precision and recall
precision = precision_score(actual_labels, predicted_labels, average='macro')
recall = recall_score(actual_labels, predicted_labels, average='macro')

# Print precision and recall
print("Precision =", precision)
print("Recall =", recall)


# In[10]:


from sklearn.metrics import classification_report

# Get the predicted labels for the test features
predicted_labels = np.argmax(loaded_model.predict(features_test), axis=1)

# Get the true labels from one-hot encoded format
true_labels = np.argmax(labels_test, axis=1)

# Generate a classification report
report = classification_report(true_labels, predicted_labels, target_names=CLASS_LABELS)

# Print the classification report
print(report)


# In[17]:


# Provide the path to the test video file
test_video_path = "shahar_run.avi"

# Test the model on the provided video file
process_video_file(test_video_path, loaded_model)


# In[15]:


def process_video_file(video_path, model):
    frames_list = frames_extraction(video_path)

    if len(frames_list) == SEQUENCE_LENGTH:
        input_sequence = np.expand_dims(frames_list, axis=0)  # Add batch dimension
        predictions = model.predict(input_sequence)
        predicted_label_index = np.argmax(predictions)
        predicted_label = CLASS_LABELS[predicted_label_index]

        print(f"Predicted activity for video {video_path}: {predicted_label}")

        # Display the video
        display_video(video_path)
    else:
        print(f"Skipping video {video_path} due to insufficient frames.")

def display_video(video_path):
    cap = cv2.VideoCapture(video_path)

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Display the frame
            cv2.imshow('Video', frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
# Load the saved model for testing
loaded_model1 = tf.keras.models.load_model("InceptionV3_model3.keras")


# In[16]:


# Initialize dictionaries to store counts of correct predictions and total samples for each label
label_correct_counts = {label: 0 for label in CLASS_LABELS}
label_total_counts = {label: 0 for label in CLASS_LABELS}

# Iterate through each sample in the test dataset
for i in range(len(features_test)):
    
    # Make predictions using the trained model for the current sample
    predicted_label = np.argmax(loaded_model1.predict(np.expand_dims(features_test[i], axis=0))[0])
    
    # Get the actual label from the test labels
    actual_label = np.argmax(labels_test[i])
    
    # Increment the total count for the actual label
    label_total_counts[CLASS_LABELS[actual_label]] += 1
    
    # Check if the predicted label matches the actual label
    if predicted_label == actual_label:
        # Increment the correct count for the actual label
        label_correct_counts[CLASS_LABELS[actual_label]] += 1

# Calculate and print the accuracy for each label
print("Accuracy for each label:")
for label in CLASS_LABELS:
    label_accuracy = (label_correct_counts[label] / label_total_counts[label]) * 100
    print(f"{label}: {label_accuracy:.2f}%")


# In[19]:


from sklearn.metrics import classification_report

# Get the predicted labels for the test features
predicted_labels = np.argmax(loaded_model1.predict(features_test), axis=1)

# Get the true labels from one-hot encoded format
true_labels = np.argmax(labels_test, axis=1)

# Generate a classification report
report = classification_report(true_labels, predicted_labels, target_names=CLASS_LABELS)

# Print the classification report
print(report)


# In[23]:


def process_video_file(video_path, model):
    frames_list = frames_extraction(video_path)

    if len(frames_list) == SEQUENCE_LENGTH:
        input_sequence = np.expand_dims(frames_list, axis=0)  # Add batch dimension
        predictions = model.predict(input_sequence)
        predicted_label_index = np.argmax(predictions)
        predicted_label = CLASS_LABELS[predicted_label_index]

        print(f"Predicted activity for video {video_path}: {predicted_label}")

        # Display the video
        display_video(video_path)
    else:
        print(f"Skipping video {video_path} due to insufficient frames.")

def display_video(video_path):
    cap = cv2.VideoCapture(video_path)

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Display the frame
            cv2.imshow('Video', frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
# Load the saved model for testing
loaded_model2 = tf.keras.models.load_model("ResNet50_LSTM_model1.h5")


# In[24]:


# Initialize dictionaries to store counts of correct predictions and total samples for each label
label_correct_counts = {label: 0 for label in CLASS_LABELS}
label_total_counts = {label: 0 for label in CLASS_LABELS}

# Iterate through each sample in the test dataset
for i in range(len(features_test)):
    
    # Make predictions using the trained model for the current sample
    predicted_label = np.argmax(loaded_model2.predict(np.expand_dims(features_test[i], axis=0))[0])
    
    # Get the actual label from the test labels
    actual_label = np.argmax(labels_test[i])
    
    # Increment the total count for the actual label
    label_total_counts[CLASS_LABELS[actual_label]] += 1
    
    # Check if the predicted label matches the actual label
    if predicted_label == actual_label:
        # Increment the correct count for the actual label
        label_correct_counts[CLASS_LABELS[actual_label]] += 1

# Calculate and print the accuracy for each label
print("Accuracy for each label:")
for label in CLASS_LABELS:
    label_accuracy = (label_correct_counts[label] / label_total_counts[label]) * 100
    print(f"{label}: {label_accuracy:.2f}%")


# In[25]:


from sklearn.metrics import classification_report

# Get the predicted labels for the test features
predicted_labels = np.argmax(loaded_model2.predict(features_test), axis=1)

# Get the true labels from one-hot encoded format
true_labels = np.argmax(labels_test, axis=1)

# Generate a classification report
report = classification_report(true_labels, predicted_labels, target_names=CLASS_LABELS)

# Print the classification report
print(report)


# In[ ]:




