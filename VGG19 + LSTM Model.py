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
from tensorflow.keras.applications import VGG19


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


# # One-Hot Encoding Labels and Splitting Data into Train and Test Sets

# In[6]:


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


# # VGG19-LSTM Video Classification Model

# In[7]:


# VGG19-LSTM Video Classification Model
# Define a function for creating a VGG19 + LSTM model
def vgg19_lstm_model():
    # Load the pre-trained VGG19 model without the top (fully connected) layers
    pretrained_model = VGG19(include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    
    # Set the pre-trained model to be non-trainable
    pretrained_model.trainable = False
    
    # Define the input layer for sequences of images
    inp = Input((SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    
    # Apply the pre-trained VGG19 to each frame in the sequence
    x = TimeDistributed(pretrained_model)(inp)
    
    # Apply Global Average Pooling to reduce spatial dimensions for each frame in the sequence
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    
    # Apply LSTM layers to capture temporal dependencies in the sequence
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(16)(x)
    
    # Apply a dense layer with ReLU activation
    x = Dense(24, activation='relu')(x)
    
    # Apply the final dense layer with softmax activation for classification
    out = Dense(len(CLASS_LABELS), activation='softmax')(x)
    
    # Create the final model
    model = Model(inp, out)
    
    # Compile the model with categorical crossentropy loss and Adam optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Display a summary of the model architecture
    model.summary()
    
    # Return the created model
    return model

# Create the VGG19-LSTM model
model = vgg19_lstm_model()


# # Training and Validation 

# In[8]:


# Define an early stopping callback
early_stopping_callback = EarlyStopping(monitor = 'accuracy', patience = 10, mode = 'max', restore_best_weights = True)

# Train the model using the provided training data
hist = model.fit(x = features_train, y = labels_train, epochs = 30, batch_size = 4 , shuffle = True, validation_split = 0.25)


# In[9]:


from tensorflow.keras.utils import plot_model

# Assuming 'model' is your Keras model
plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True)


# # Plot Training, Validation Accuracy, Validation Loss

# In[10]:


# Function to plot training and validation accuracy, and loss over epochs
def plot(history):
    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'val'], loc = "lower right")
    plt.show()
    
    # Plot training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'val'], loc = "upper right")
    plt.show()
    
    # Extract the final accuracy and validation accuracy for reporting
    accuracy = round(history.history['accuracy'][-1],4)
    validation_accuracy = round(history.history['val_accuracy'][-1],4)
    return (accuracy, validation_accuracy)


# In[11]:


plot(hist)


# # Calculate and Print Accuracy on Test Dataset

# In[12]:


# Initialize variable to store accuracy
acc = 0

# Iterate through each sample in the test dataset
for i in range(len(features_test)):
    
    # Make predictions using the trained model for the current sample
    predicted_label = np.argmax(model.predict(np.expand_dims(features_test[i],axis =0))[0])
    
    # Get the actual label from the test labels
    actual_label = np.argmax(labels_test[i])
    
    # Check if the predicted label matches the actual label
    if predicted_label == actual_label:
        acc += 1
        
# Calculate the accuracy as a percentage
acc = (acc * 100)/len(labels_test)

# Print the accuracy
print("Accuracy =",acc)


# In[14]:


# Save your Model.
model.save("VGG19_model2.keras")


# In[ ]:




