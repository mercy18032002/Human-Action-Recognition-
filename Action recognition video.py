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


# In[2]:


#pip install opencv-python


# In[8]:


# Setting the height and width of the images in the dataset
IMAGE_HEIGHT , IMAGE_WIDTH = 128,128

# Setting the length of sequences (e.g., for video frames or time series data)
SEQUENCE_LENGTH = 30

# Specifying the path where the dataset is stored
DATASET_PATH = "dataset"

# Defining the class labels that represent different actions in the dataset
CLASS_LABELS = ["running","walking","handwaving"]


# In[9]:


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


# In[10]:


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


# In[11]:


# Load the saved model for testing
loaded_model = tf.keras.models.load_model("Xception_model4.h5")

# Provide the path to the test video file
test_video_path = "walk video.avi"

# Test the model on the provided video file
process_video_file(test_video_path, loaded_model)


# In[21]:


# Provide the path to the test video file
test_video_path = "walk video.avi"

# Test the model on the provided video file
process_video_file(test_video_path, loaded_model)


# In[32]:


# Provide the path to the test video file
test_video_path = "run.avi"

# Test the model on the provided video file
process_video_file(test_video_path, loaded_model)


# In[8]:


# Provide the path to the test video file
test_video_path = "handwaving.avi"

# Test the model on the provided video file
process_video_file(test_video_path, loaded_model)


# In[31]:


# Provide the path to the test video file
test_video_path = "lyova_walk.avi"

# Test the model on the provided video file
process_video_file(test_video_path, loaded_model)


# In[11]:


# Provide the path to the test video file
test_video_path = "shahar_run.avi"

# Test the model on the provided video file
process_video_file(test_video_path, loaded_model)


# In[15]:


# Provide the path to the test video file
test_video_path = "video_29.avi"

# Test the model on the provided video file
process_video_file(test_video_path, loaded_model)


# In[17]:


# Provide the path to the test video file
test_video_path = "Green+screen+films+++Male+walking+to+work+Savefrom.live.mp4"

# Test the model on the provided video file
process_video_file(test_video_path, loaded_model)


# In[18]:


IMAGE_HEIGHT , IMAGE_WIDTH =360, 640


# In[25]:


# Provide the path to the test video file
test_video_path = "WhatsApp Video 2023-12-28 at 4.08.01 PM.mp4"

# Test the model on the provided video file
process_video_file(test_video_path, loaded_model)


# In[ ]:




