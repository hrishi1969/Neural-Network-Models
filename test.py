import os
import glob
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D,\
     MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image

#Inflating the image dataset into the google colab directory

!unzip "/content/drive/My Drive/archive.zip" -d "/content/"

#The class is being used to store configuration parameters for training models
#Context-Free Grammar (CFG):
class CFG:
    EPOCHS= 50
    BATCH_SIZE= 64
    SEED= 42
    TF_SEED= 768
    HEIGHT= 224
    WIDTH= 224
    CHANNELS= 3
    IMAGE_SIZE=(224,224,3)

#Graphical representation of dataset

# List of directories to count files in
directories = ['/content/Data/test/COVID19',\
               '/content/Data/test/NORMAL',\
                '/content/Data/test/PNEUMONIA',\
               '/content/Data/train/COVID19',\
               '/content/Data/train/NORMAL',\
               '/content/Data/train/PNEUMONIA']

# Dictionary to store counts for each directory
file_counts = {}

# Loop through each directory
for directory in directories:
    # Initialize count to 0
    count = 0
    # Walk through the directory and count files
    for dirpath, _, filenames in os.walk(directory):
        count += len(filenames)
    # Store the count in the dictionary
    file_counts[directory] = count

# Assign values to variables
test_covid = file_counts['/content/Data/test/COVID19']
test_normal = file_counts['/content/Data/test/NORMAL']
test_pneumonia = file_counts['/content/Data/test/PNEUMONIA']
train_covid = file_counts['/content/Data/train/COVID19']
train_normal = file_counts['/content/Data/train/NORMAL']
train_pneumonia = file_counts['/content/Data/train/PNEUMONIA']

total_covid = test_covid + train_covid
total_normal = test_normal + train_normal
total_pneumonia = test_pneumonia + train_pneumonia

#Graphical representation of test dataset
# Plotting for test dataset
plt.figure(figsize=(10, 5))
test_bar = plt.bar(x=['COVID19', 'NORMAL', 'PNEUMONIA'],\
                   height=[test_covid, test_normal, test_pneumonia])
plt.ylabel('Count')
plt.title('Count of Each Type in Test Dataset')

# Displaying values on top of each bar
for bar in test_bar:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height}',\
             ha='center', va='bottom')

#Graphical representation of train dataset
plt.show()

# Plotting for train dataset
plt.figure(figsize=(10, 5))
train_bar = plt.bar(x=['COVID19', 'NORMAL', 'PNEUMONIA'],\
                    height=[train_covid, train_normal, train_pneumonia])
plt.ylabel('Count')
plt.title('Count of Each Type in Train Dataset')

# Displaying values on top of each bar
for bar in train_bar:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height}',\
             ha='center', va='bottom')

plt.show()

#Graphical representation of total dataset

# Plot size
# Plotting
plt.figure(figsize=(10, 5))
bars = plt.bar(x=['COVID19', 'NORMAL', 'PNEUMONIA'],\
               height=[total_covid, total_normal, total_pneumonia])
plt.ylabel('Count')
plt.title('Total count in test and train directory')

# Displaying values on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height}',\
             ha='center', va='bottom')

plt.show()
