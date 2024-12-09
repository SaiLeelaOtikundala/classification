import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers,models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

def visualize(images, labels, num_images=10):
    """
    Visualizes a random subset of images with their corresponding labels.
    
    Arguments:
    - images (numpy array or list): A 4D array containing images with shape (num_samples, height, width, channels).
    - labels (numpy array or list): A 1D array containing the labels corresponding to the images.
    - num_images (int): The number of images to display (default is 10).
    
    The function displays 'num_images' random images from the dataset along with their labels.
    """
    # Select random indices without going out of range
    random_indices = random.sample(range(len(images)), num_images)  # Select random indices
    
    # Create a 5x2 grid layout for the images (2 columns, 5 rows)
    fig, axes = plt.subplots(5, 2, figsize=(10, 15))  
    
    # Flatten axes to simplify indexing (since axes is 2D, we'll treat it as a 1D array)
    axes = axes.flatten()

    # Loop over the selected random indices to display images
    for i, index in enumerate(random_indices):
        ax = axes[i]  # Access the correct axis from the flattened axes
        image = images[index]  # Get the image at the selected index
        label = labels[index]  # Get the label at the selected index

       
        image = np.transpose(image, (1, 2, 0))

        # Display the image with aspect='auto' to maintain correct aspect ratio
        ax.imshow(image) 
        
        # Set the title with the label (each image gets its own label)
        ax.set_title(f"Label: {label}")
        
        # Hide axis labels for a cleaner look
        ax.axis('off')
    
    # Adjust layout to prevent overlap and ensure tight layout
    plt.tight_layout()  # Increase padding to make the layout more spacious
    plt.show()

# Assuming `df` is a pandas DataFrame containing pixel data and labels
pixel_columns = [f'pixel_{i}' for i in range(0, 3072)]  # Column names for CIFAR-10 pixels
pixel_data = df[pixel_columns].values  # Shape (num_images, 3072)

# Reshape pixel data from (num_images, 3072) to (num_images, 32, 32, 3)
images = pixel_data.reshape(-1,3,32,32)

# Extract the label data as a list
label_data = list(df['label'])

# Visualize 10 random CIFAR-10 images in a 2-column, 5-row grid
visualize(images, label_data, num_images=10)
