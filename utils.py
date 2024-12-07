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
