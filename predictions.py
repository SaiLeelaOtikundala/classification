import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers,models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image

x_train, x_test, y_train, y_test = train_test_split(images, label_data, test_size=0.2, random_state=42)

x_train = np.transpose(x_train, (0, 2, 3, 1))  # Convert to (batch_size, 32, 32, 3)
x_test = np.transpose(x_test, (0, 2, 3, 1))  # Convert to (batch_size, 32, 32, 3)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
])
# Summary of the model architecture
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=15, batch_size=64, validation_data=(x_test, y_test))

# Function to visualize images with true and predicted labels (displaying integer labels)
def visualize_predictions(images, true_labels, predicted_labels, num_images=10):
    # Ensure num_images does not exceed the number of available test images
    num_images = min(num_images, len(images))  # Use the minimum of the requested number and available images


    # Select random indices to display
    random_indices = np.random.choice(range(len(images)), num_images, replace=False)

    # Set up the grid for displaying images
    fig, axes = plt.subplots(num_images, 2, figsize=(15, 20))
    #axes = axes.flatten()  # Flatten the axes array for easier indexing

    for i, idx in enumerate(random_indices):
        image = images[idx]
        true_label = true_labels[idx]
        predicted_label = predicted_labels[idx]
        
        # Ensure the image is in the correct shape (32, 32, 3) and rescale from [0, 1] to [0, 255]
        if image.shape != (32, 32, 3):
            image = np.transpose(image, (1, 2, 0))

        # If the image is a tensor, convert to numpy
        if isinstance(image, np.ndarray) is False:
            image = image.numpy()  # Convert tensor to numpy if using a framework like PyTorch

        # If the image is normalized (i.e., values are between 0 and 1), scale it back to [0, 255]
        if image.max() <= 1.0:  # This indicates normalization
            image = (image * 255).astype(np.uint8)


        # Access the axes for the current image pair
        ax_true = axes[i][0]  # Left column: True images
        ax_pred = axes[i][1]  # Right column: Predicted images

        
        # Display the true image
        ax_true.imshow(image)
        ax_true.axis('off')
        ax_true.set_title(f"True: {true_label}")

        # Display the predicted image
        ax_pred.imshow(image)
        ax_pred.axis('off')
        ax_pred.set_title(f"Pred: {predicted_label}")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Make predictions using the trained model
predictions = model.predict(x_test)

# Convert predictions to labels (highest probability class)
predicted_labels = np.argmax(predictions, axis=1)


# Visualize the predictions for 10 random test images
visualize_predictions(x_test, np.argmax(y_test, axis=1), predicted_labels, num_images=10)
