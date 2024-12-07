import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred, class_names=None):
    """
    Evaluates the model using accuracy, classification report, and confusion matrix. 
    Parameters:
    - y_true: True labels (numpy array or list)
    - y_pred: Predicted labels (numpy array or list)
    - class_names: List of class names (optional, for labeling confusion matrix)
    """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)  
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
# Assuming y_test is your true labels and predicted_labels are the model's predictions
evaluate_model(np.argmax(y_test, axis=1), predicted_labels, class_names=['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'])
