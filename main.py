import numpy as np
import matplotlib.pyplot as plt
from activations import *
from layer import *
from loss import *
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load MNIST Digits Dataset
digits = datasets.load_digits()
images = digits.images
labels = digits.target


# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, shuffle=False)
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
# Image - (8,8) -> (1,64)

# Convert the test labels into arrays of probabilities
new_labels = []
for label in y_train:
    probs = [0] * 10
    probs[label] = 1
    new_labels.append(probs)

# Label - 2
# [0,0,1,0,0,0,0,0,0,0]

y_train = np.array(new_labels)
y_train_reshaped = y_train.reshape(y_train.shape[0], -1)


def display_images(images, Labels, title=None, predictions=None):
    """Helper functions to display images and model predictions

    Args:
        images (np.array): imges to classify
        labels (list): expected labels of images
        title (str, optional): graph title. Defaults to None.
        predictions (list, optional): model predicted labels of images. Default to None.
    """
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(8,8))
    fig.subplots_adjust(hspace=0.8)
    if title is not None: fig.suptitle(title, fontsize=20, fontweight="bold")

    for i in range(10):
        for j in range(10):
            axs[i][j].axis("off")
            axs[i][j].imshow(images[10*i+j].reshape((8,8)), cmap="Greys")
            # Display as model prediction/actual label
            if predictions is not None: axs[i][j].set_title(f"{predictions[10*i+j]}/{Labels[10*i+j]}")
            else: axs[i][j].set_title(f"A:{Labels[10*i+j]}")

    plt.show()

def convert(classifier_predictions):
    """Helper function to convert numpy arrays of probabilities
        returned by the model into digit labels

    Args:
        classifier_predictions (np.array): model predictions

    Returns:
        list: dataset labels    
    """

    predictions = []

    for prediction in classifier_predictions:
        curr_pred = -1
        curr_prob = 0

        for i,val in enumerate(prediction[0]):
            if val > curr_prob:
                curr_pred = i
                curr_prob = val

        predictions.append(curr_pred)

    return predictions


if __name__=="__main__":
    # Scale input data to avoid explaoding gradients
    X_train_reshaped /= 16
    X_test_reshaped /= 16

    # Hyperparameters
    alpha = 1e-3
    batch_size = 32

    # MNIST digits classifier
    # use LayerList to initialize neural network model
    # Add Layer objects to model either in LayerList instantiation or using LayerList.append() method
    classifier = LayerList(Layer(64,32,alpha,batch_size,activation_func=ReLU()),
                           Layer(32,10,alpha,batch_size,activation_func=Softmax()))
    
    # Train using LayerList.fit() method
    classifier.fit(X_train_reshaped, y_train_reshaped, 1000, alpha, batch_size, categorical_cross_entropy_loss)

    # Evaluate using LayerList.predict() method
    display_images(X_test, y_test, title="NumpyNN Predictions", predictions=convert(classifier.predict(X_test_reshaped)))

    y_pred = convert(classifier.predict(X_test_reshaped))
    print("Test Accuracy:", accuracy_score(y_test, y_pred))


    # Assuming y_test and y_pred are your true and predicted labels

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # 2. Classification Report
    print("Classification Report:\n", classification_report(y_test, y_pred))
