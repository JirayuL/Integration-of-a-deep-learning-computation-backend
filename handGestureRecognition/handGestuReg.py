import os

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import cv2
import pandas as pd

# Sklearn
# Helps with organizing data for training
from sklearn.model_selection import train_test_split
# Helps present results as a confusion-matrix
from sklearn.metrics import confusion_matrix

# Import of keras model and hidden layers for our convolutional network
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten

print(tf.__version__)


def rm_main():
    X = []  # Image data
    y = []  # Labels

    imageLabel = ['A', 'B', 'C', 'Five', 'Point', 'V']
    # path = "./Marcel-Train"
    path = "../../../../../../../Users/Gear/Desktop/venv/RapidMiner-Keras-Extension-Installation/handGestureRecognition/Marcel-Train"

    # Loops through imagepaths to load images and labels into arrays
    for label in imageLabel:
        newPath = str(path)+'/'+str(label)
        for image_path in os.listdir(newPath):
            # Reads image and returns np.array
            img = cv2.imread(newPath+'/'+image_path)
            # Converts into the corret colorspace (GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Reduce image size so training can be faster
            img = cv2.resize(img, (320, 120))
            X.append(img)

            # Processing label in image path
            y.append(imageLabel.index(label))

    # Turn X and y into np.array to speed up train_test_split
    X = np.array(X)
    # Needed to reshape so CNN knows it's different images
    X = X.reshape(len(X), 120, 320, 1)
    y = np.array(y)
    print("Images loaded: ", len(X))
    print("Labels loaded: ", len(y))

    # Percentage of images that we want to use for testing.
    # The rest is used for training.
    ts = 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ts, random_state=42)

    # Construction of model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 320, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # Configures the model for training
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Trains the model for a given number of epochs (iterations on a dataset) and validates it.
    model.fit(X_train, y_train, epochs=5, batch_size=64,
              verbose=2, validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy: {:2.2f}%'.format(test_acc*100))

    # Make predictions towards the test set
    predictions = model.predict(X_test)

    # Transform predictions into 1-D array with label number
    y_pred = np.argmax(predictions, axis=1)
    return pd.DataFrame(confusion_matrix(y_test, y_pred),
                        columns=["Predicted A", "Predicted B", "Predicted C", "Predicted Five", "Predicted Point",
                                 "Predicted V"],
                        index=["Actual A", "Actual Palm B", "Actual C", "Actual Five", "Actual Point", "Actual V"])


# rm_main()
