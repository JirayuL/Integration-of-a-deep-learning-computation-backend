import os

# Helper libraries
import numpy as np
import cv2
import pandas as pd

def rm_main():
    X = []  # Image data

    imageLabel = ['Christmas_bear', 'Lab-keys',
                  'Apricot', 'Round_candle', 'Nut', 'Pot']
    # path = "./ImageClassification/grayscaleImage"
    path = "../../../../../../../Users/Gear/Downloads/venv/RapidMiner-Keras-Extension-Installation/ImageClassification/grayscaleImage"

    for image_path in os.listdir(path):
        # print(image_path)
        indexOfLabel = int(image_path[0])-1

        # label = imageLabel[indexOfLabel]
        label = indexOfLabel

        # Reads image and returns np.array
        img = cv2.imread(path+'/'+image_path)
        # Converts into the corret colorspace (GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize the image
        img = cv2.resize(img, (96, 72))
        flatArray = [label]
        for grayValue in img.flatten():
            flatArray.append(grayValue)
        X.append(flatArray)

    # Turn X into np.array
    X = np.array(X)

    return pd.DataFrame(X)


# rm_main()
