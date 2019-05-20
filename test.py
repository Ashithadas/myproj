#!/usr/bin/env python2

# import the libraries
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2
import operator


def do_processing(path_to_file):
    #load the image
    image = cv2.imread(path_to_file)
    orig = image.copy()
     
    #pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    #load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model("MODEL")
     
    #classify the input image
    (Unknown, Orange, Apple, Raspberry) = model.predict(image)[0]

    #build the label
    label_dict = {
        "Orange": Orange,
        "Apple": Apple,
        "Raspberry": Raspberry,
	    "Unknown": Unknown
    }

    label = max(label_dict.items(), key=operator.itemgetter(1))[0]
    max_value = label_dict[label]

    label = "{}: {:.2f}%".format(label, max_value * 100)
     
    #draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	    0.7, (0, 255, 0), 2)

    #write the output image
    cv2.imwrite("output.jpg", output)
