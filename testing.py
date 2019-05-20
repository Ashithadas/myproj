#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
from keras.models import load_model
import numpy as np
from keras.applications import xception
import imutils
import cv2
from tkinter import *
from tkinter.filedialog import askopenfilename
import csv
import os
from tkinter import ttk
import time


# In[2]:




# In[3]:




def process(test_img_path):
	#test_img_path = "l3.jpeg"
	img_data=Image.open(test_img_path)
	img=img_data.resize((224,224), Image.ANTIALIAS)
	x = np.array(img)
	y=np.expand_dims(x, axis=0)
	image = y.astype('float32')/255
	#Extract Xception features and weights 
	xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
	preds = xception_bottleneck.predict(image, batch_size=32, verbose=1)
	model = load_model('saved_models/weights.best.test.hdf5')
	xception_predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in preds]
	xception_predictions[0]
	classes = ['Eosinophil','Lymphocyte','Monocyte','Neutrophil','Basophil']
	name=classes[xception_predictions[0]]
	return(name)


# In[15]:


def count(test_img_path):
	counter = {}
	image_orig = cv2.imread(test_img_path)
	image_contours = image_orig.copy()
	color='blue'
	image_to_process = image_orig.copy()
	counter[color] = 0
	lower = np.array([ 60, 100,  20])
	upper = np.array([170, 180, 150])
	image_mask = cv2.inRange(image_to_process, lower, upper)
	image_res = cv2.bitwise_and(image_to_process, image_to_process, mask = image_mask)
	image_gray = cv2.cvtColor(image_res, cv2.COLOR_BGR2GRAY)
	image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
	image_edged = cv2.Canny(image_gray, 50, 100)
	image_edged = cv2.dilate(image_edged, None, iterations=1)
	image_edged = cv2.erode(image_edged, None, iterations=1)
	cnts,_ = cv2.findContours(image_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	for c in cnts:

		hull = cv2.convexHull(c)
		if color == 'blue':
			cv2.drawContours(image_contours,[hull],0,(0,0,255),1)
			counter[color] += 1

	count=counter[color]
	return(count)








