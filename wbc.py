#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
from keras.models import load_model
import numpy as np
from keras.applications import xception
import imutils
import cv2


# In[2]:


#test_img_path = "e11.jpeg"
def process(test_img_path)
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
	classes = ['Eosinophil','Lymphocyte','Monocyte','Neutrophil']
	name=classes[xception_predictions[0]]
	#img_data

    
        


# In[3]:

def count
# dict to count colonies
	counter = {}
# load the image
	image_orig = cv2.imread(test_img_path)
#height_orig, width_orig = image_orig.shape[:2]
# output image with contours
	image_contours = image_orig.copy()
	color='blue'
# copy of original image
	image_to_process = image_orig.copy()
# initializes counter
	counter[color] = 0
# define NumPy arrays of color boundaries (GBR vectors)
	lower = np.array([ 60, 100,  20])
	upper = np.array([170, 180, 150])
# find the colors within the specified boundaries
	image_mask = cv2.inRange(image_to_process, lower, upper)
# apply the mask
	image_res = cv2.bitwise_and(image_to_process, image_to_process, mask = image_mask)
## load the image, convert it to grayscale, and blur it slightly
	image_gray = cv2.cvtColor(image_res, cv2.COLOR_BGR2GRAY)
	image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
	image_edged = cv2.Canny(image_gray, 50, 100)
	image_edged = cv2.dilate(image_edged, None, iterations=1)
	image_edged = cv2.erode(image_edged, None, iterations=1)
 
    # find contours in the edge map
	cnts,_ = cv2.findContours(image_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
 
    # loop over the contours individually
	for c in cnts:
         
        # if the contour is not sufficiently large, ignore it
        #if cv2.contourArea(c) < 5:
            #continue
         
        # compute the Convex Hull of the contour
    		hull = cv2.convexHull(c)
    		if color == 'blue':
            # prints contours in red color
        		cv2.drawContours(image_contours,[hull],0,(0,0,255),1)
			counter[color] += 1
        #cv2.putText(image_contours, "{:.0f}".format(cv2.contourArea(c)), (int(hull[0][0][0]), int(hull[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
 
    # Print the number of colonies of each color
	count=counter[color]
	return(count) 
	if count > 15 :
    		if name == 'Eosinophil' :
        	print(name)
        	print("parasitic infection")
    	elif name == 'Lymphocyte' :
        	print("viral infection")
    	elif name == 'Monocyte' :
        	print(name)
        	print("inflamatory disease")
    	elif name== 'Neutrophil' :
        	print("bacterial infection")
	else :
    		print(name)
    		print("not diseased")


# In[ ]:




