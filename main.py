#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
from keras.models import load_model
import numpy as np
from keras.applications import xception
import imutils


# In[2]:


def select_image():
	# grab a reference to the image panels
	global panelA, panelB, panelC, path, disease, name
	path = filedialog.askopenfilename()
	if len(path) > 0:
		im = Image.open(path)
		im = ImageTk.PhotoImage(im)
		if panelA is None or panelB is None or panelC is None:
			panelA = Label(image=im)
			panelA.image = im
			panelA.grid()
			

		else:
			panelA.configure(image=im)
			panelA.image = im


# In[3]:


def process():
	global panelA, panelB, panelC, path, disease, name
	test_img_path=path
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
	panelB=Label(root,text=name,bg='gold',fg='blue')
	#label3=Label(root,text=name,bg='gold',fg='blue')
	panelB.grid(row=1, column=1)


# In[4]:


def count():
	global panelA, panelB, panelC, path, disease, name,count
# dict to count colonies
	counter = {}
# load the image
	test_img_path=path
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
	image_gray = cv2.cvtColor(image_res, cv2.COLOR_BGR2GRAY)
	image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
	image_edged = cv2.Canny(image_gray, 50, 100)
	image_edged = cv2.dilate(image_edged, None, iterations=1)
	image_edged = cv2.erode(image_edged, None, iterations=1)
	cnts,_ = cv2.findContours(image_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#cnts = cnts[0] if imutils.is_cv2() else cnts[1]
 
	for c in cnts:
		hull = cv2.convexHull(c)
		if color == 'blue':
            # prints contours in red color
			cv2.drawContours(image_contours,[hull],0,(0,0,255),1)
			counter[color] += 1
       
	count=counter[color]  



    # find contours in the edge map

	if count >= 15 :
		if name == 'Eosinophil' :
			disease="parasitic infection"
		elif name == 'Lymphocyte' :
			disease="viral infection"
		elif name == 'Monocyte' :
			disease="inflamatory disease"
		elif name== 'Neutrophil' :
			disease="bacterial infection"
		else:
			disease="diseased"
	elif count <= 1 :
		if name == 'Eosinophil' :
			disease="parasitic infection"
		elif name == 'Lymphocyte' :
			disease="viral infection"
		elif name == 'Monocyte' :
			disease="inflamatory disease"
		elif name== 'Neutrophil' :
			disease="bacterial infection"
	else :
		
		disease="not diseased"
	panelC=Label(root,text=disease,bg='gold',fg='blue',)
	#label3=Label(root,text=name,bg='gold',fg='blue')
	panelC.grid(row=1, column=2)


# In[5]:


def exits():
    root.destroy()


# In[9]:


root = Tk()
#back = Tk.Frame(master=root, width=500, height=500, bg='black')
#back.pack()
root.geometry("1200x500")

panelA = None
panelB = None
panelC = None
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Input", command=select_image,  height = 2, width = 20)
btn.grid(row = 0, column = 0, pady = 15, padx = 20) 

btn1 = Button(root, text="classify", command=process,  height = 2, width = 20)
btn1.grid(row = 0, column = 1, pady = 15, padx = 20) 

btn2 = Button(root, text="Disease", command=count, height = 2, width = 20)
#btn2.pack(side="left")
btn2.grid(row = 0, column = 2, pady = 15, padx = 20) 


btn3 = Button(root, text="exit", command=exits, height = 2, width = 20)
#btn3.pack(side="left")
btn3.grid(row = 0, column = 3, pady = 15, padx = 20) 
# kick off the GUI


root.mainloop()


# In[ ]:





# In[ ]:




