
# coding: utf-8

# In[1]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
import csv
import cv2
import scipy


# In[2]:


BASE_PATH = '../'


# In[3]:


def get_filename_for_index(index):
    PREFIX = 'Original_Images/BloodImage_'
    num_zeros = 5 - len(index)
    path = '0' * num_zeros + index
    return PREFIX + path + '.jpg'


# In[4]:


reader = csv.reader(open(BASE_PATH + 'labels.csv'))
# skip the header
next(reader)

X = []
y = []

for row in reader:
    label = row[2]
    if len(label) > 0 and label.find(',') == -1 and label is not 'BASOPHIL':
        filename = get_filename_for_index(row[1])
        img_file = cv2.imread(BASE_PATH + filename)
        if img_file is not None:
            img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
            img_file = scipy.misc.imresize(arr=img_file, size=(240, 320, 3))
            img_arr = np.asarray(img_file)
            X.append(img_arr)
            y.append(label)
        else:
            print("No file found", BASE_PATH + filename)


X = np.asarray(X)
y = np.asarray(y)


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


# In[6]:


eosinophil_samples = X_train[np.where(y_train == 'EOSINOPHIL')]
lymphocyte_samples = X_train[np.where(y_train == 'LYMPHOCYTE')]
monocyte_samples = X_train[np.where(y_train == 'MONOCYTE')]
neutrophil_samples = X_train[np.where(y_train == 'NEUTROPHIL')]


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

len(eosinophil_samples)
plt.imshow((eosinophil_samples[5])/(127.5 - 1))
plt.imshow((eosinophil_samples[5]))


# In[8]:


datagen = ImageDataGenerator(
    rotation_range=20,
    fill_mode='constant',
    height_shift_range=0.1,
    width_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2
)


# In[9]:


eosinophil_generator = datagen.flow(
        eosinophil_samples,
        y_train[np.where(y_train == 'EOSINOPHIL')],
        save_to_dir='../images/TRAIN/EOSINOPHIL',
        batch_size=1)

lymphocyte_generator = datagen.flow(
        lymphocyte_samples,
        y_train[np.where(y_train == 'LYMPHOCYTE')],
        save_to_dir='../images/TRAIN/LYMPHOCYTE',
        batch_size=1)

monocyte_generator = datagen.flow(
        monocyte_samples,
        y_train[np.where(y_train == 'MONOCYTE')],
        save_to_dir='../images/TRAIN/MONOCYTE',
        batch_size=1)

neutrophil_generator = datagen.flow(
        neutrophil_samples,
        y_train[np.where(y_train == 'NEUTROPHIL')],
        save_to_dir='../images/TRAIN/NEUTROPHIL',
        batch_size=1)


# In[14]:


datagen_simple = ImageDataGenerator()


# In[10]:


for i in range(2500):
    next(eosinophil_generator)
    next(lymphocyte_generator)
    next(monocyte_generator)
    next(neutrophil_generator)


# In[11]:


eosinophil_test_samples = X_test[np.where(y_test == 'EOSINOPHIL')]
lymphocyte_test_samples = X_test[np.where(y_test == 'LYMPHOCYTE')]
monocyte_test_samples = X_test[np.where(y_test == 'MONOCYTE')]
neutrophil_test_samples = X_test[np.where(y_test == 'NEUTROPHIL')]


# In[15]:


eosinophil_test_generator = datagen_simple.flow(
        eosinophil_test_samples,
        y_test[np.where(y_test == 'EOSINOPHIL')],
        save_to_dir='../images/TEST_SIMPLE/EOSINOPHIL',
        batch_size=1)

lymphocyte_test_generator = datagen_simple.flow(
        lymphocyte_test_samples,
        y_test[np.where(y_test == 'LYMPHOCYTE')],
        save_to_dir='../images/TEST_SIMPLE/LYMPHOCYTE',
        batch_size=1)

monocyte_test_generator = datagen_simple.flow(
        monocyte_test_samples,
        y_test[np.where(y_test == 'MONOCYTE')],
        save_to_dir='../images/TEST_SIMPLE/MONOCYTE',
        batch_size=1)

neutrophil_test_generator = datagen_simple.flow(
        neutrophil_test_samples,
        y_test[np.where(y_test == 'NEUTROPHIL')],
        save_to_dir='../images/TEST_SIMPLE/NEUTROPHIL',
        batch_size=1)


# In[18]:


for i in range(len(eosinophil_test_samples)):
    next(eosinophil_test_generator)
for i in range(len(lymphocyte_test_samples)):
    next(lymphocyte_test_generator)
for i in range(len(monocyte_test_samples)):
    next(monocyte_test_generator)
for i in range(len(neutrophil_test_samples)):
    next(neutrophil_test_generator)


# In[55]:


datagen = ImageDataGenerator(
    rotation_range=20,
    fill_mode='constant',
    height_shift_range=0.1,
    width_shift_range=0.1)

eosinophil_test_generator = datagen.flow(
        eosinophil_test_samples[0:1],
        y_test[np.where(y_test == 'EOSINOPHIL')][0:1],
        save_to_dir='../image_augmentation_example',
        batch_size=1)

for i in range(5):
    next(eosinophil_test_generator)

