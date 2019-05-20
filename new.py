#!/usr/bin/env python
# coding: utf-8

# In[8]:


from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.preprocessing import image                  
from tqdm import tqdm
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True      
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    cell_files = np.array(data['filenames'])
    cell_targets = np_utils.to_categorical(np.array(data['target']), 4)
    return cell_files, cell_targets


# In[10]:



# load train, test, and validation datasets
train_files, train_targets = load_dataset('datasets/TRAIN')
test_files, test_targets = load_dataset('datasets/TEST')


# In[11]:


#split training sets into validation and training set 
train_files, validation_files, train_targets, validation_targets = train_test_split(train_files, train_targets, test_size = 0.33, random_state=42)


# In[12]:


# print statistics about the dataset
print('There are %s total white blood cell images.\n' % len(np.hstack([train_files, test_files])))
print('There are %d training white blood cell images.' % len(train_files))
print('There are %d validation white blood cell images.'% len(validation_files))
print('There are %d test white blood cell images.'% len(test_files))


# In[13]:


#preprocess
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = (path_to_tensor(img_path) 
    for img_path in tqdm(img_paths))
    return np.vstack(list_of_tensors)
print('image preprocessed with keras') 


# In[14]:


# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
valid_tensors = paths_to_tensor(validation_files).astype('float32')/255


# In[16]:


print('Train Images Shape: {} size: {:,}'.format(train_tensors.shape, train_tensors.size))
print('Validation Images Shape: {} size: {:,}'.format(valid_tensors.shape, valid_tensors.size))
print('Test Images Shape: {} size: {:,}'.format(test_tensors.shape, test_tensors.size))


# In[17]:


from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications import xception
from keras.applications import inception_v3
from keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint


# In[18]:


#Extract Xception features and weights 
xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
train_xception = xception_bottleneck.predict(train_tensors, batch_size = 32, verbose = 1)
valid_xception = xception_bottleneck.predict(valid_tensors, batch_size = 32, verbose = 1)
test_xception = xception_bottleneck.predict(test_tensors, batch_size = 32, verbose = 1)


# In[19]:


print('Train Images Shape: {} size: {:,}'.format(train_xception.shape, train_xception.size))
print('Validation Images Shape: {} size: {:,}'.format(valid_xception.shape, valid_xception.size))
print('Test Images Shape: {} size: {:,}'.format(test_xception.shape, test_xception.size))


# In[2]:


input_shape = train_xception.shape[1:]
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', activation='relu', input_shape = train_xception.shape[1:]))
model.add(GlobalAveragePooling2D(input_shape=train_xception.shape[1:]))
model.add(Dense(4, activation='softmax'))
model.summary()


# In[21]:


# Compile the model.
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[38]:


#Train the model
xception_checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.test.hdf5', verbose=1, save_best_only=True)
epochs =20
batch_size = 64
xception_history = model.fit(train_xception, train_targets, validation_data = (valid_xception, validation_targets),
          epochs=epochs, batch_size=batch_size, callbacks=[xception_checkpointer], verbose=1)
  


# In[1]:


# Load the model weights with the best validation loss.
model.load_weights('saved_models/weights.best.test.hdf5')
   


# In[40]:


#Test the model
xception_predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in test_xception]

#test accuracy
test_accuracy = 100*np.sum(np.array(xception_predictions)==np.argmax(test_targets, axis=1))/len(xception_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# In[41]:


# Visualize Xception Accuracy Score 
plt.plot(xception_history.history['acc'])
plt.plot(xception_history.history['val_acc'])
plt.title('Xception Transfer Learning Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[42]:


#Visualize Xception Model Loss
plt.plot(xception_history.history['loss'])
plt.plot(xception_history.history['val_loss'])
plt.title('Xception Transfer Learning Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:




