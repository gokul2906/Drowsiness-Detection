#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[10]:


def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)


# In[4]:


BS= 32
TS=(24,24)


# In[12]:


train_batch= generator(r'C:\Users\vevin\Desktop\New folder\Dataset\train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator(r'C:\Users\vevin\Desktop\New folder\Dataset\train',shuffle=True, batch_size=BS,target_size=TS)


# In[15]:


test_data = generator(r'C:\Users\vevin\Desktop\New folder\Dataset\test',target_size=(24,24),batch_size=BS)


# In[16]:


SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS
print(SPE,VS)


# In[17]:


model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
#32 convolution filters used each of size 3x3
#again
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

#64 convolution filters used each of size 3x3
#choose the best features via pooling
    
#randomly turn neurons on and off to improve convergence
    Dropout(0.25),
#flatten since too many dimensions, we only want a classification output
    Flatten(),
#fully connected to get all relevant data
    Dense(128, activation='relu'),
#one more dropout for convergence' sake :) 
    Dropout(0.5),
#output a softmax to squash the matrix into output probabilities
    Dense(2, activation='softmax')
])


# In[18]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[19]:


model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS)


# In[24]:


model.save(r'C:\Users\vevin\Desktop\New folder\models\dddmodel.h5', overwrite=True)


# In[23]:


acc_tr, loss_tr = model.evaluate(test_data)
print(acc_tr)
print(loss_tr)


# In[ ]:





# In[ ]:




