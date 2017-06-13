
# coding: utf-8

# In[5]:



import cv2
import numpy as np
import sklearn
from random import shuffle
import scipy.misc
import matplotlib.pyplot as plt


def generator(samples, batch_size=32):
    num_samples = len(samples)
    print (num_samples)
    correlation=0.12
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)   #shuffle samples
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]  #extract batch samples

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '.'+batch_sample[0]
                center_image = cv2.imread(name)
                center_image=cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB)
              
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                if center_angle !=0.0:      #ignore zero angles
                    
                    images.append(cv2.flip(center_image,1))    #flip image
                    angles.append(center_angle*-1)
                
                name = '.'+batch_sample[1]
             
                left_image = cv2.imread(name)
                left_image=cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB)
                left_angle = float(batch_sample[3])
                if left_angle !=0.0:      #ignore zero angles
		            left_angle=left_angle+correlation
                    images.append(left_image)
                    angles.append(left_angle)
                    images.append(cv2.flip(left_image,1))    #flip image
                    angles.append(left_angle*-1)
                name = '.'+batch_sample[2]
                right_image = cv2.imread(name)
                right_image=cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)                
                right_angle = float(batch_sample[3])
                if right_angle!=0.0:                    #ignore zero ang
                    
		            right_angle=right_angle+correlation
                
                    images.append(right_image)
                    angles.append(right_angle)
                    images.append(cv2.flip(right_image,1))    #flip image
                    angles.append(right_angle*-1)
                    

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
#             print(y_train)
            sklearn.utils.shuffle(X_train, y_train)
            
           
            yield (X_train, y_train)




# In[6]:

def layerIntermed_output(inputs,outputs,numch):   #this function visualise output layes
    
    
    intermediate_layer_model= Model(inputs,
                                outputs)
    
    intermediate_output = intermediate_layer_model.predict(lst0)
    print(intermediate_output.shape)
    sampleI=intermediate_output[0]
    print(sampleI.shape)
    return sampleI[:,:,0]
    


# In[7]:


import itertools    
from keras.utils import np_utils
import csv
from sklearn.model_selection import train_test_split
from random import randint


lines=[]
with open ('./testImages/driving_log.csv') as csvfile:   #load data
    next(csvfile)
    reader =csv.reader(csvfile)
    for line in reader:
        lines.append(line)
with open ('./data/driving_log.csv') as csvfile:   #load data
    next(csvfile)
    reader =csv.reader(csvfile)
    for line in reader:
        lines.append(line)

        



images=[]
mesurements=[]


train_samples, validation_samples = train_test_split(lines, test_size=0.2)   #split data to training and validation data
# print("tran samples")
# print(len(train_samples))

ltrain=len(train_samples)
lval=len(validation_samples)

train_generator=generator(train_samples)        #generate train
validation_generator = generator(validation_samples)      #generate validation


lst = list(itertools.islice(train_generator, 1))[0]   #extract sample of data
lst0=lst[0]


from keras.models import Sequential, Model

from keras import backend as k

from keras.layers import Flatten, Dense, Lambda, Cropping2D,Convolution2D,Dropout,Activation, Reshape
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


import matplotlib
from keras import layers

embedding_size = 50
maxlen=10
r= (100, 100,3)
model= Sequential()
## layer1


model.add(Cropping2D(cropping=((60,20),(0,0)),input_shape=(160,320,3)))
im = layerIntermed_output(model.input,model.layers[0].output,3) 
print(im.shape)
plt.title("copped")
plt.imshow(im)
plt.savefig("cropped.png")
plt.show()


## layer1
model.add(Convolution2D(24,(3,3),strides=(2, 2),input_shape=(40,160,3)
                        ,border_mode='same'
                        ,W_regularizer=l2(0.0002)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

im = layerIntermed_output(model.input,model.layers[3].output,3) 
print(im.shape)
plt.title("conv1")
plt.imshow(im)
plt.savefig("./out/conv1_1.png")
plt.show()


## layer2

model.add(Convolution2D(36,(3,3),border_mode='same',activation='elu',W_regularizer=l2(0.0002)))


im = layerIntermed_output(model.input,model.layers[3].output,3) 
print(im.shape)
plt.title("conv2")
plt.imshow(im) 
plt.savefig("./out/conv2_1.png")
plt.show()

model.add(MaxPooling2D((2, 2), strides=(1, 1)))
model.add(Dropout(0.5))

## layer3
model.add(Convolution2D(48,(2,2),strides=2,border_mode='same',activation='elu',W_regularizer=l2(0.0002)))


im = layerIntermed_output(model.input,model.layers[6].output,3) 
print(im.shape)
plt.title("conv3")
plt.imshow(im) 
plt.savefig("./out/conv3_1.png")
plt.show()
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.25))


## layer4
model.add(Convolution2D(48,(3,3),border_mode='same',activation='elu',W_regularizer=l2(0.0002)))
im = layerIntermed_output(model.input,model.layers[9].output,3) 
print(im.shape)
plt.title("conv4")
plt.imshow(im) 
plt.savefig("./out/conv4_1.png")
plt.show()

model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.25))


## layer5
model.add(Convolution2D(64,(3,3),border_mode='same',activation='elu',W_regularizer=l2(0.0002)))

im = layerIntermed_output(model.input,model.layers[12].output,3) 
print(im.shape)
plt.title("conv5")
plt.imshow(im)
plt.savefig("./out/conv5_1.png")
plt.show()

#model.add(MaxPooling2D((2, 2), strides=(1, 1)))
model.add(Dropout(0.5))

## layer6

model.add(Convolution2D(64,(3,3),strides=(2, 2),border_mode='same',activation='elu'))


im = layerIntermed_output(model.input,model.layers[15].output,3) 
print(im.shape)
plt.title("conv6")

plt.imshow(im) 
plt.savefig("./out/conv6_1.png")
plt.show()

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))



model.compile(loss='mse',optimizer='adam')
history_object= model.fit_generator(train_generator,
                                    steps_per_epoch=ltrain,
                                    nb_epoch=5,
                                    validation_data=validation_generator,
                                    nb_val_samples=lval)






model.summary()
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("./out/data.png")
plt.show()


# In[ ]:

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("./out/history.png")
plt.show()

