import cv2
import numpy as np
import sklearn
from random import shuffle
import scipy.misc
import matplotlib.pyplot as plt


def generator(samples, batch_size=32):
    num_samples = len(samples)
    print (num_samples)
    correlation=0.1
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '.'+batch_sample[0]
#                 print(name)
                center_image = cv2.imread(name)
                center_image=cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB)
#                 plt.imshow(center_image)
#                 plt.show()
              
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
#                 if center_angle!=0.0:
                    
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1)
                
                name = '.'+batch_sample[1]
#                 print(name)
             
                left_image = cv2.imread(name)
#                 plt.imshow(left_image)
#                 plt.show()
                left_image=cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB)
                left_angle = float(batch_sample[3])
                if left_angle !=0.0:
#                 if left_angle >=-.2:                        
#                     left_angle=left_angle-correlation

                    images.append(left_image)
                    angles.append(left_angle)
                images.append(cv2.flip(left_image,1))
                angles.append(left_angle*-1)
                name = '.'+batch_sample[2]
                right_image = cv2.imread(name)
                right_image=cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)                
                right_angle = float(batch_sample[3])
                if right_angle!=0.0:  
#                 if right_angle <=.2:
#                     right_angle=right_angle+correlation

                    images.append(right_image)
                    angles.append(right_angle)
                images.append(cv2.flip(right_image,1))
                angles.append(right_angle*-1)
                    

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
#             print(y_train)
            sklearn.utils.shuffle(X_train, y_train)
            
           
            yield (X_train, y_train)
            
def layerIntermed_output(inputs,outputs,numch):
        intermediate_layer_model= Model(inputs,outputs)
    
        intermediate_output = intermediate_layer_model.predict(lst0)
        print('in shape',intermediate_output.shape)
        sampleI=intermediate_output[0]
        print(sampleI.shape)
        return sampleI[:,:,0]

    
    
    
import itertools    
from keras.utils import np_utils
import csv
from sklearn.model_selection import train_test_split
from random import randint
import  tensorflow as tf


lines=[]

with open ('./testImages/testImages6/driving_log2.csv') as csvfile:
    next(csvfile)
    reader =csv.reader(csvfile)
    for line in reader:
        lines.append(line)
with open ('./testImages/testImages7/driving_log.csv') as csvfile:
    next(csvfile)
    reader =csv.reader(csvfile)
    for line in reader:
        lines.append(line)


with open ('./testImages/testmages12/driving_log.csv') as csvfile:
    next(csvfile)
    reader =csv.reader(csvfile)
    for line in reader:
        lines.append(line)
with open ('./testImages/testImages13/driving_log.csv') as csvfile:
    next(csvfile)
    reader =csv.reader(csvfile)
    for line in reader:
        lines.append(line)
with open ('./testImages/testImages14/driving_log.csv') as csvfile:
    next(csvfile)
    reader =csv.reader(csvfile)
    for line in reader:
        lines.append(line)
   #### if didnt work multiply filters*2     

print(len(lines))

images=[]
mesurements=[]
print(len(lines))

train_samples, validation_samples = train_test_split(lines, test_size=0.2) 
# print("tran samples")
# print(len(train_samples))

ltrain=len(train_samples)
lval=len(validation_samples)

train_generator=generator(train_samples)
validation_generator = generator(validation_samples)


lst = list(itertools.islice(train_generator,1))[0]
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


model.add(Lambda(lambda x: ((x/255)-0.5),input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,20),(0,0))))
im = layerIntermed_output(model.input,model.layers[1].output,1) 

print(im.shape)
plt.title("copped")
plt.imshow(im,cmap='gray')
plt.savefig("./out/cropped.png")
plt.show()

model.add(Convolution2D(24,(5,5),strides=3,border_mode='same',activation='elu'))


im = layerIntermed_output(model.input,model.layers[2].output,3) 
print(im.shape)
plt.title("conv1")
plt.imshow(im) 
plt.savefig("./out/conv1_1.png")
plt.show()

model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.25))


model.add(Convolution2D(36,(5,5),strides=2,border_mode='same', activation='elu'))


im = layerIntermed_output(model.input,model.layers[5].output,3) 
print(im.shape)
plt.title("conv2")
plt.imshow(im) 
plt.savefig("./out/conv2_1.png")
plt.show()
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(48,(3,3),strides=2,border_mode='same',activation='elu'))
im = layerIntermed_output(model.input,model.layers[6].output,3) 
print(im.shape)
plt.title("conv3")
plt.imshow(im) 
plt.savefig("./out/conv3_1.png")
plt.show()

model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64,(5,5),border_mode='same'
                        ,activation='elu'))


im = layerIntermed_output(model.input,model.layers[9].output,3) 
print(im.shape)
plt.title("conv4")
plt.imshow(im)
plt.savefig("./out/conv4_1.png")
plt.show()

# model.add(MaxPooling2D((2, 2), strides=(1, 1)))
model.add(Dropout(0.5))

# model.add(Convolution2D(64,(5,5),border_mode='same',
#                        activation='elu'))


# im = layerIntermed_output(model.input,model.layers[10].output,3) 
# print(im.shape)
# plt.title("conv6")
# plt.imshow(im) 
# plt.savefig("./out/conv5_1.png")
# plt.show()

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))



model.compile(loss='mse',optimizer='adam')
history_object= model.fit_generator(train_generator,
                                    steps_per_epoch=ltrain,
                                    nb_epoch=3,
                                    validation_data=validation_generator,
                                    nb_val_samples=lval)






model.summary()
model.save('modelf.h5')

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