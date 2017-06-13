# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
[//]: # (Image References)
[image1]: ./out/cropped.png "Cropped Image"
[image2]: ./out/conv1.png "Conv1 Visualization"
[image3]: ./out/conv1_1.png "Conv1 Visualization single channel"
[image4]: ./out/conv2.png "Conv2 Visualization"
[image5]: ./out/conv2_1.png "Conv2 Visualization single channel"
[image6]: ./out/conv3.png "Conv3 Visualization"
[image7]: ./out/conv3_1.png "Conv3 Visualization single channel"
[image8]: ./out/conv4.png "Conv4 Visualization"
[image9]: ./out/conv4_1.png "Conv4 Visualization single channel"
[image10]: ./out/conv5.png "Conv5 Visualization"
[image11]: ./out/conv5_1.png "Conv5 Visualization single channel"
[image12]: ./out/conv6.png "Conv6 Visualization"
[image13]: ./out/conv6_1.png "Conv6 Visualization single channel"
[image14]: ./out/output_conv_6.png "Conv6 Visualization--prev tuning"
[image15]: ./out/history.png "Recovery Image"

---
#### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* track1 video result track1.mp4
* track2 video result track2.mp4
* writeup_report.md or writeup_report.pdf summarizing the results

#### Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
#### Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

## Model architecture 

I tried different models and architectures strats from simple architechture to Nvidia and GoogleLenet models and this work decided to implement Nvidia architecture and the discussion part will illustrate why.

The following code is the model architectures from `model.py` file. it consist of 6 convolution layers with filter (3x3) and depth from 24 to 64 and stride 2. and to reduce overfitting I applied different dropout range.

```python
model.add(Convolution2D(24,(3,3),strides=(2, 2),input_shape=(40,160,3),border_mode='same'W_regularizer=l2(0.0002)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(36,(3,3),border_mode='same',activation='elu',W_regularizer=l2(0.0002)))
model.add(MaxPooling2D((2, 2), strides=(1, 1)))
model.add(Dropout(0.5))


model.add(Convolution2D(48,(2,2),strides=2,border_mode='same',activation='elu',W_regularizer=l2(0.0002)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(48,(3,3),border_mode='same',activation='elu',W_regularizer=l2(0.0002)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64,(3,3),border_mode='same',activation='elu',W_regularizer=l2(0.0002)))
model.add(Dropout(0.5))

model.add(Convolution2D(64,(3,3),strides=(2, 2),border_mode='same',activation='elu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

```
The model was tuned dozens of time and `Elu activation` has solved the problem of of negative values and which is beneficial for learning and it helps to learn representations that are more robust to noise.
The other parameters where tuned and produced the following visualized layers after normalisation by`model.add(Lambda(lambda x: ((x/255)-0.5),input_shape=(160,320,3)))` and cropping by `Cropping2D(cropping=((60,20),(0,0)),input_shape=(160,320,3)))`:
![alt text][image1] 
![alt text][image3]
![alt text][image5]
![alt text][image6]
![alt text][image8]
![alt text][image10]

The model was tuned to obtain the following output layer as this segmentation for the image is the closest to the real world. 
![alt text][image13]
#### Model parameter tuning

The model used adam optimizer, so the learning rate was not tuned manually (model.py line 250). Also, as it its regression task the mean square error was chosen for calculating loss.

#### Appropriate training data

The provided `data.zip` and combination of my own data. My data where collected for both two tracks and data contais 3 labs for each track in defualt and reverse  direction and  where augmented and spitted to `X_train,y_train,X_test,y_test`. For avoiding high bias of particular steering angle , the appended images was the all the central angles, non 0.0: flipped center, left camera, filliped left camera, right camera, flipped right camera. this data is not the best as it has low number of 0.0 angles but it produced fully autonomous driving on the road and the loading data in function `def generator(samples, batch_size=32)`. Generator was used to generate batches of data in real-time for optimizing the training process time. The total number of data = 38394 and train data= 10238




---

## Disscussion 
After many trails the good result generated thanks to many tuning to get the output that represent the road segment. in the following output is a 3 segment for road which left off road , right off road and central road.
![alt text][image14]

However the following output which 5 segment of the image which I think it considers 5 road segment as shown above the the `conv1` output layer as each color represent 1 segment. 
![alt text][image13]
The final step was to run the simulator to see how well the car was driving around track one. due to low range of zero angles , the vehicle steers left and right with higher rate. however I think due to final output the vehicle drives it self in both tracks in endless of labs and file `Model_summary` contains the model summary and number of tuned parameters.

The archticture with out data normalisation made the car drive it self on both tracks`not_nor_track1.mp4, not_nor_track2.mp4`, however after normalisation the car drive  itself just on first track`track1.mp4` 
