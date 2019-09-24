from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import Input
from keras.regularizers import l2
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
import json
import os
import cv2 
import numpy as np

##Create the model
def build(height, width, depth, classes):
    # set the input shape to match the channel ordering
    if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            channel_dim = 1
    else:
        input_shape = (height, width, depth)
        channel_dim = -1

    # first (and only) CONV => RELU => POOL block
    inpt = Input(shape = input_shape)
    x = Conv2D(32, (3, 3), padding = "same", kernel_regularizer = 'l2')(inpt)
    x = Activation("relu")(x)
    x = BatchNormalization(axis = channel_dim)(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Dropout(0.25)(x)

    # first CONV => RELU => CONV => RELU => POOL block
    x = Conv2D(64, (3, 3), padding = "same", kernel_regularizer = 'l2')(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis = channel_dim)(x)
    x = Conv2D(64, (3, 3), padding = "same", kernel_regularizer = 'l2')(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis = channel_dim)(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Dropout(0.25)(x)

    # second CONV => RELU => CONV => RELU => POOL block
    x = Conv2D(128, (3, 3), padding = "same", kernel_regularizer = 'l2')(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis = channel_dim)(x)
    x = Conv2D(128, (3, 3), padding = "same", kernel_regularizer = 'l2')(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis = channel_dim)(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Dropout(0.25)(x)

    # flatten layer
    fl_out = Flatten()(x)
    
    # binary classifier
    bin_classifier = Dense(64, kernel_regularizer = 'l2')(fl_out)
    bin_classifier = Activation("relu")(bin_classifier)
    bin_classifier = BatchNormalization(axis = channel_dim)(bin_classifier)
    bin_classifier = Dense(1, kernel_regularizer = 'l2')(bin_classifier)
    bin_classifier = Activation("sigmoid", name = "bin_classifier")(bin_classifier)

    # regression head
    reg_head = Dense(64, kernel_regularizer = 'l2')(fl_out)
    reg_head = Activation("relu")(reg_head)
    reg_head = BatchNormalization(axis = channel_dim)(reg_head)
    reg_head = Dense(1, name = "reg_head", kernel_regularizer = 'l2')(reg_head)

    # return the constructed network architecture
    model = Model(inputs = inpt, outputs = [bin_classifier, reg_head])
    return model



model_weights = '/home/harsha/CammCann/model_098 - expt6.h5'
model = build(128, 128, 3, 2)
model.load_weights(model_weights)
	



for folders in os.listdir(filepath):
	
	data = {} #JSON Dictionary
	ni = 0 #Count of Images in the folder
    time = []  #List of time stamp of Images in the folder
    age = [] #List of age predictions of the person
    male = 0 #Count of Male Predictions
    female = 0 #Count of Female Predictions

	for image in os.listdir(filepath + '/' + folder):

		image_path = filepath + '/' + folder + '/' + image

		img = cv2.imread(image_path)
		img = cv2.resize(img, (128, 128))
		img = img.astype('float')/255
		ni+=1
		result =  model.predict(img)
		if result[0][0] > 0.5:
			male+=1
		else:
			female+=1


		age.append(result[0][1])

        time.append()

    if male>female:
        gen = 'Male'
    else:
        gen = 'Female'

    avgage = age/len(age) #Average age of the Person

    mintime = min(time)
    maxtime = max(time)

	#Writing the JSON file
	
	
    data['No.Of.Images'] =  str(ni)
    data['Gender'] =  str(gen)
    data['Age'] =  str(avgage)
    data['Duration'] = str(maxtime - mintime)

    jsonfile = open('Data.txt', 'w')
    json.dump(data, jsonfile) 

