from keras.models import Model, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os


class SmallerVGGNet:

    @staticmethod
    def build_output1(inputs, width, height, depth):

        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        x = Conv2D(32, (3,3), padding="same", input_shape=inputShape)(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3,3))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3,3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)

        x = Conv2D(64, (3,3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3,3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)

        x = Conv2D(128, (3,3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        x = Dense(1)(x)
        x = Activation('relu')(x)
        x = BatchNormalization(name = 'output_1')(x)

        return x

    @staticmethod
    def build_output2(inputs, width, height, depth, classes):

        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        x = Conv2D(32, (3, 3), padding="same", input_shape=inputShape)(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        x = Dropout(0.5)(x)
        x = Dense(classes)(x)
        x = Activation("sigmoid", name = 'output_2')(x)

        return x

    @staticmethod
    def build(width, height, depth, classes):

        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        intlayer = Input(shape=inputShape)

        output_1 = SmallerVGGNet.build_output1(intlayer, width, height, depth)

        output_2 = SmallerVGGNet.build_output2(intlayer, width, height, depth, classes)

        final = Model(intlayer, [output_1, output_2])

        return final


lossweights = {'output_1': 1.0, 'output_2': 1.0}
losses = {'output_1': 'mse', 'output_2': 'binary_crossentropy'}

resized_image_path  = '/home/harsha/CammCann/CamCann_Dataset/Resized_Images'
image_file_path = '/home/harsha/CammCann/CamCann_Dataset/image_data.npy'
csv_path = '/home/harsha/CammCann/CamCann_Dataset/modified.csv'

##============Processing The CSV==============##
print("[INFO]Processing The CSV")


dataframe = pd.read_csv(csv_path, header= None)

genders = dataframe.iloc[:, 1]
lb = LabelBinarizer()
genders = lb.fit_transform(genders)
image_name = dataframe.iloc[:, 3]
age = dataframe.iloc[:, 2]
resized = []

# for file in os.listdir(resized_image_path):
#     resized.append(str(file))
# dataframe = dataframe[dataframe[2].isin(resized)]
# dataframe.to_csv('/home/harsha/CammCann/CamCann_Dataset/modified.csv', header = None)

##============Splitting The Data=============##

print("[INFO]Splitting The Data")

img_data = np.load(image_file_path)
img_train, img_valid, o2_train, o2_valid, o1_train, o1_valid = train_test_split(img_data, genders, age, test_size=0.2, random_state= 24 )

##==========Creating The Checkpoints=========##

filepath = "/home/harsha/CammCann/CamCann_Dataset/Weights/weights-improvement-{epoch:02d}-{val_output_1_acc:.2f}-{val_output_2_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_output_2_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

##============Setting Up the model============##

epochs = 50
batch_size = 64

model = SmallerVGGNet.build(128, 128, 3, 1)
model.compile(optimizer= 'adam', loss= losses, loss_weights = lossweights, metrics=["accuracy"])
model.summary()

#=============Training the Model==============##
print("[INFO]Training the Model")

model.fit(img_train, {'output_1': o1_train, 'output_2': o2_train}, batch_size = batch_size, epochs=epochs, callbacks= callbacks_list,  validation_data=(img_valid, {'output_1': o1_valid, 'output_2': o2_valid}))


