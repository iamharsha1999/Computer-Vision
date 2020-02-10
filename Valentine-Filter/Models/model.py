from keras.layers import Dense, Conv2D, Activation, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten
from keras.models import Sequential

def build_model(height = 96, width = 96, depth = 1):

     model = Sequential()

     ## Conv Block 1
     model.add(Conv2D(32, (5,5), input_shape = (height, width, depth)))
     model.add(Activation('swish'))
     model.add(MaxPooling2D(pool_size = (2,2)))


     ## Conv Block 2
     model.add(Conv2D(64, (3,3)))
     model.add(Activation('swish'))
     model.add(MaxPooling2D(pool_size = (2,2)))
     model.add(Dropout(0.2))

     # Conv Block 3
     model.add(Conv2D(128, (3,3)))
     model.add(Activation('swish'))
     model.add(MaxPooling2D(pool_size = (2,2)))
     model.add(Dropout(0.3))

     # conv Block 3
     model.add(Conv2D(32, (3,3)))
     model.add(Activation('swish'))
     model.add(MaxPooling2D(pool_size = (2,2)))
     model.add(Dropout(0.3))

     ## Squeeze the extracted features into single dimension
     model.add(Flatten())

     # Introduce FC Layers
     model.add(Dense(64))
     model.add(Activation('swish'))
     model.add(BatchNormalization())

     model.add(Dense(128))
     model.add(Activation('swish'))
     model.add(BatchNormalization())

     model.add(Dense(256))
     model.add(Activation('swish'))
     model.add(BatchNormalization())

     model.add(Dense(64))
     model.add(Activation('swish'))
     model.add(BatchNormalization())

     model.add(Dense(30))


     return model
