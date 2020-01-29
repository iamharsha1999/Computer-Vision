from keras.layers import Dense, Conv2D, Activation, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential

def build_model(height = 96, width = 96, depth = 1):

     model = Sequential()

     ## Conv Block 1
     model.add(Conv2D(64, (5,5), input_shape = (height, width, depth)))
     model.add(Activation('relu'))
     # model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size = (2,2)))

     ## Conv Block 2
     model.add(Conv2D(128, (3,3)))
     model.add(Activation('relu'))
     # model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size = (2,2)))

     # conv Block 3
     model.add(Conv2D(32, (3,3)))
     model.add(Activation('relu'))
     # model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size = (2,2)))

     ## Squeeze teh extracted features into single dimension
     model.add(GlobalAveragePooling2D())

     # Introduce FC Layers
     model.add(Dense(64))
     model.add(Activation('relu'))
     model.add(BatchNormalization())

     model.add(Dense(128))
     model.add(Activation('relu'))
     model.add(BatchNormalization())

     model.add(Dense(256))
     model.add(Activation('relu'))
     model.add(BatchNormalization())

     model.add(Dense(64))
     model.add(Activation('relu'))
     model.add(BatchNormalization())

     model.add(Dense(30))


     return model
