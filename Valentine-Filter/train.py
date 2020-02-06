from Models import build_model
from DataPrep import load_data
from time import time
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Activation
import  matplotlib.pyplot as plt
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

## Define Swish Fucntion
class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return K.sigmoid(x) * x

get_custom_objects().update({'swish': Swish(swish)})


model = build_model()

# Instantiate Tensorboard
tb = TensorBoard(log_dir = 'TB_logs/logs/{}'.format(time()))


# Compile the Model
model.compile(optimizer = 'adadelta', loss = 'mean_absolute_error', metrics = ['accuracy'])

# Load the data
x_train, y_train = load_data()

# Creating Checkpoints
filepath = 'Weights/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Train the model
history = model.fit(x_train, y_train, epochs = 100, batch_size = 256, verbose = 1, validation_split = 0.2, callbacks = callbacks_list)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
