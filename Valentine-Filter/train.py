from Models import build_model
from DataPrep import load_data
from time import time
from keras.callbacks import TensorBoard
import  matplotlib.pyplot as plt

model = build_model()

# Instantiate Tensorboard
tb = TensorBoard(log_dir = 'TB_logs/logs/{}'.format(time()))


# Compile the Model
model.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['accuracy'])

# Load the data
x_train, y_train = load_data()

# Train the model
history = model.fit(x_train, y_train, epochs = 100, batch_size = 256, verbose = 1, validation_split = 0.2, callbacks = [tb])

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
