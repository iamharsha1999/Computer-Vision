from Models import build_model
import cv2
import  numpy as np
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import keras.backend as K
## Define Swish Fucntion
class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return K.sigmoid(x) * x

get_custom_objects().update({'swish': Swish(swish)})

## Build the model and load the weights
weight_path = 'Weights/weights-improvement-85-0.70.hdf5'
model = build_model()
model.load_weights(weight_path)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define the upper and lower boundaries for a color to be considered "Blue"
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)



# Load the video - O for webcam input
camera = cv2.VideoCapture(0)

while True:

    (grabbed, img) = camera.read()
    img = cv2.flip(img, 1)


    # Convert to GRAY for convenience
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces using the haar cascade object
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)

    # For each detected face using tha Haar cascade
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        img_copy = np.copy(img)
        img_copy_1 = np.copy(img)
        roi_color = img_copy_1[y:y+h, x:x+w]

        width_original = roi_gray.shape[1]      # Width of region where face is detected
        height_original = roi_gray.shape[0]     # Height of region where face is detected
        img_gray = cv2.resize(roi_gray, (96, 96))       # Resize image to size 96x96
        img_gray = img_gray/255         # Normalize the image data

        img_model = np.reshape(img_gray, (1,96,96,1))   # Model takes input of shape = [batch_size, height, width, no. of channels]
        keypoints = model.predict(img_model)[0]         # Predict keypoints for the current input

        # Keypoints are saved as (x1, y1, x2, y2, ......)
        x_coords = keypoints[0::2]      # Read alternate elements starting from index 0
        y_coords = keypoints[1::2]      # Read alternate elements starting from index 1

        x_coords_denormalized = (x_coords+0.5)*width_original       # Denormalize x-coordinate
        y_coords_denormalized = (y_coords+0.5)*height_original      # Denormalize y-coordinate

        for i in range(len(x_coords)):          # Plot the keypoints at the x and y coordinates
            cv2.circle(roi_color, (x_coords_denormalized[i], y_coords_denormalized[i]), 2, (255,255,0), -1)

        # Particular keypoints for scaling and positioning of the filter
        left_lip_coords = (int(x_coords_denormalized[11]), int(y_coords_denormalized[11]))
        right_lip_coords = (int(x_coords_denormalized[12]), int(y_coords_denormalized[12]))
        top_lip_coords = (int(x_coords_denormalized[13]), int(y_coords_denormalized[13]))
        bottom_lip_coords = (int(x_coords_denormalized[14]), int(y_coords_denormalized[14]))
        left_eye_coords = (int(x_coords_denormalized[3]), int(y_coords_denormalized[3]))
        right_eye_coords = (int(x_coords_denormalized[5]), int(y_coords_denormalized[5]))
        brow_coords = (int(x_coords_denormalized[6]), int(y_coords_denormalized[6]))

        # Scale filter according to keypoint coordinates
        glasses_width = right_eye_coords[0] - left_eye_coords[0]

        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2BGRA) # Used for transparency overlay of filter using the alpha channel

        # Glasses filter
        glasses = cv2.imread('images-removebg-preview.png', -1)
        glasses = cv2.resize(glasses,  None, (glasses_width*2,150))
        gw,gh,gc = glasses.shape

        for i in range(0,gw):       # Overlay the filter based on the alpha channel
            for j in range(0,gh):
                if glasses[i,j][3] != 0:
                    img_copy[brow_coords[1]+i+y-50, left_eye_coords[0]+j+x-60] = glasses[i,j]

        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGRA2BGR)       # Revert back to BGR

        cv2.imshow('Output',img_copy)           # Output with the filter placed on the face
        cv2.imshow('Keypoints predicted',img_copy_1)        # Place keypoints on the webcam input

    cv2.imshow('Webcam',img)        # Original webcame Input

    if cv2.waitKey(1) & 0xFF == ord("e"):   # If 'e' is pressed, stop reading and break the loop
        break
