import cv2
import numpy as np
from Models import build_model
from keras.layers import Activation
import keras.backend as K
from keras.utils import get_custom_objects
import timeit


class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return K.sigmoid(x) * x
get_custom_objects().update({'swish': Swish(swish)})

model = build_model()
model.load_weights('Weights/weights-improvement-fswish-100-0.00.hdf5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


kernel=np.ones((5,5),np.uint8)
filters='Image_Filters/gettyimages-186840067-612x612-removebg-preview (1).png'

camera=cv2.VideoCapture(0)

while True:
    start = timeit.timeit()
    (grabbed,frame)=camera.read()
    frame=cv2.flip(frame,1)
    frame2=np.copy(frame)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    faces=face_cascade.detectMultiScale(gray,1.25,6)

    for (x,y,w,h) in faces:
    	gray_face=gray[y:y+h,x:x+w]
    	color_face=frame[y:y+h,x:x+w]

    	gray_normalized=gray_face/255

    	original_shape=gray_face.shape
    	face_resized=cv2.resize(gray_normalized,(96,96),interpolation=cv2.INTER_AREA)
    	face_resized_copy=face_resized.copy()
    	face_resized=face_resized.reshape(1,96,96,1)

    	keypoints=model.predict(face_resized)

    	keypoints=keypoints*48+48

    	face_resized_color=cv2.resize(color_face,(96,96),interpolation=cv2.INTER_AREA)
    	face_resized_color2=np.copy(face_resized_color)

    	points=[]
    	for i,co in enumerate(keypoints[0][0::2]):
    		points.append((co,keypoints[0][1::2][i]))

    	#Add filter
    	sunglasses=cv2.imread(filters,cv2.IMREAD_UNCHANGED)
    	sunglass_width = int((points[7][0]-points[9][0])*1.2)
    	sunglass_height = int((points[10][1]-points[8][1])/1.1)
    	sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC)
    	transparent_region = sunglass_resized[:,:,:3] != 0
    	face_resized_color[int(points[9][1]):int(points[9][1])+sunglass_height, int(points[9][0]):int(points[9][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]

    	frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)

    	for keypoint in points:
    		cv2.circle(face_resized_color2, keypoint, 1, (0,255,0), 1)

    	frame2[y:y+h, x:x+w] = cv2.resize(face_resized_color2, original_shape, interpolation = cv2.INTER_CUBIC)

    	cv2.imshow("Selfie Filters", frame)
    	cv2.imshow("Facial Keypoints", frame2)

    end = timeit.timeit()
    print(end - start)
    if cv2.waitKey(1) & 0xFF==ord("q"):
    	break

camera.release()
cv2.destroyAllWindows()
