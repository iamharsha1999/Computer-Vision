# import the necessary packages
from imutils.object_detection import non_max_suppression
from pyimagesearch.centroidtracker import CentroidTracker
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import numpy as np




# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

## Capture Video through Camera
cap = cv2.VideoCapture(0)

ct = CentroidTracker()
(H, W) = (None, None)

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()

    # Resize the frame
    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)


    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # Update the centroid tracker
    objects = ct.update(pick)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
    	# draw both the ID of the object and the centroid of the
    	# object on the output frame
    	text = "ID {}".format(objectID)
    	cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    	cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
    	cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.imshow("Live Feed", image)

    ## Break the loop to stop camera capture
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
