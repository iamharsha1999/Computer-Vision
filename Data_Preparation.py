import cv2
import os
import numpy as np

img_file_path = '/home/harsha/CammCann/CamCann_Dataset/images2'
target_path = '/home/harsha/CammCann/CamCann_Dataset/Resized_Images'
npy_path = '/home/harsha/CammCann/CamCann_Dataset'

imgs = []

for file in os.listdir(target_path):
    img = cv2.imread(target_path + '/' + file, cv2.IMREAD_COLOR)
    imgs.append(img)
imgs = np.array(imgs)

np.save(npy_path + '/image_data.npy', imgs)




