import numpy as np
import cv2
import os
from PIL import Image, ImageSequence
from scipy import ndimage
import screeninfo


import argparse

##Building Argument Parser
# parser = argparse.ArgumentParser()
# parser.add_argument('-g','--gif',required = True, help = "Path for the GIF File")
# args = vars(parser.parse_args())

##Reading the GIF File
gif = Image.open('giphy.gif')
frames_folder_path = 'gif_frames'

def extract_frames(gif, frames_folder_path):
	if os.path.exists(frames_folder_path):
		for frames in os.listdir(frames_folder_path):
			os.remove(frames_folder_path + '/' + frames)
	else:
		os.makedir(frames_folder_path)

	for i in range(0, gif.n_frames):
		gif.seek(i)
		gif.save(frames_folder_path + '/frame{}.png'.format(i))

##Setting the Alpha Value for Generating the Addded Image
alpha = 0.4

def GIF_Superimposition(frames_folder_path, alpha):

	#Screen Info
	screen = screeninfo.get_monitors()[0]
	width, height = screen.width, screen.height

	##Capture Live Video Feed
	cap = cv2.VideoCapture(0)
	cap.set(3, width)
	cap.set(4, height)
	window_name = 'Live Feed'
	cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	flag = True

	##Process Frame by Frame
	while (flag):
		logo1 = cv2.imread('WhatsApp Image 2019-09-26 at 16.09.17.jpeg')
		logo1 = cv2.resize(logo1, (logo1.shape[1]//2 ,logo1.shape[0]//2))
		logo2 = cv2.imread('WhatsApp Image 2019-09-26 at 16.09.17 (3).jpeg')
		logo2 = cv2.resize(logo2, (logo2.shape[1]//2, logo2.shape[0]//2))
		logo3 = cv2.imread('WhatsApp Image 2019-09-26 at 16.09.17 (6).jpeg')
		logo3 = cv2.resize(logo3, (logo3.shape[1]//2, logo3.shape[0]//2))
		logo4 = cv2.imread('logo5.png')
		logo4 = cv2.resize(logo4, (logo4.shape[1]*2, logo4.shape[0]*2))


		for frame in os.listdir(frames_folder_path):

			#Resize the Frame
			gifframe = cv2.imread(frames_folder_path + '/' + frame)
			gifframe = cv2.resize(gifframe, (320,240))

			hgif = gifframe.shape[0]
			wgif = gifframe.shape[1]

			# Read the Frame and Flip it
			ret, background = cap.read()
			background = cv2.flip(background, 1)

			#Select the Overlay Lay Region
			vtop = background.shape[0]//2 - gifframe.shape[0]//2
			vbot = vtop + gifframe.shape[0]
			hmin = background.shape[1]//2 - gifframe.shape[1]//2
			hmax = hmin + gifframe.shape[1]


			# Capture frame-by-frame and Flip it Horizontally
			ret, background = cap.read()
			background = cv2.flip(background, 1)


			##Overlaying GIF
			added_image = cv2.addWeighted(background[vtop:vbot,hmin:hmax,:],alpha,gifframe[0:hgif,0:wgif,:],1-alpha,0)
			background[vtop:vbot,hmin:hmax] = added_image

			##Overlaying Logo
			added_image1 = cv2.addWeighted(background[0:logo1.shape[0],0:logo1.shape[1],:],alpha,logo1[0:logo1.shape[0],0:logo1.shape[1],:],1-alpha,0)
			added_image2 = cv2.addWeighted(background[background.shape[0] - logo2.shape[0]:background.shape[0], 0:logo2.shape[1], :], alpha,logo2[0:logo2.shape[0], 0:logo2.shape[1], :], 1 - alpha, 0)
			added_image3 = cv2.addWeighted(background[0:logo3.shape[0], background.shape[1]-logo3.shape[1]:background.shape[1], :], alpha,logo3[0:logo3.shape[0], 0:logo3.shape[1], :], 1 - alpha, 0)
			added_image4 = cv2.addWeighted(background[background.shape[0] - logo4.shape[0]:background.shape[0], background.shape[1] - logo4.shape[1]:background.shape[1], :], alpha, logo4[0:logo4.shape[0], 0:logo4.shape[1],:], 1-alpha, 0)
			background[0:logo1.shape[0], 0:logo1.shape[1]] = added_image1
			background[background.shape[0] - logo2.shape[0]:background.shape[0],0:logo2.shape[1],:] = added_image2
			background[0:logo3.shape[0], background.shape[1]-logo3.shape[1]:background.shape[1], :] = added_image3
			background[background.shape[0] - logo4.shape[0]:background.shape[0], background.shape[1] - logo4.shape[1]:background.shape[1], :] = added_image4
			cv2.imshow('Live Feed', background)


			##Press q to Exit the Program
			k = cv2.waitKey(1)
			if k == ord('q'):
				flag = False
				break
			# press a to increase alpha by 0.1
			if k == ord('a'):
				alpha +=0.1
				if alpha >=1.0:
					alpha = 1.0

			# press d to decrease alpha by 0.1
			elif k== ord('d'):
				alpha -= 0.1
				if alpha <=0.0:
					alpha = 0.0




	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

	
extract_frames(gif,frames_folder_path)
print("GIF Frames Ready")
GIF_Superimposition(frames_folder_path, alpha)
