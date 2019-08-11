import sys
import cv2
import numpy as np


def convert_to_vector(filename='download.png',outname='vector.png'):
	print("trying to open" + filename)
	img = cv2.imread(filename)
	print("trying to read blue channel")
	blue = img[:,:,0]
	print("thresholding")
	ret,img = cv2.threshold(blue,32,255,cv2.THRESH_BINARY)
	#kernel = np.ones((2,2),np.uint8)

	kernel1 = np.array([[0, 0, 0],
	                    [0, 1, 0],
	                    [0, 0, 0]], np.uint8)
	kernel2 = np.array([[1, 1, 1],
	                    [1, 0, 1],
	                    [1, 1, 1]], np.uint8)

	input_image = img
	input_image_comp = cv2.bitwise_not(img)  # could just use 255-img

	hitormiss1 = cv2.morphologyEx(input_image, cv2.MORPH_ERODE, kernel1)
	hitormiss2 = cv2.morphologyEx(input_image_comp, cv2.MORPH_ERODE, kernel2)
	hitormiss = cv2.bitwise_and(hitormiss1, hitormiss2)

	hitormiss_comp = cv2.bitwise_not(hitormiss) 
	del_isolated = cv2.bitwise_and(input_image, input_image, mask=hitormiss_comp)

	cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	dilated = cv2.dilate(del_isolated,cross,iterations = 2)
	eroded = cv2.erode(dilated,cross,iterations = 2)

	print("Cleaning up raster input to file " + outname)
	cv2.imwrite(outname, eroded)

if __name__ == "__main__":
    convert_to_vector('input.png', 'vector.png')