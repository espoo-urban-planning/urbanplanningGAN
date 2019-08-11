import cv2
import numpy as np
import glob

files = glob.glob("_out8/*.png")

for file in files: 
	img = cv2.imread(file)

	img1 = img[0:512, 0:512]
	img2 = img[0:512, 512:1024]

	b1,g1,r1 = cv2.split(img1)
	b2,g2,r2 = cv2.split(img2)

	cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	ones = np.ones((3,3))
	r1_hacked = cv2.dilate(r1,cross,iterations = 3)
	r1_hacked = cv2.erode(r1_hacked,cross,iterations = 3)

	r1_hacked = cv2.dilate(r1_hacked,ones,iterations = 2)
	r1_hacked = cv2.erode(r1_hacked,ones,iterations = 2)

	r1_hacked = cv2.dilate(r1_hacked,ones,iterations = 2)
	r1_hacked = cv2.erode(r1_hacked,ones,iterations = 2)

	r1_hacked = cv2.dilate(r1_hacked,ones,iterations = 2)
	r1_hacked = cv2.erode(r1_hacked,ones,iterations = 2)
	
	r1_dilated = cv2.dilate(r1,cross,iterations = 3)
	b2 = cv2.bitwise_and(b2, b2, mask=r1_dilated)

	out = np.zeros((512, 1024, 3))
	out[0:512, 0:512] = img1
	out[0:512, 512:1024, 0] = b2

	fix_name = file.replace("out8/", "")
	#cv2.imwrite(fix_name, out)
	
	#fix_name = file.replace(".png", "_2.png")
	out[0:512, 0:512, 2] = r1_hacked
	
	print(fix_name)
	cv2.imwrite("fix8/" + fix_name, out)

	