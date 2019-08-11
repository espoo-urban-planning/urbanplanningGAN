import glob
import cv2

imgs = glob.glob("scrape8_test0/*.png")


for img_name in imgs:
	img = cv2.imread(img_name)
	if(img.shape[2] != 3):
		print("OHMY")
