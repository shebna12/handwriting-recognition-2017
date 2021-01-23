
import glob
import cv2
import numpy as np 	
import math
# from matplotlib import pyplot as plt
# from sklearn import svm, datasets
import os
# import skimage
# from skimage import io
# from PIL import Image
from utils import get_median_area,i_and_j_preprocessor,resize_img,get_bounding_area,otsu_preprocess,remove_noise,remove_coinsides,get_ave_area




def check_negatives_general(folder_path):
	numfiles = 0
	num = 0
	for root, dirs, files in os.walk(folder_path):
		print("files length: ",len(files))
		print("files: ",files)
		for image_path in files:   
			image = cv2.imread(folder_path + "/" + image_path)
			# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			print("image_path: ",image_path)

			# Preprocess image
			prepimage,contours = otsu_preprocess(image)
			# cv2.imshow("PREPIMAGE OTHERS",prepimage)
			# cv2.waitKey(1)
			
			# Initially, image may or may not have contours so let's ceck if it has contours first.
			# Perform additional preprocessing steps to the image if it falls on the else statemtn.
			# Check if image has no contour (contour length = 0)
			# Check if image has a contour but improperly segmented (contour length = 1)
			if(len(contours) == 0 or len(contours) > 10 ):
				print("Gonna delete: ",image_path)
				try:
					os.remove(folder_path + "/" +image_path)
				except:
					continue
			else:
				print("Con leength: ",len(contours))
				contours1 = remove_noise(contours)
				contours2 = remove_coinsides(contours1)
				ave_area = get_ave_area(contours2)
				if(len(contours) == 0):
					try:
						os.remove(folder_path + "/" +image_path)
					except:
						continue
				if(ave_area < 200):
					print("Gonna delete: ",image_path)
					try:
						os.remove(folder_path + "/" +image_path)
					except:
						continue
			# cv2.waitKey(1)

# Replace folder name (must be in the same directory as the code)
# check_neg = check_negatives_general("delme")