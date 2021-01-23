
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
from utils import get_median_area,i_and_j_preprocessor,resize_img,get_bounding_area,otsu_preprocess,remove_noise,remove_coinsides,get_ave_area,i_and_j_preprocessor


def ave_area_cuts(folder_path):
	# areas = []
	areas = 0
	for root, dirs, files in os.walk(folder_path):
		# print("files length: ",len(files))
		print("files: ",files)
		n = 0
		for image_path in files:
			print("IMG PATH: ",folder_path + "/"+image_path)
			image = cv2.imread(folder_path + "/" +image_path) 
			n = n + 1 
			print("image.shape: ",image.shape)
			h,w,_ = image.shape
			print("hala area: ",h*w)
			curr_area = w*h
			areas = areas + curr_area
		global_ave_area = areas/n
		print("global_ave_area: ",global_ave_area)
	return global_ave_area


def check_negatives(folder_path):
	numfiles = 0
	num = 0
	global_ave_area = ave_area_cuts(folder_path)

	for root, dirs, files in os.walk(folder_path):
		print("files length: ",len(files))
		print("files: ",files)
		for image_path in files:   
			image = cv2.imread(folder_path + "/" + image_path)
			# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			print("image_path: ",image_path)

			# Preprocess image
			prepimage,contours = otsu_preprocess(image)
			cv2.imshow("PREPIMAGE IJ",prepimage)
			# cv2.waitKey(0)
			
			# Initially, image may or may not have contours so let's ceck if it has contours first.
			# Perform additional preprocessing steps to the image if it falls on the else statemtn.
			# Check if image has no contour (contour length = 0)
			# Check if image has a contour but improperly segmented (contour length = 1)
			if(len(contours) != 2 and image_path[0] == "i"):
				print("1Gonna delete: ",image_path)
				try:
					os.remove(folder_path  +image_path)
				except:
					continue
			else:
				print("Con leength: ",len(contours))
				contours1 = remove_noise(contours)
				contours2 = remove_coinsides(contours1)
				ave_area = get_ave_area(contours2)
				if(image_path[0] == "i" and len(contours2) < 2 or ave_area < global_ave_area*0.05):
					print("Less than  2 contours or less than 400 Gonna delete: ",image_path)
					print("cont length: ",len(contours2))
					print("ave_area: ",ave_area)
					try:
						os.remove(folder_path +image_path)
					except:
						continue
			if(len(contours) > 2):
				print("3Gonna delete: ",image_path)
				try:
					os.remove(folder_path  +image_path)
				except:
					continue
			

# Replace folder name (must be in the same directory as the code)
# check_neg = check_negatives("delete")