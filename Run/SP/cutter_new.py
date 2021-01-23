# Code must ONLY be used if image has been checked of perfect segmentation
# Check out the ## comments as a guide for checking perfect segmentation
# Instructions:
# --Input folder path of training images that are freshly captured
# -- Check out the ## comments
# -- Perfect segmentation means that there are no unwanted blobs captured by the cv2.rectangle



import glob
import cv2
import numpy as np 	
import math
from matplotlib import pyplot as plt
from sklearn import svm, datasets
import os
import skimage
from skimage import io
from PIL import Image
from utils import get_ave_area,resize_img,get_median_area,otsu_preprocess,remove_coinsides,get_area_real_contours


def remove_noise(contours0):
	contours = []
	ave_area = get_ave_area(contours0)
	# print("average area: ",ave_area)
	threshold_area = (0.8*(ave_area))
	# print("threshold area: ",threshold_area)
	# contours = [c for cv2.boundingRect(c) in contours0 if (((c[0]+c[2])*(c[1]+c[3])) >= ave_area)]
	for c in contours0:
		[x,y,w,h] = cv2.boundingRect(c)
		if ((w*h) >= threshold_area ):
			contours.append(c)
	# print("Length of unfiltered contours: ",len(contours0))
	# print("Length of filtered contours: ",len(contours))
	return contours


def cutter_new(folder_path):
	numfiles = 0
	
	for root, dirs, files in os.walk(folder_path):
			
		for image_path in files:   
			n=0
			numfiles = numfiles + 1
			image_name = image_path.split(".")[0]
			# image_name = image_path[0]
			print("\nimage_path: ",image_path)
			image = cv2.imread(folder_path + "/" + image_path)
			
			# label = image_path.split("_")[0]
			# print(label)


			gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
			_,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY) 
			nimage = resize_img(thresh)
			# thresh = cv2.bitwise_not(thresh)
			thresh,contours = otsu_preprocess(thresh)

			# nimage = resize_img(thresh)
			# cv2.imshow("thresh",nimage)
			# image = thresh
			# cv2.waitKey(0)
			# _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours
			contours = remove_noise(contours)
			# contours = remove_coinsides(contours)
			stop = 0
			while(stop < 2):
				areas = get_area_real_contours(contours)
				rem_i = areas.index(max(areas))
				print(rem_i)
				del contours[rem_i]
				stop = stop + 1
				
			contours = remove_coinsides(contours)

			# for c in contours:
			# 	x, y, w, h = cv2.boundingRect(c)
			# 	cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),6)

			# cv2.imshow("here",image)
			# cv2.waitKey(0)



			average_area = get_ave_area(contours)
			print("contours length: ",len(contours))
			print (average_area)
			for contour in contours:
				[x,y,w,h] = cv2.boundingRect(contour)

				if((w*h) > 0.1*average_area):
					# Uncomment this if you're checking
					# cv2.rectangle(image,(x,y),(x+w,y+h),(180,55,99),6)
					
					height = y+h
					width = x+w
					roi = image[y:height, x:width]
					
					path = "C:/Users/User/AppData/Local/Programs/Python/Python36-32/delme/"
					## Comment this if you are checking
					cv2.imwrite(path + image_name + "_a_"+ str(n)  + ".jpg", roi)
					# cv2.rectangle(image,(x,y),(x+w,y+h),(180,55,99),6)
				n = n+1
			image = resize_img(image)

			font  = cv2.FONT_HERSHEY_SIMPLEX
			bottomLeftCornerOfText = (10,20)
			fontScale              = 1
			fontColor              = (0,0,0)
			lineType               = 2

			cv2.putText(image,str(image_path), 
			bottomLeftCornerOfText, 
			font, 
			fontScale,
			fontColor,
			lineType)
			cv2.imshow("last",image)


			cv2.waitKey(1)


# Replace folder name (must be in the same directory as the code)
# cut_files = cutter_new("Shebna_raw_small_print") 
cut_files = cutter_new("delme2") 

