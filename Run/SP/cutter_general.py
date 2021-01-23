# Code must ONLY be used if image has been manually cropped to contain only the bond paper part
# DO NOT USE for freshly captured images that have not been cropped yet.




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
from utils import get_median_area, otsu_preprocess,resize_img
from check_negatives_general import check_negatives_general
def cut_dataset(name):
	home = os.path.expanduser("~/")

	folder_path = home + "workflow_structure/USERS/" + name + "/training_clean/others/"
	if not os.path.exists(home + "workflow_structure/USERS/" + name + "/training_cutter/others/"):
		os.mkdir(home + "workflow_structure/USERS/" + name + "/training_cutter/others/")
	numfiles = 0
	
	for root, dirs, files in os.walk(folder_path):
		for image_path in files:   
			n=0
			numfiles = numfiles + 1
			# image_name = image_path[0]
			image_name = image_path.split(".")[0]
			print(image_name)
			# print(im[1])
			print("\nimage_path: ",image_path)
			print("\nimage_name: ",image_name)
			image = cv2.imread(folder_path + "/" + image_path)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



			# _, contours0, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			closing,contours0 = otsu_preprocess(gray)
			temp_image = resize_img(closing)

			sliding = []
			contours2 =[]
			localCut = []
			differences=[]
			globalCut = [] #stores indices of differences that has 50% chance of being a space
			words = []
			contours=[]
			current_area = 0

			temp = 0

			indices=[] #stores values of the contours that are co-inside the contour
			for c in contours0:
				[x, y, w, h] = cv2.boundingRect(c)
				current_area = w*h
				# print("current area: ",current_area)
				temp = temp + current_area
			ave_area = temp/len(contours0)

			threshold_area = (0.05*(ave_area))

			for c in contours0:
				[x,y,w,h] = cv2.boundingRect(c)
				if ((w*h) >= threshold_area ):
					contours.append(c)

			for ic,c in enumerate(contours):
				x, y, w, h = cv2.boundingRect(c)
				
			for c in contours:
				x, y, w, h = cv2.boundingRect(c)

			indices=[] #stores values of the contours that are co-inside the contour
			for c in contours:
				[x, y, w, h] = cv2.boundingRect(c)
				for index,cn in enumerate(contours):
					[i,j,k,l] = cv2.boundingRect(cn)
					if ((i < (x+w) and (i > x)) and ((i+k) < (x+w) and (i+k) > x )):
						if((((j+l) < (y+h)) and ((j+l) > y )) and ((j < (y+h)) and (j > y))):
							indices.append(index)

			contours2 = [c for i,c in enumerate(contours) if i not in indices]
			print("Length of contours2: ",len(contours2))
			for c in contours2:
				x, y, w, h = cv2.boundingRect(c)
				# cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),6)


				

				height = y+h
				width = x+w
				roi = image[y:height, x:width]
				
				path = home + "workflow_structure/USERS/"+name+"/training_cutter/others/"
				cv2.imwrite(path + image_name + "_m_"+str(n)  + ".jpg", roi)
				n = n+1

		print("Number of files: ", numfiles)

	check_negatives_general(path)

# Replace folder name (must be in the same directory as the code)
# cut_files = cut_dataset("Shebna")