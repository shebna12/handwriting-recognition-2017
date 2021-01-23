# Use this code on the python36-32 path. Folder must be in the said path.



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

def skew_dataset(folder_path):
	numfiles = 0
	n=0
	for root, dirs, files in os.walk(folder_path):
			
		for image_path in files:   
			numfiles = numfiles + 1
			print("\nimage_path: ",image_path)
			image = cv2.imread(folder_path + "/" + image_path)
			
			# label = image_path.split("_")[0]
			label = image_path.split(".")[0]
			print(label)
			
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			gray = cv2.bitwise_not(gray)
			 
			# threshold the image, setting all foreground pixels to
			# 255 and all background pixels to 0
			thresh = cv2.threshold(gray, 0, 255,
				cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

			rows,cols = thresh.shape

			# M = cv2.getRotationMatrix2D((cols/2,rows/2),0,1)
			# M = cv2.getRotationMatrix2D((cols/2,rows/2),15,1)
			M = cv2.getRotationMatrix2D((cols/2,rows/2),-15,1)
			

			dst = cv2.warpAffine(thresh,M,(cols,rows))

			backorig = cv2.threshold(dst, 0, 255,
				cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

			# cv2.imwrite("G_new.jpg",backorig)

			h,w = dst.shape[:2]
			ar = w / h
			nw = 500
			nh = int(nw / ar)
			dst = cv2.resize(dst,(nw,nh))

			h,w = thresh.shape[:2]
			ar = w / h
			nw = 500
			nh = int(nw / ar)
			thresh = cv2.resize(thresh,(nw,nh))

			h,w = backorig.shape[:2]
			ar = w / h
			nw = 500
			nh = int(nw / ar)
			backorig = cv2.resize(backorig,(nw,nh))

			
			new_folder = "delaft_skew"
			# name = label + "_"+ "orig"+ "_" + str(n)
			# name = label + "_"+ "pos"+ "_" + str(n)
			name = label + "_"+ "neg"+ "_" + str(n)
			path = "C:/Users/User/AppData/Local/Programs/Python/Python36-32/"
			cv2.imwrite(path + new_folder +"/"+ name + ".jpg", backorig)
			n = n+1
			
			# cv2.waitKey(0)
		print("Number of files: ", numfiles)


 # Replace your "folder name in cut-skew" 
cut_files = skew_dataset("delaft") 