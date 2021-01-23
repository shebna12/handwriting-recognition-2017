# Use this code on the python36-32 path. Folder must be in the said path.



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

def skew_dataset(folder,username,marker,value):
	folder_path = "/home/lincy/workflow_structure/USERS/"+username+"/training_cutter/" + folder 

	numfiles = 0
	n=0
	end_flag = 0
	while(end_flag < 3):
		skew_val = value[end_flag]
		mark_string = marker[end_flag]
		print("skew_val: ",skew_val)
		print("mark_string: ",mark_string)
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
				M = cv2.getRotationMatrix2D((cols/2,rows/2),skew_val,1)
				

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

				
				# new_folder = "delaft_skew"
				# name = label + "_"+ "orig"+ "_" + str(n)
				# name = label + "_"+ "pos"+ "_" + str(n)
				name = label + "_"+ mark_string+ "_" + str(n)
				path = "/home/lincy/workflow_structure/USERS/"+username+"/training_skews/" + folder +"/"
				print("STORED FILE AT: ",path+name+".jpg")
				cv2.imwrite(path + name + ".jpg", backorig)
				n = n+1
				
				# cv2.waitKey(0)
			print("Number of files: ", numfiles)
			end_flag = end_flag + 1

def initialize_skew(username):
	if not os.path.exists("/home/lincy/workflow_structure/USERS/" + username + "/training_skews/ij/"):
		os.mkdir("/home/lincy/workflow_structure/USERS/" + username + "/training_skews/ij/")
	if not os.path.exists("/home/lincy/workflow_structure/USERS/" + username + "/training_skews/others/"):
		os.mkdir("/home/lincy/workflow_structure/USERS/" + username + "/training_skews/others/")
	marker = ["orig","pos","neg"]
	value = [0,10,-10]
	cut_files = skew_dataset("others",username,marker,value)
	cut_files = skew_dataset("ij",username,marker,value)  

if __name__ == '__main__':
# Replace your "folder name in cut-skew" 
	folder_path = ""
	initalize_skew(folder_path)
