
import numpy as np 	
import math
# from matplotlib import pyplot as plt
import os
import cv2
import glob
import sys
from utils import resizeTo20,resizeTo28,transform20x20,otsu_preprocess





def preprocess_dataset_ij(username):
	# folder_path = "ALL_new"
	# folder_dest = "post_otsu"

	folder_path = "/home/lincy/workflow_structure/USERS/"+username+"/training_skews/ij/" 
	print("IJ folder_path: ",folder_path)
	num = 0

	for root, dirs, files in os.walk(folder_path):
		for image_path in files:   
			image = cv2.imread(folder_path  + image_path)
			label = image_path[:1]      
			image = resizeTo20(image)
			print("image.shape: ",image.shape)

			prepimage,contours = otsu_preprocess(image)
			# centered = transform20x20(image)
			fin = resizeTo28(prepimage)
			print("finshape:" ,fin.shape)
			folder_dest = "/home/lincy/workflow_structure/USERS/"+username+"/training_temp/"
			print(folder_dest + "/" + label + "_"+ str(num) +".jpg" )
			cv2.imwrite(folder_dest + "/" + label + "_"+ str(num) +".jpg" ,fin)
			num = num + 1                   

	return


if __name__ == '__main__':
	print("I'm in the main and I am doing nothing. Better checkout the code.")
	preprocess_dataset("delaft_skew","delaft_fin")
# Find out why your method doesn't work for all data even if same naman yung shape nila     