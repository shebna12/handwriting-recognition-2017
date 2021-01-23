import numpy as np 	
import math
# from matplotlib import pyplot as plt
import os
import cv2
import glob
import sys
from utils import resizeTo20,resizeTo28,transform20x20





def preprocess_dataset_general(username):
	# folder_path = "ALL_new"
	# folder_dest = "post_otsu"
	home = os.path.expanduser("~/")

	folder_path = home + "workflow_structure/USERS/"+username+"/training_skews/others/" 

	num = 0

	for root, dirs, files in os.walk(folder_path):
		for image_path in files:   
			image = cv2.imread(folder_path + image_path)
			label = image_path[:1]  
			label = image_path.split(".")[0]   
			# print("label: ",label)
			# import sys
			# sys.exit(0)
			image = resizeTo20(image)
			print("image.shape: ",image.shape)
		
			print("IMAGE NAME:", image_path)
			centered = transform20x20(image)
			fin = resizeTo28(centered)
			print("finshape:" ,fin.shape)
			folder_dest = home + "workflow_structure/USERS/"+username+"/training_temp/"
			cv2.imwrite(folder_dest + "/" + label + "_"+ str(num) +".jpg" ,fin)
			num = num + 1                   

	return


if __name__ == '__main__':
	
	preprocess_dataset("delaft_skew","delaft_fin")