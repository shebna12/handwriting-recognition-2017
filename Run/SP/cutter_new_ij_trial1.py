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
from utils import get_ave_area,resize_img,get_median_area,otsu_preprocess,remove_coinsides,get_area_real_contours,get_bounding_area,i_and_j_preprocessor


def dataset_segmentation(num,name,threshold_area,global_ave_area,image,contours3):
	
	rect = [cv2.boundingRect(c) for c in contours3]
			# label_count = 0
			# allWords = []
			# sort all rect by their y
	rect.sort(key=lambda b: b[1]) 
	# initially the line bottom is set to be the bottom of the first rect
	line_bottom = rect[0][1]+rect[0][3]-1 #y+h amo na ang bottom mo
	line_begin_idx = 0
	p=0
	# print("RECT SIZE: ",len(rect))
	# print("length of rect:",len(rect))
	for ix in range(len(rect)):

 
          
		# when a new box's top is below current line's bottom
		# it's a new line
		# print("line_bottom:", line_bottom)
		if rect[ix][1] > line_bottom: #if y is greater than bottom meaning next line
			# print("-------------LABELS: ",true_labels[label_count])
			prev_contours = rect[line_begin_idx:ix]      
			print("prev_contours length: ",len(prev_contours))
			print("PREV CONTOURS: ",prev_contours)
			current_ave_area = get_bounding_area(prev_contours)
			print("current_ave_area: ",current_ave_area,global_ave_area*0.5)
			if(current_ave_area > global_ave_area*0.5):
				print("I'm not dots")

				line_begin_idx,num = new_line(num,name,global_ave_area,image,rect,line_begin_idx,ix,contours3)
			# print("P: ",p) 
		
		# words,differences,sliding,localCut,globalCut,intersection,contours2 = reset_values(words,differences,sliding,localCut,globalCut,contours2)
		# regardless if it's a new line or not
		# always update the line bottom
		# line_bottom = max(rect[i][1]+rect[i][3]-1, line_bottom)
		line_bottom = (rect[ix][1]+rect[ix][3]-1)
		print("line_bottom: ",line_bottom,ix)
	# label_length = len(true_lab          els)
	# label_count = len(true_labels)

	lastline(num,name,global_ave_area,image,line_begin_idx,rect,contours3)
	# sort the last line

	return num
def new_line(num,name,global_ave_area,image,rect,line_begin_idx,ix,contours3):
	print("Neeeeeeeeeeeeeeeew Liiiiiiiiiiiiiiine")
	rect[line_begin_idx:ix] = sorted(rect[line_begin_idx:ix], key=lambda b: b[0])
	contours3 = rect[line_begin_idx:ix]
	# for c in contours3:
		# cv2.rectangle(image, (c[0], c[1]), (c[2], c[3]), (0, 0 , 0), 6)
	# image = resize_img(image)
	# cv2.imshow("ONE LINE",image)
	# cv2.waitKey(0)
	# import sys
	# sys.exit(0)
	line_begin_idx,num = i_and_j_preprocessor(num,name,image,contours3,global_ave_area,ix,line_begin_idx)
	
	return line_begin_idx,num

         
def lastline(num,name,global_ave_area,image,line_begin_idx,rect,contours3):
	rect[line_begin_idx:] = sorted(rect[line_begin_idx:], key=lambda b: b[0])
	contours3 = rect[line_begin_idx:]
	print("Length of contours3: ",len(contours3))
	_,num = i_and_j_preprocessor(num,name,image,contours3,global_ave_area,0,line_begin_idx)
	print("The end")

def remove_noise(contours0):
	contours = []
	ave_area = get_ave_area(contours0)
	# print("average area: ",ave_area)
	threshold_area = (0.01*(ave_area))
	global_ave_area = ave_area

	# print("threshold area: ",threshold_area)
	# contours = [c for cv2.boundingRect(c) in contours0 if (((c[0]+c[2])*(c[1]+c[3])) >= ave_area)]
	for c in contours0:
		[x,y,w,h] = cv2.boundingRect(c)
		if ((w*h) >= threshold_area ):
			contours.append(c)
	# print("Length of unfiltered contours: ",len(contours0))
	# print("Length of filtered contours: ",len(contours))
	return contours,global_ave_area,threshold_area


def cutter_new(folder_path):
	numfiles = 0
	num=0
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
			contours,global_ave_area,threshold_area = remove_noise(contours)
			# contours = remove_coinsides(contours)




			stop = 0
			while(stop < 2):
				areas = get_area_real_contours(contours)
				rem_i = areas.index(max(areas))
				print(rem_i)
				del contours[rem_i]
				stop = stop + 1
				
			contours2 = remove_coinsides(contours)

			# for c in contours2:
			# 	x, y, w, h = cv2.boundingRect(c)
			# 	cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),6)

			# image = resize_img(image)
			# cv2.imshow("here",image)
			# cv2.waitKey(0)
			# import sys
			# sys.exit(0)


			differences= []
			sliding=[]
			localCut=[]
			globalCut=[]
			words=[]
			path = "C:/Users/User/AppData/Local/Programs/Python/Python36-32/delme/"
			name = path + image_name
			num = dataset_segmentation(num,name,threshold_area,global_ave_area,image,contours2)
			



			# for c in contours2:
			# 	x, y, w, h = cv2.boundingRect(c)
			# 	cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),6)
			# 	height = y+h
			# 	width = x+w
			# 	roi = image[y:height, x:width]
				
			# 	# path = "C:/Users/User/AppData/Local/Programs/Python/Python36-32/deleteafter/"
			# 	# cv2.imwrite(path + image_name + "_m_"+str(n)  + ".jpg", roi)
			# 	n = n+1
			cv2.namedWindow('final',cv2.WINDOW_NORMAL)
			cv2.imshow('final',image)
			cv2.waitKey(0)
		print("Number of files: ", numfiles)




################################################################################################



			# for c in contours:
			# 	x, y, w, h = cv2.boundingRect(c)
			# 	cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),6)

			# cv2.imshow("here",image)
			# cv2.waitKey(0)
			# # import sys
			# # sys.exit(0)


			# average_area = get_ave_area(contours)
			# print("contours length: ",len(contours))
			# print (average_area)
			# for contour in contours:
			# 	[x,y,w,h] = cv2.boundingRect(contour)

			# 	if((w*h) > 0.1*average_area):
			# 		# Uncomment this if you're checking
			# 		# cv2.rectangle(image,(x,y),(x+w,y+h),(180,55,99),6)
					
			# 		height = y+h
			# 		width = x+w
			# 		roi = image[y:height, x:width]
					
			# 		path = "C:/Users/User/AppData/Local/Programs/Python/Python36-32/delme/"
			# 		## Comment this if you are checking
			# 		cv2.imwrite(path + image_name + "_a_"+ str(n)  + ".jpg", roi)
			# 		# cv2.rectangle(image,(x,y),(x+w,y+h),(180,55,99),6)
			# 	n = n+1
			# image = resize_img(image)

			# font  = cv2.FONT_HERSHEY_SIMPLEX
			# bottomLeftCornerOfText = (10,20)
			# fontScale              = 1
			# fontColor              = (0,0,0)
			# lineType               = 2

			# cv2.putText(image,str(image_path), 
			# bottomLeftCornerOfText, 
			# font, 
			# fontScale,
			# fontColor,
			# lineType)
			# cv2.imshow("last",image)


			# cv2.waitKey(1)


# Replace folder name (must be in the same directory as the code)
# cut_files = cutter_new("Shebna_raw_small_print") 
if __name__ == '__main__':
	cut_files = cutter_new("delme2") 

