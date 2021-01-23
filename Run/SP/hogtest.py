'''
Problems:
	letter i and j must be contoured as one box
	difference between two adjacent contours looks off

'''

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


# def coords_to_feature_map(corners):
# 	print("corners: ",corners)
# 	img = np.zeros((64,64, 3), np.uint8)
	
# 	blank = np.zeros((256, 256), np.uint8)
	
# 	for c in corners:
# 		x,y = c.ravel()

# 		cv2.circle(blank, (x,y), 3, (80, 250, 150), -1)
# 		cv2.imshow('with corners', blank)

# 		print("x: ",x)
# 		print("y: ",y)
# 		img[x,-y] = 255
# 	cv2.imshow("hola",img)
# 	return img 

full_list = []

# path = "A*.jpg"
image_paths = glob.glob("*.jpg")
print ("image path:",image_paths)
for image_path in image_paths:   
	image = cv2.imread(image_path)
	label = image_path[:1]
	# print("\nLabels: ", label)

	# h,w = image.shape[:2]
	# ar = w / h
	# nw = 1300
	# nh = int(nw / ar)
	# image = cv2.resize(image,(nw,nh))

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


	# Smooth image
	blur = cv2.GaussianBlur(gray, (3, 3), 0)
	filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 77, 3)

	# Some morphology to clean up image
	kernel = np.ones((7,7), np.uint8)
	opening = cv2.morphologyEx(filtered,cv2.MORPH_OPEN, kernel, iterations = 1)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations =2)
	kernel2 = np.ones((17,17),np.uint8)
	closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel2, iterations =1)
	# cv2.namedWindow('closing',cv2.WINDOW_NORMAL)
	# # cv2.namedWindow('res',cv2.WINDOW_NORMAL)
	cv2.namedWindow('filtered',cv2.WINDOW_NORMAL)
	cv2.namedWindow('opening',cv2.WINDOW_NORMAL)
	cv2.imshow("filtered",filtered)
	cv2.imshow("closing",closing)
	cv2.imshow("opening",opening)

	_, contours0, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	sliding = []
	contours2 =[]
	localCut = []
	differences=[]
	globalCut = [] #stores indices of differences that has 50% chance of being a space
	words = []
	contours=[]
	current_area = 0

	temp = 0
	data = []
	labels = []
	indices=[] #stores values of the contours that are co-inside the contour

	for c in contours0:
		[x, y, w, h] = cv2.boundingRect(c)
		current_area = w*h
		# print("current area: ",current_area)
		temp = temp + current_area
	ave_area = temp/len(contours0)
	# print("average area: ",ave_area)
	threshold_area = (0.05*(ave_area))
	# print("threshold area: ",threshold_area)
	# contours = [c for cv2.boundingRect(c) in contours0 if (((c[0]+c[2])*(c[1]+c[3])) >= ave_area)]
	for c in contours0:
		[x,y,w,h] = cv2.boundingRect(c)
		if ((w*h) >= threshold_area ):
			contours.append(c)
	# print("Length of unfiltered contours: ",len(contours0))
	# print("Length of filtered contours: ",len(contours))
	for ic,c in enumerate(contours):
		x, y, w, h = cv2.boundingRect(c)
		# print(x,y,w,h)
		cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 4)
		# cv2.putText(image, str(ic), (x - 10, y - 10),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		# print(x,y,w,h)
		cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 2)


	indices=[] #stores values of the contours that are co-inside the contour
	for c in contours:
		[x, y, w, h] = cv2.boundingRect(c)
		for index,cn in enumerate(contours):
			[i,j,k,l] = cv2.boundingRect(cn)
			if ((i < (x+w) and (i > x)) and ((i+k) < (x+w) and (i+k) > x )):
				# print("inside")
				if((((j+l) < (y+h)) and ((j+l) > y )) and ((j < (y+h)) and (j > y))):
			
					indices.append(index)
					# contours.remove(cn)
	# print("num of contours",len(contours))
	# print("# of indices:",len(indices))
	# print ('1st: Number of contours are: %d -> ' %len(contours))

	contours2 = [c for i,c in enumerate(contours) if i not in indices]
	# print ('2nd: Number of contours are: %d -> ' %len(contours2))
	for c in contours2:
		x, y, w, h = cv2.boundingRect(c)
		print(x,y,w,h)
	count = 0 
	print("Length of contours2: ",len(contours2))
	for c in contours2:
		[x, y, w, h] = cv2.boundingRect(c)
		crop = closing[y:y+h, x:x+w]
		crop = cv2.resize(crop,(64,64))
		cv2.imshow('extracted',crop)
		blank = np.zeros((64, 64), np.uint8)


		corners = cv2.goodFeaturesToTrack(crop, 8, 0.02, 10)
		corners = np.int0(corners)
		# img_feed = coords_to_feature_map(corners[0])
		img = np.zeros((64,64, 3), np.uint8)
		print("corners lenght: ",len(corners))
		for corner in corners: 
			x, y = corner.ravel()
			cv2.circle(crop, (x, y), 3, (80,250, 150), -1)
			cv2.imshow('with corners',crop)

			print("x: ",x)
			print("y: ",y)
			img[-x,y] = 255
			cv2.imshow("hola",img)
 
	cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
		
			
	

	




#####Comment temporarily
# model = svm.SVC(gamma = 0.0001, C = 100)
# # print("Labels", label)
# print("Unique: ", np.unique(Y_train))
# X_train = (np.array(X_train)).reshape(-1,1)
# Y_train =	(np.array(Y_train)).reshape(-1,1).ravel()
# print("Y_train: ",Y_train)
# x,y = X_train[:2],Y_train[:2]
# model.fit(x,y)
#####Comment end temporarily
# model.score(Xlist, Ylist)

# print("Prediction: ", model.predict(X_train[:2])) 
		# predicted = model.predict()
#=====================================


