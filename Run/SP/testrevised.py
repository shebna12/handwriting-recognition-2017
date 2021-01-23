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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
import os
import skimage
from skimage import io
from PIL import Image
X_train = []
path = "SP Data Set/training/*.jpg"
image_paths = glob.glob(path)
for image_path in image_paths:   
	image = cv2.imread(image_path)
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
	# cv2.namedWindow('filtered',cv2.WINDOW_NORMAL)
	# cv2.namedWindow('opening',cv2.WINDOW_NORMAL)
	# cv2.imshow("filtered",filtered)
	# cv2.imshow("closing",closing)
	# cv2.imshow("opening",opening)

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
	print("Length of unfiltered contours: ",len(contours0))
	print("Length of filtered contours: ",len(contours))
	for ic,c in enumerate(contours):
		x, y, w, h = cv2.boundingRect(c)
		print(x,y,w,h)
		cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 4)
		# cv2.putText(image, str(ic), (x - 10, y - 10),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		print(x,y,w,h)
		cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 2)


	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		print(x,y,w,h)
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
	
	count = 0 
	print("Length of contours2: ",len(contours2))
	for c in contours2:
		count = count + 1
		print("count: ",count)

		[x, y, w, h] = cv2.boundingRect(c)
		crop = closing[y:y+h, x:x+w]
		crop = cv2.resize(crop,(256,256))
		cv2.imshow('extracted',crop)
		blank = np.zeros((256, 256, 3), np.uint8)

		corners = cv2.goodFeaturesToTrack(crop, 5, 0.2, 80)
		if(count == 28):
			print(image_path)
		corners = np.int0(corners)


		for corner in corners: 
			x, y = corner.ravel()
			cv2.circle(blank, (x, y), 3, (80,250, 150), -1)
			# cv2.imshow('with corners',blank)
		
			
	

	print("CORNERS: ", corners)
	X_train.append(corners)
	print("X_train: ",X_train)
Xlist = []
Ylist = []


for xtr in X_train:
	label = os.path.split(image_path[1].split("-")[0])

	print("\nLabels: ", label)

	featurevector = np.array(pix).flatten()
	Xlist.append(featurevector)
	Ylist.append(label)

model = svm.SVC(gamma = 0.0001, C = 100)

# print("Labels", label)
print("Unique: ", np.unique(Ylist))

model.fit(Xlist, Ylist)
model.score(Xlist, Ylist)

print("Prediction: ", model.predict(Xlist[3])) 
		# predicted = model.predict()
#=====================================
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)

