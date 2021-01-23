from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn import model_selection
from random import shuffle
import numpy as np 	
import math
from matplotlib import pyplot as plt
from sklearn import svm, datasets
import os
import skimage
from skimage import io
from PIL import Image
import cv2
import glob
from revised_func import remove_coinsides,preprocess,remove_noise
# from tune2 import feat_extract

def feat_extract(image,Y_test):
	data = []
	labels = []
	label = Y_test
	# print("\nimage_path: ",image_path)
	image = cv2.imread("w.jpeg")
	label = "B" #store as label the first character only of the image path/image name
	# print("\nLabels: ", label)
	image = cv2.resize(image, (64,64), interpolation = cv2.INTER_AREA)
	# cv2.imshow("resized", image)

	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


	# Smooth image
	blur = cv2.GaussianBlur(gray, (3, 3), 0)
	filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 77, 3)

	# Some morphology to clean up image,originally 7,7
	kernel = np.ones((3,3), np.uint8)
	opening = cv2.morphologyEx(filtered,cv2.MORPH_OPEN, kernel, iterations = 1)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations =2)
	# kernel2 = np.ones((17,17),np.uint8)
	# closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel2, iterations =1)
	# closing = opening
	# # cv2.namedWindow('closing',cv2.WINDOW_NORMAL)
	# # cv2.namedWindow('res',cv2.WINDOW_NORMAL)
	cv2.namedWindow('filtered',cv2.WINDOW_NORMAL)
	cv2.namedWindow('opening',cv2.WINDOW_NORMAL)
	# cv2.imshow("filtered",filtered)
	# cv2.imshow("closing",closing)
	# cv2.imshow("opening",opening)
	

	_, contours0, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contours0 = remove_coinsides(contours0)
	sliding = []
	contours2 =[]
	localCut = []
	differences=[]
	globalCut = [] #stores indi   ces of differences that has 50% chance of being a space
	words = []
	contours=[]
	current_area = 0

	temp = 0

	indices=[] #stores values of the contours that are co-inside the contour
	print("Num of contours before getting the ROI contour: ",len(contours0))
	for c in contours0:
		[x, y, w, h] = cv2.boundingRect(c)
		current_area = w*h
		print("current area: ",current_area)
		temp = temp + current_area
	ave_area = temp/len(contours0)
	
	threshold_area = (0.05*(ave_area))
	
	for c in contours0:
		[x,y,w,h] = cv2.boundingRect(c)
		if ((w*h) >= threshold_area ):
			contours.append(c)
	
	for ic,c in enumerate(contours):
		x, y, w, h = cv2.boundingRect(c)
		
		cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 4)
	

	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		
		cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 2)


	indices=[] #stores values of the contours that are co-inside the contour
	for c in contours:
		[x, y, w, h] = cv2.boundingRect(c)
		for index,cn in enumerate(contours):
			[i,j,k,l] = cv2.boundingRect(cn)
			if ((i < (x+w) and (i > x)) and ((i+k) < (x+w) and (i+k) > x )):
				if((((j+l) < (y+h)) and ((j+l) > y )) and ((j < (y+h)) and (j > y))):
					indices.append(index)
					
	
	contours2 = [c for i,c in enumerate(contours) if i not in indices]
	for c in contours2:
		x, y, w, h = cv2.boundingRect(c)
		print(x,y,w,h)


#####------START OF MACHINE LEARNING AND FEATURE EXTRACTION-------####
	print("CLOSING SHAPE: ",closing.shape)
	print("Number of ROI: ",len(contours2))
	if(len(contours2) > 1):
		[x, y, w, h] = cv2.boundingRect(c)
		cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 4)
		# cv2.imshow("ROI",image)

	for c in contours2:
		[x, y, w, h] = cv2.boundingRect(c)
		if(h > w):
			crop1 = closing[y:y+h, x:(x+w)]
			diff = (h)-(w)
			padleft = math.ceil(diff/2)
			padright = math.floor(diff/2)
			# padEx = (math.ceil(diff/2) - diff/2)
			padded_img = np.zeros((h,h),np.uint8)
			# print("diff: ",diff)
			# print("pad: ",padleft)
			# print("pad: ",padright)
			# # print("padEx: ",padEx)
			padded_img[:,padleft:h-padright] = crop1
		elif(h < w):
			crop1 = closing[y:y+h, x:(x+w)]
			diff = w-h
			padtop = math.ceil(diff/2)
			padbottom = math.floor(diff/2)
			padded_img = np.zeros((w,w),np.uint8)
			padded_img[padtop:w-padbottom,:] = crop1
		else:
			padded_img = closing
		crop1 = cv2.resize(padded_img,(64,64))
		print("CLOSING SHAPE: ",crop1.shape)

		cv2.imshow('extracted',crop1)
		cv2.imshow("CROP1: ",crop1)
		winSize = (64,64)
		blockSize = (16,16)
		blockStride = (8,8)
		cellSize = (8,8)
		nbins = 9
		derivAperture = 1
		winSigma = -1.
		histogramNormType = 0
		L2HysThreshold = 0.2
		gammaCorrection = 1
		nlevels = 64
		signedGradient = True
		 
		hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient) 
		# hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)
		descriptor = hog.compute(crop1)  
		
		
		data.append(descriptor)
		##Labels contains all the labels of the image ..see line 46
		labels.append((label))
		# print("LABELS:",labels)
		# import sys; sys.exit(0)
#  # ensemble
	return data,labels
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 100]
    gammas = [0.001, 0.01, 0.1, 10]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = model_selection.GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    grid_search.best_score_
    # print(grid_search.cv_results_)
    return grid_search.best_params_,grid_search.best_score_

def otsu_preprocess(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	prepimage = cv2.threshold(gray, 0, 255,
				cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	_, contours, hierarchy = cv2.findContours(prepimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	return contours,prepimage

if __name__ == '__main__':

	image = cv2.imread("PIXIE.jpg")
	# true_labels = ['W','W','W','W']

	true_labels = ['P','I','X','I','E']
	h,w = image.shape[:2]
	ar = w / h
	nw = 600
	nh = int(nw / ar)
	image = cv2.resize(image,(nw,nh))
	
	# insert preprocessing step here
	
	contours,prepimage = otsu_preprocess(image)
	contours1 = remove_noise(contours)
	contours2 = remove_coinsides(contours)
	contours3 = [cv2.boundingRect(c) for c in contours2]
	contours3.sort(key=lambda b: b[0])
	print("Sorted contours: ",contours3)
	print(len(contours2))

	i=0
	for c in contours3:
		if(i < len(contours2)):
		# x,y,w,h = cv2.boundingRect(c)
			x,y,w,h = c[0],c[1],c[2],c[3]
			print(x,y,w,h)
			cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 2)
			letter = prepimage[y:y+h,x:x+w]
			# print('>>',letter.shape)
			letter = cv2.bitwise_not(letter)
			cv2.imwrite("word.jpeg",letter)
			cv2.imshow("letra",letter)
			
			# letter = cv2.cvtColor(letter,cv2.COLOR_GRAY2BGR)
			X_test,Y_test = feat_extract(letter,true_labels[0])
			X_test = np.squeeze(X_test)
			Y_test = np.squeeze(Y_test)
			X_test = X_test.reshape(1,-1)
			Y_test = Y_test.reshape(1,-1)
			# # X_test = X_test[0].transpose()
			# X_test = X_test[0].reshape((1,1764))
			# Y_test = np.array(Y_test)
			# print(len(Y_test))
			# print(type(Y_test[0]))
			print(X_test.shape)
			# print(image.shape)
			# print(X_test[0].shape)
			# import sys; sys.exit()
			# X_test = X_test.reshape()
			# print(X_test.shape)
			# print(Y_test)
		# cv2.imshow("image",image)
			# break
			loaded_model = joblib.load("clf_svm.pkl")
			# print(loaded_model.__dict__.keys())
			# import sys; sys.exit()
			pred = loaded_model.predict(X_test)
			result = loaded_model.score(X_test, Y_test)
			print(result)
			print(pred)
			break
			cv2.imshow("letter",letter)
			cv2.waitKey(0)
		i = i + 1
		
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	exit(0) 