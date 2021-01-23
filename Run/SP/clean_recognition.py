#Check feat_extract for bug kasi sumosobra ng isa yung output niya
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



from random import shuffle
def feat_extract(folder_path):

	full_list = []
	data = []
	labels = []
	for root, dirs, files in os.walk(folder_path):
		# print ("root:",root)
		# print ("dirs: ",dirs)
		# print ("files :",files)
		shuffle(files)
		for image_path in files:   
			print("\nimage_path: ",image_path)
			image = cv2.imread(folder_path + "/" + image_path)
			label = image_path[:1] #store as label the first character only of the image path/image name
			print("\nLabels: ", label)
			image = cv2.resize(image, (64,64), interpolation = cv2.INTER_AREA)
			cv2.imshow("resized", image)

			
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
			cv2.imshow("filtered",filtered)
			cv2.imshow("closing",closing)
			cv2.imshow("opening",opening)
			

			_, contours0, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
				cv2.imshow('extracted',crop1)
				
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

 
	return data,labels

if __name__ == '__main__':
	#Change the value inside feat extract into the name of your folder containing the training set (only works if inside Python directory)
	train_data,train_labels = feat_extract("ALL")
	train_data = np.squeeze(train_data) #Equivalent of flatten except that whole array
		
	####convert data and labels into a numpy array 
	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
	print(train_data.shape)
	print (train_labels.shape)

	#TRAINING
	model3 = svm.SVC(gamma = 0.0001, C = 100)
	print("Unique: ", np.unique(train_labels))

	##Reassign the values to X_train and Y_train for name convention purposes
	X_train = train_data 
	Y_train =	train_labels



	print("Y_train: ",Y_train)
	print("Type of X_train: ",type(X_train))
	print("Type of Y_train: ",type(Y_train))
	print(X_train.shape,Y_train.shape)
	model3.fit(X_train,Y_train)

	#----Change the value inside feat extract into the name of your folder containing 
	#---- the testset (only works if inside Python directory)
	X_test,Y_test = feat_extract("tesi")
	X_test = np.squeeze(X_test) #Equivalent of flatten except that whole array
	print("Xtest: ", len(X_test))

	prediction = model3.predict(X_test)
	scoring = model3.score(X_test, Y_test) #Tests the confidence level of the prediction
	print(prediction)	
	print(scoring)

	#=====================================
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	exit(0)