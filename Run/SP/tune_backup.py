
#Find out why OTSU method doesn't work
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
from sklearn import model_selection
from sklearn.externals import joblib
from revised_func import remove_coinsides

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
			image = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
			# cv2.imshow("resized", image)

			
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			closing = cv2.threshold(gray, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]	
			# ret, thresh = cv2.threshold(thresh,127,255,0)

			# Smooth image
			# blur = cv2.GaussianBlur(gray, (3, 3), 0)
			# filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 77, 3)

			# # Some morphology to clean up image,originally 7,7
			# kernel = np.ones((3,3), np.uint8)
			# opening = cv2.morphologyEx(filtered,cv2.MORPH_OPEN, kernel, iterations = 1)
			# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations =2)
			# # kernel2 = np.ones((17,17),np.uint8)
			# # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel2, iterations =1)
			# # closing = opening
			# # # cv2.namedWindow('closing',cv2.WINDOW_NORMAL)
			# # # cv2.namedWindow('res',cv2.WINDOW_NORMAL)
			# cv2.namedWindow('filtered',cv2.WINDOW_NORMAL)
			# cv2.namedWindow('opening',cv2.WINDOW_NORMAL)
			# # cv2.imshow("filtered",filtered)
			# # cv2.imshow("closing",closing)
			# # cv2.imshow("opening",opening)
			

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

 # ensemble
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
if __name__ == '__main__':
	#Change the value inside feat extract into the name of your folder containing the training set (only works if inside Python directory)
	train_data,train_labels = feat_extract("ALL")
	train_data = np.squeeze(train_data) #Equivalent of flatten except that whole array
		
	####convert data and labels into a numpy array 
	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
	print(train_data.shape)
	print (train_labels.shape)


	#Use these train and test values for prediction,training,testing
	# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_data, train_labels, test_size=0.20, shuffle=True)

	# Try the best parameters
	# best_params,best_score = svc_param_selection(X_train,Y_train,10)
	# print(best_params)
	# print(best_score)
	# TRAINING
	print("X_train: ",X_train.shape)
	# clf2 = svm.SVC(gamma = 0.01, C = 100)
	# print("Unique: ", np.unique(train_labels))


	#Uncomment this if you need to train your model with the best params value 
	# clf2.fit(X_train,Y_train)
	
	
	X_test,Y_test = feat_extract("tesi")
	X_test = np.squeeze(X_test) #Equivalent of flatten except that whole array
	print("Xtest: ", len(X_test))
	print(X_test.shape)
	# print(X_test[0])
	# print("Ytest: ",Y_test[0])
	# import sys; sys.exit()
	#======  Uncomment this if you are going to test your model with the model  
	# prediction = clf2.predict(X_test[0].reshape(1,-1))
	# scoring = clf2.score(X_test[0].reshape(1,-1), Y_test[0].reshape(1,-1))

	# joblib_file = "clf2_model.pkl"  
	# joblib.dump(clf2, joblib_file)
	# joblib.dump(clf2, joblib_file)
	# print(Y_test)
	# print(prediction) 
	# print(scoring)

	# uncomment if you are going to train only one sample\
	X_test = np.squeeze(X_test)
	Y_test = np.squeeze(Y_test)
	X_test = X_test.reshape(1,-1)
	Y_test = Y_test.reshape(1,-1)


	loaded_model = joblib.load("new_model.pkl")
	# print(loaded_model.__dict__.keys())
	# import sys; sys.exit()
	pred = loaded_model.predict(X_test)
	print("Ytest: ", Y_test)
	print(pred)
	result = loaded_model.score(X_test, Y_test)
	print(result)
	        # =====================================
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	exit(0)