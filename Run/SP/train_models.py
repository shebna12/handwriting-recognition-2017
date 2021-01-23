#####
####   {'bootstrap': True, 'max_depth': 20, 'n_estimators': 30}
######   0.948034188034
#####
#Find out why OTSU method doesn't work
from random import shuffle
import numpy as np 	
import math
# from matplotlib import pyplot as plt
from sklearn import svm, datasets
import os
# import skimage
# from skimage import io
from PIL import Image
import cv2
import glob
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
# from revised_func import remove_coinsides
import sys
from utils import resizeTo20,resizeTo28,transform20x20
# import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier

# {'C': 100, 'gamma': 0.01}
# 0.927754677755

def cross_validate_score(clf,data,labels):
	scores = cross_val_score(clf,data,labels)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	# return scores
def feat_extract(folder_path):
	num = 0
	data=[]
	labels=[]

	for root, dirs, files in os.walk(folder_path):
		for image_path in files:   
			image = cv2.imread(folder_path + "/" + image_path,0)
			label = image_path[:1]     
			
			fin = image
			# print("Extractin feature....")
			winSize = (28,28)
			blockSize = (8,8)
			blockStride = (4,4)
			cellSize = (4,4)
			nbins = 9
			derivAperture = 1
			winSigma = -1.
			histogramNormType = 0
			L2HysThreshold = 0.2
			gammaCorrection = 1
			nlevels = 28
			signedGradient = True
			 
			hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient) 
			# hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)
			descriptor = hog.compute(fin)  
			
			
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

def rfc_param_selection(X, y, nfolds):
    rfc = RandomForestClassifier(n_jobs=-1)
    param_grid = {
    'bootstrap': [True],
    'max_depth': [5, 10, 15, 20, 22 , 24 , 26],
    'n_estimators': [10, 20, 30]
    }
    grid_search = model_selection.GridSearchCV(estimator = rfc, param_grid=param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    grid_search.best_score_

    # print(grid_search.cv_results_)
    return grid_search.best_params_,grid_search.best_score_
def train_models(username):
	folder_path = "/home/lincy/workflow_structure/USERS/"+username+"/training_final/" 
	train_data,train_labels = feat_extract(folder_path)
	train_data = np.squeeze(train_data) #Equivalent of flatten except that whole array
	
	# ####convert data and labels into a numpy array 
	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
	
	# #Use these train and test values for prediction,training,testing
	# # X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_data, train_labels, test_size=0.20, shuffle=True)
	X_train = train_data
	Y_train = train_labels
	
	svm_clf = svm.SVC(gamma = 0.01, C = 100,probability=True)
	svm_clf.fit(X_train,Y_train)

	rfc_clf = RandomForestClassifier(bootstrap=True,n_jobs=-1,max_depth=20,n_estimators=30)
	rfc_clf.fit(X_train,Y_train)

	score_svm = svm_clf.score(X_train,Y_train)
	score_rfc = rfc_clf.score(X_train,Y_train)

	print("SVM Score: ",score_svm)
	print("Randdom Forest Score: ",score_rfc)

	output_path = "/home/lincy/workflow_structure/USERS/"+username+"/models/"
	joblib_file = output_path + username+"_svm.pkl"  
	joblib.dump(svm_clf, joblib_file)

	joblib_file = output_path + username+"_rfc.pkl"  
	joblib.dump(rfc_clf, joblib_file)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# exit(0)

if __name__ == '__main__':
	train_models("Shebna")
	