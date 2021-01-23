from sklearn import model_selection
from sklearn.externals import joblib
from random import shuffle
import numpy as np 	
import math
import sys
import cv2
from utils import remove_coinsides,remove_noise,sort_LR,get_max_area_contour
from spellchecker import correction
import string
import heapq
import itertools as it

# Use for getting the name of the file
# Name of the file will be used as a true label
class ImageName:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name)
        self.__name = img_name

    def __str__(self):
        return self.__name

# Preprocessing technique using OTSU
def otsu_preprocess(image):	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	prepimage = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	_, contours, hierarchy = cv2.findContours(prepimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	return prepimage,contours

def filter_contours(image,contours):
	chosen_contour = []
	print("Length of contours: ",len(contours))
	# cv2.imshow("IMAGE1:  ",image)
	cv2.waitKey(0)
	
	chosen_contour = get_max_area_contour(contours)
	new_img = np.zeros((image.shape), np.uint8)
	
	cv2.drawContours(new_img,[chosen_contour],0,(255,255,255),-1)
	cv2.imshow("newimg",new_img)
	cv2.waitKey(0)
	image = new_img
	
	return image,chosen_contour

# Displays the contours on the input image
def show_contours(contours,image):
	areas=[]
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		print(x,y,w,h)
		cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 2)
		cv2.imshow("IMAGE",image)
		cv2.waitKey()
def feat_extract(image,Y_test):
	data = []
	labels = []
	label = Y_test

	_, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contours2 = remove_coinsides(contours)
	

#####------START OF MACHINE LEARNING AND FEATURE EXTRACTION-------####

	print("Number of ROI: ",len(contours2))


	# Uncomment for error checking if more than 1 roi was detected for recognition
	# Feature extractor must only detect 1 roi
	if(len(contours2) > 1):
		for c in contours2:
			[x, y, w, h] = cv2.boundingRect(c)
			cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 4)
			# cv2.imshow("ROI",image)
			print("Your image has more than 1 ROI.")
		image,contours = filter_contours(image,contours2)
		contours2 = []
		contours2.append(contours)
		
			# sys.exit()



	# Pad image just in case it is not 64x64
	print("hahaha: ",len(contours2))
	#  sys.exit() 
	for c in contours2:
		[x, y, w, h] = cv2.boundingRect(c)
		if(h > w):
			crop1 = image[y:y+h, x:(x+w)]
			diff = (h)-(w)
			padleft = math.ceil(diff/2)
			padright = math.floor(diff/2)
			padded_img = np.zeros((h,h),np.uint8)
			
			padded_img[:,padleft:h-padright] = crop1
		elif(h < w):
			crop1 = image[y:y+h, x:(x+w)]
			diff = w-h
			padtop = math.ceil(diff/2)
			padbottom = math.floor(diff/2)
			padded_img = np.zeros((w,w),np.uint8)
			padded_img[padtop:w-padbottom,:] = crop1
		else:
			padded_img = image
		crop1 = cv2.resize(padded_img,(64,64))
		cv2.imshow("CROP1",crop1)
		cv2.waitKey(0)
		# Set parameters for HOG
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
		##Labels contains all the labels of the image
		labels.append((label))
		print("LABEL: ",label)
	return data,labels


def check(entry):
	
	import mmap

	f = open('bigger.txt')
	s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
	b = entry.encode()
	if s.find(b) != -1:
		return True
	else:
		return False

# print check()


# Calculates the score for all the letters.
def calc_total_score(true_labels,Ppred):
	correct = 0
	print("ALL PREDS: ",Ppred)
	for i,label in enumerate(true_labels):
		if label == Ppred[i]:
			correct = correct + 1
	fscore = correct/len(Ppred)
	print("FINAL SCORE: ", fscore)

def disp_word(Ppred):
	finalString = ''.join(Ppred)
	print(finalString)
	return finalString

def find_proba_of(pred_item,proba_all,set_of_words):
	alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	
	pred_string = np.array_str(pred_item)
	# print("pred_string: ",pred_string)
	# print("length:",len(pred_string))
	label_index = alphabet.index(pred_string)
	print("label_index: ",label_index)
	prob = proba_all[0][label_index]
	# print("proba: ",prob)
	print("proba of {} : {} " .format(pred_string,prob))
	if(prob < 0.60):
		possibles = sorted(zip(proba_all[0], alphabet), reverse=True)[:3]
		set_of_words.append(possibles)
	else:
		possibles = sorted(zip(proba_all[0], alphabet), reverse=True)[:1]
		set_of_words.append(possibles)

	return set_of_words

		# largest_integers = heapq.nlargest(3, proba_all[0].tolist())
		# print("largest integers: ",largest_integers )


def process_possible_words(set_of_words):
	groupLetters = []
	for i in set_of_words:
		letters = []
		print("i: ",i )
		for j in i:
			print("j: ",j[1])
			letters.append(j[1])
		groupLetters.append(letters)
	print("GROUP: ",groupLetters)
	calc_total_score(true_labels,Ppred)
	final_word = disp_word(Ppred)
	final_word = final_word.lower()
	print(correction(final_word))
	words=[]
	products = it.product(*groupLetters)
	# print("PRODUCTS: ",products)
	for prod in products:
		print(prod)
		word = ''.join(prod)
		words.append(word)
	words = [x.lower() for x in words]
	print("WORDS: ",words)
	for word in words:
		if(check(word)):
			final_word = word
			break
	return final_word
	

if __name__ == '__main__':
	Ppred = []
	set_of_words=[]
	x = ImageName("BEHIND.jpg") # Change the file name
	image_filename = str(x)
	image = cv2.imread(image_filename)
	image_string = image_filename.split('.')[0]
	true_labels = list(image_string)
	
	# Resize image so it'll fit on screen
	h,w = image.shape[:2]
	ar = w / h
	nw = 600 
	nh = int(nw / ar)
	image = cv2.resize(image,(nw,nh))
	

	#--------Preprocessing starts here ------#
	prepimage,contours = otsu_preprocess(image)
	contours1 = remove_noise(contours)
	contours2 = remove_coinsides(contours1)
	print("Length of contours2: ",len(contours2))
	show_contours(contours2,image)
	# sys.exit()
	
		# contours2 = filter_contours(contours2)
	show_contours(contours2,image)


	# Get the corresponding x,y,w,h points of each contour
	# This is necessary for sorting the letters L-R
	contours3 = [cv2.boundingRect(c) for c in contours2]

	# Contours are sorted L-R for proper prediction results
	contours3 = sort_LR(contours3)


	i=0
	for c in contours3: 
		if(i < len(contours2)):
		
			x,y,w,h = c[0],c[1],c[2],c[3]
			print(x,y,w,h)
			cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 2)
			letter = prepimage[y:y+h,x:x+w]
			cv2.imshow("letter",letter)
			cv2.waitKey(0)
			
			X_test,Y_test = feat_extract(letter,true_labels[i])
			
			# Transform to numpy array
			X_test = np.squeeze(X_test)
			Y_test = np.squeeze(Y_test)
			
			# Reshape to 1,-1 because there is only a single sample
			X_test = X_test.reshape(1,-1)
			Y_test = Y_test.reshape(1,-1)
			
			# Load the model we previously trained
			# This uses the joblib library from sklearn
			loaded_model = joblib.load("clf_svm.pkl")
			# print("num of X_train: ",len(X_train))          
			pred = loaded_model.predict(X_test)
			result = loaded_model.score(X_test, Y_test)
			proba = loaded_model.predict_proba(X_test)
			print("pred: ",pred)
			print("result:",result)
			print("proba: ",proba)

			set_of_words = find_proba_of(pred[0],proba,set_of_words)
			print("set_of_words: ",set_of_words)


			# Compile every single recognition to a list
			Ppred.append(pred.tolist()[0])
			
			cv2.imshow("letter",letter)
			cv2.waitKey(0)
		i = i + 1
	final_word = process_possible_words(set_of_words)
	
			# might have more values so don't break it instead put it in an array 


	print("Final word is: ",final_word)
	# insert .lower function so we can compare it with the spellchec ker
		
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	exit(0) 