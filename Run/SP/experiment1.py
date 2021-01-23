	
from utils import remove_coinsides, show_contours, otsu_preprocess, resize_img, get_ave_area, remove_noise


import numpy as np 	
import math
from matplotlib import pyplot as plt
import os
import cv2
import glob
import sys
# from utils import resizeTo20,resizeTo28,transform20x20
class ImageName:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name) 
        self.__name = img_name

    def __str__(self):
        return self.__name



# Smooth image
def preprocess(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (3, 3), 0)
	filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 77, 3)

	# Some morphology to clean up image
	kernel = np.ones((3,3), np.uint8)
	opening = cv2.morphologyEx(filtered,cv2.MORPH_OPEN, kernel, iterations = 1)
	# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations =1)
	# kernel2 = np.ones((5,5),np.uint8)
	# closing = cv2.dilate(closing,kernel2,iterations = 1)
	# kernel2 = np.ones((3,3), np.uint8)
	kernel2 = np.ones((3,3),np.uint8)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2, iterations =1)
	cv2.namedWindow('closing',cv2.WINDOW_NORMAL)
	cv2.namedWindow('res',cv2.WINDOW_NORMAL)
	cv2.namedWindow('filtered',cv2.WINDOW_NORMAL)
	cv2.namedWindow('opening',cv2.WINDOW_NORMAL)
	cv2.imshow("filtered",filtered)
	cv2.imshow("closing",closing)
	cv2.imshow("opening",opening)

	_, contours0, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	return contours0,closing



def get_ave_area(contours0):
	temp = 0
	for c in contours0:
		[x, y, w, h] = cv2.boundingRect(c)
		current_area = w*h
		# print("current area: ",current_area)
		temp = temp + current_area
	ave_area = temp/len(contours0)
	# print("average area: ",ave_area)
	return ave_area
def remove_noise(contours0):
	contours = []
	ave_area = get_ave_area(contours0)
	# print("average area: ",ave_area)
	threshold_area = (0.010*(ave_area))
	# print("threshold area: ",threshold_area)
	# contours = [c for cv2.boundingRect(c) in contours0 if (((c[0]+c[2])*(c[1]+c[3])) >= ave_area)]
	for c in contours0:
		[x,y,w,h] = cv2.boundingRect(c)
		if ((w*h) >= threshold_area ):
			contours.append(c)
	return contours






def remove_coinsides(contours):
	indices = []
	for c in contours:
		[x, y, w, h] = cv2.boundingRect(c)
		for index,cn in enumerate(contours):
			
			[i,j,k,l] = cv2.boundingRect(cn)
			
			if ((i < (x+w) and (i > x)) and ((i+k) < (x+w) and (i+k) > x )):
				# print("inside")
				if((((j+l) < (y+h)) and ((j+l) > y )) and ((j < (y+h)) and (j > y))):
			
					indices.append(index)
					
	contours2 = [c for i,c in enumerate(contours) if i not in indices]

	return contours2







# SORT contours2 according t the x values in increasing order(largest pixel value at the last)
# Determine the maximum probable case that it is a space between words
# def sort_prevline():

def get_differences(contours2,differences):	
	lastIndex = len(contours2) - 1
	#Checking the biggest space
	for index,c1 in enumerate(contours2):
		# print ('SIZE: %d' %(len(contours2)))
		# if(index < (len(contours2))-1):
		if index == lastIndex: continue
		[x, y, w, h] = contours2[index]
		[i, j, k, l] = contours2[index+1]
		dif = abs((i-(x+w))); #upper-left point of the next contour minus the upper-right point of the current contour
		print("dif: ", dif)
		differences.append(dif)


	print("length of differences:",len(differences))
	print("differences:",differences)
	return differences
def get_localcut(differences,localCut,sliding):
	for idx, d in enumerate(differences):
			if len(sliding) < 3:  #if one word with
				sliding.append(d)
				if len(differences) < 3: #max is 3 words
					# print("i:",idx)
					localCut.append(idx)
			else:	#if two words
				# print("else  i:",idx)
				if (math.ceil(sum(sliding)/len(sliding))) < differences[idx]:
					localCut.append(idx)
				else: 
					del sliding[0]
					sliding.append(d)


	print("localcut values are:")
	print(localCut)
	return sliding,localCut
def get_rare(globalCut,localCut):
	rare = [val for i,val in enumerate(globalCut) if val not in localCut]
	print("RARE:", rare)
	return rare
def get_globalcut(differences,globalCut,localCut):
	# if(len(localCut)!=0):

	print("length of differences: ",len(differences))
	max_val = max(differences)

	for index, d in enumerate(differences):
		cur = (d / max_val) * 100

		# print("cur:",cur)
		if (cur > 30):
			globalCut.append(index)

	print("global cut: ")
	print(globalCut)

	intersection = [overlap for i, overlap in enumerate(globalCut) if overlap in localCut]

	globalCut = [cut for i, cut in enumerate(globalCut) if i not in intersection ]


	print("Intersection values", intersection)
	print("new global cut: ", globalCut)
	return intersection,globalCut
# def get_words(true_labels,image,contours2,intersection):
def get_words(contours2,intersection):
	x1,y1 = None,None
	i=0
	while i < len(contours2)-1:
		c = contours2[i]
		d = contours2[i+1]
		if(x1 is None):
			x1 = c[0]
		if(y1 is None):
			y1 = c[1]
	#adjust height of the y1 to align with the height of the taller contour
		if(y1 > d[1]): 
			y1 = d[1] #y1 is the the y-coordinate stored in words array

			
		print("i: ",i)
	#if contour[i+1][y+h] > contour[i][y+h]

		#ang index ara sa final cuts
		if(i in intersection): #pakicheck if the index overflows
			print("Pumasok si i:")
			# print(i)
			x2 = c[0] + c[2] #x+w
			y2 = c[1] + c[3] #y+h
			box = [x1,y1,x2,y2]
			words.append(box)
			x1 = d[0]
			y1 = d[1]
			x2 = None
			y2 = None
		i=i+1
	if(i == len(contours2)-1): #if lagpas na sya sa last 2 element, then consider it as out of bounds
		x2 = d[0] + d[2] #x+w
		y2 = d[1] + d[3] #y+h
		box = [x1,y1,x2,y2]
		words.append(box)
	return words
		# words.append(box)
	# return words
def find_trueCuts(differences,rare,intersection):
	# 	start =[]
	# 	end = []
	end_max = len(differences) - 1
# 	# start
	for d in rare:
		# compare if the average of start and end is less than the d/midpoint, d is a space
		print("d: ",d)
		if(d<3):
				if(d == 0) or (d == 1):
				# if(d==0):
					avg_start = 0
				else:
					start=range(0,d)
					# if d < 3:
		else:
			start=range(d-3,(d-1)+1)
		if(d > 1):
		# if(d != 0):
			avg_start = math.ceil(sum(differences[start[0]:start[1] + 1])/len(start))
			
			
		print("avg_start: ",avg_start)


		# print("d is:",d)
		end_max = len(differences)-1

		if(d>= (len(differences)-3)):
			if d == end_max or d == end_max - 1:
				avg_end=differences[end_max-1]
			else:  
				end=range(d+1,end_max+1)
		else:
			end=range(d+1,(d+3)+1)

		
		# if(d!=len(differences)-1):
		if(d < end_max-1):
			# print("d: ",d)
			print("end is:",end)
			print("length of diff:",len(differences))
			print("diff[end[0]]:", end[0])
			print("diff[end[1]]:", end[1])
			print("length of end: ", len(end))

			avg_end = math.ceil(sum(differences[end[0]:end[1] + 1])/len(end))
		print("avg_end: ", avg_end)

		if((avg_start <= differences[d]) and (avg_end <= differences[d])):
			intersection.append(d)
	return intersection

def make_pad(image):
	h,w = image.shape
	print(image.shape)
	new_img = np.zeros((h,w+5),np.uint8)
	print(new_img.shape)
	new_img[0:h,5:w+5] = image
	return new_img
def draw_words(words,image):
	print("Drawing wordsss")
	for box in words:
		print("box[0]: ",box[0])
		print("box[1]: ",box[1])               
		print("box[2]: ",box[2])
		print("box[3]: ",box[3])

		# [x, y, w, h] = cv2.boundingRect(c)
		cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (90, 60 , 80), 3)
		cv2.imshow("WORDS",image)
def new_line(image,rect,line_begin_idx,ix,contours2,differences,sliding,localCut,globalCut,words):
	
	# sort the previous line by their x
	rect[line_begin_idx:ix] = sorted(rect[line_begin_idx:ix], key=lambda b: b[0])
	contours2 = rect[line_begin_idx:ix] #hanggang saan sa rect ang line na yan
	print("Inside new_line checking length of contours: ",len(contours2))
	differences = get_differences(contours2,differences)
	if(len(differences) == 0):
		print("Premature end because there is only one contour for this line :(")
		print("Line number: ", line_begin_idx)
		sys.exit(0)
	sliding,localCut = get_localcut(differences,localCut,sliding)
	intersection,globalCut = get_globalcut(differences,globalCut,localCut)
	
	#If space threshold found, create words array and and get the points of the 
	#DRAWING
	rare = get_rare(globalCut,localCut)
	if rare:
		intersection = find_trueCuts(differences,rare,intersection)


	words = get_words(contours2,intersection)
	draw_words(words,image)
	return line_begin_idx,words
def lastline(image,line_begin_idx,rect,contours2,differences,sliding,localCut,globalCut,words):
	print("inside last line!")
	rect[line_begin_idx:] = sorted(rect[line_begin_idx:], key=lambda b: b[0])
	contours2 = rect[line_begin_idx:]
	# print("line_begin_idx:", line_begin_idx)
	differences = get_differences(contours2,differences)
	sliding,localCut = get_localcut(differences,localCut,sliding)
	intersection,globalCut = get_globalcut(differences,globalCut,localCut)
	
	#If space threshold found, create words array and and get the points of the 
	#DRAWING
	rare = get_rare(globalCut,localCut)
	if rare:
		intersection = find_trueCuts(differences,rare,intersection)

	# words = get_words(true_labels,image,contours2,intersection)
	words = get_words(contours2,intersection)
	draw_words(words,image)
	return

def reset_values(words,differences,sliding,localCut,globalCut,contours2):
	words = []
	differences=[]
	sliding=[]
	localCut=[]
	globalCut=[]
	intersection=[]
	contours2=[]

	return words,differences,sliding,localCut,globalCut,intersection,contours2

def segmentation(image,contours2,differences,sliding,localCut,globalCut,words):
	rect = [cv2.boundingRect(c) for c in contours2]
	label_count = 0
	allWords = []
	# sort all rect by their y
	rect.sort(key=lambda b: b[1])
	# initially the line bottom is set to be the bottom of the first rect
	line_bottom = rect[0][1]+rect[0][3]-1 #y+h amo na ang bottom mo
	line_begin_idx = 0
	p=0
	print("RECT SIZE: ",len(rect))
	# print("length of rect:",len(rect))
	for ix in range(len(rect)):
		# when a new box's top is below current line's bottom
		# it's a new line
		# print("line_bottom:", line_bottom)
		if rect[ix][1] > line_bottom: #if y is greater than bottom meaning next line
			# print("-------------LABELS: ",true_labels[label_count])
			line_begin_idx,words = new_line(image,rect,line_begin_idx,ix,contours2,differences,sliding,localCut,globalCut,words)
		# 	label_count = label_count + 1
		# 	allWords.append(words)
		# 	allWords.append(["\n"])

			# print("P: ",p)
		    
		words,differences,sliding,localCut,globalCut,intersection,contours2 = reset_values(words,differences,sliding,localCut,globalCut,contours2)
		# regardless if it's a new line or not
		# always update the line bottom
		# line_bottom = max(rect[i][1]+rect[i][3]-1, line_bottom)
		line_bottom = (rect[ix][1]+rect[ix][3]-1)
	# label_length = len(true_labels)
	words = lastline(image,line_begin_idx,rect,contours2,differences,sliding,localCut,globalCut,words)




image = cv2.imread("set2.jpg")


sliding = []
contours2 =[]
localCut = []
differences=[]
globalCut = [] #stores indices of differences that has 50% chance of being a space
words = []
contours=[]
current_area = 0
indices=[] #stores values of the contours that are co-inside the contour
X_train = []
Y_train = []
closing,contours0 = otsu_preprocess(image)
# contours = remove_noise(contours0)
contours = contours0
# ave_area =get_ave_area(contours)
contours2 = remove_coinsides(contours)
contours3=[]
small_contours3 = []

show_contours(contours2,image)


segmentation(image,contours2,differences,sliding,localCut,globalCut,words)


cv2.waitKey(0)
