"""
	W A R N I N G :
	*** DO NOT CHANGE ANYTHING IN THIS FILE
		UNLESS YOU ARE ADDING A NEW FUNCTION
		for local experimentation:
			- duplicate the function in your python file 

	All global functions should be placed here
	in order to reduce redundancy of its presence
	for every file.
"""



import scipy
import cv2
import glob
import numpy as np
import math
from sklearn import model_selection,svm
import sys
import statistics


def closer_to_prev(contours3,index_c,x,y,w,h):
	new_x = contours3[index_c-1][0]
	new_y = contours3[index_c-1][1]
	new_w = contours3[index_c-1][2]
	new_h = contours3[index_c-1][3]
	# if(x < new_x):
	if(new_x < x):
		print("j case")
		new_w = new_w
		new_x = new_x
		new_h =  new_h +(new_y - y)
		new_y = y
		remove_index = index_c-1
	else:
		new_w = new_w + ((new_x+new_w)-(x+w))
		new_x = x
		new_h = new_h +(new_y - y)
		new_y = y
		remove_index = index_c-1

	return new_x,new_y,new_w,new_h,remove_index

def closer_to_next(contours3,index_c,x,y,w,h):
	new_x = contours3[index_c+1][0]
	new_y = contours3[index_c+1][1]
	new_w = contours3[index_c+1][2]
	new_h = contours3[index_c+1][3]
	
	# new_w = new_w + ((w + x) - (new_x + new_w))
	# new_x = new_x
	new_w = new_w + w
	new_x = x
	# new_h = new_h +(new_y - y)
	new_h = new_h + h + (new_y-(y+h))
	new_y = y
	remove_index = index_c+1

	return new_x,new_y,new_w,new_h,remove_index



def resizeTo28(img):
	r = 28.0 / img.shape[1]
	dim = (28, int(img.shape[0] * r))
	res = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	# cv2.imshow("To28",res)
	# cv2.waitKey(1)
	return res
def resizeTo20(img):
	# r = 20.0 / img.shape[1]
	# dim = (20, int(img.shape[0] * r))
	res = cv2.resize(img, (20,20), interpolation = cv2.INTER_AREA)
	# cv2.imshow("To20",res)
	# cv2.waitKey(1)
	return res
def get_ROI(image,contours):
	# for c in contours:
	x,y,w,h = cv2.boundingRect(contours[0])
	# print(x,y,w,h)
	patch = image[y:(y+h),x:(x+w)]
	

	return patch
		

def center_image(patch,contours):

	new_img = np.zeros((20,20),np.uint8)
	# print(new_img.shape)
	# print(patch.shape)
	# x,y,w,h = cv2.boundingRect(contours[0])
	h,w = patch.shape[0:2]
	mX = math.floor((20-w)/2)
	mY = math.floor((20-h)/2)
	# print("mX: {}  mY:{}  " .format(mX,mY))
	x1 = mX
	x2 = x1+w
	y1 = mY
	y2 = y1+h
	# print("y1: {}  y2: {}  x1: {}  x2: {}"  .format(mY,y2,x1,x2))
	new_img[y1:y2,x1:x2] = patch

	# print("--------------")
	# cv2.imshow("CENTER",new_img)
	# cv2.waitKey(0)
	return new_img

def transform20x20(image):
	# _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	original = image
	prepimage,contours = otsu_preprocess(image)
	contours1 = remove_noise(contours)
	contours = remove_coinsides_letter(contours1)



	# if(len(contours) == 1):
		# Center the position of the image to a new 28x28 image
		
		

	if(len(contours) >1 ):
		# contours = sort_LR(contours)
		contours = remove_coinsides_letter(contours)
		# show_contours(contours,image)
		print("More than 1 contour found. Hopefully this was fixed.")		
	

	patch = get_ROI(prepimage,contours)
	centered = center_image(patch,contours)

	
	return centered
def get_max_area_contour(contours):
	areas = []
	for i,c in enumerate(contours):
		[x, y, w, h] = cv2.boundingRect(c)
		current_area = w*h
		areas.append(current_area)
	contour_roi_index = areas.index(max(areas))
	contour_roi = contours[contour_roi_index]
	# print("average area: ",ave_area)
	return contour_roi
def get_ave_area(contours0):
	temp = 0
	for c in contours0:
		[x, y, w, h] = cv2.boundingRect(c)
		current_area = w*h
		# print("current area: ",current_area)
		temp = temp + current_area
	ave_area = temp/len(contours0)
	print("average area: ",ave_area)
	return ave_area
# def get_midpoint(contours):
# 	contours_length = len(contours)
# 	if(contours_length%2 == 0): #Even
# 		midpoint_index = contours_length/2
# 	else: #Odd
# 		mid_initial_index = contours_length/2
# # 		midpoint =  (contours[mid_initial_index] + contours[mid_initial_index+1])/2
	
# 	return contours[midpoint_index]
def get_median_area(contours0):
	# Sort the area ascending
	# Get the length of the sorted contours
	# find the midpoint
	# get the midpoint's area
	areas = []
	contours = [cv2.boundingRect(c) for c in contours0]
	# for c in contours0:
	# 	[x,y,w,h] = cv2.boundingRect(c)
	# 	sorted_contours = [x,y,w,h]
	sorted_contours = sort_LR(contours)
	for c in sorted_contours:
		[x,y,w,h] =c
		area = w*h
		areas.append(area)
	midpoint = statistics.median(areas)
	print(midpoint)
	return midpoint

def show_contours(contours,image):
	
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)	
		print(x,y,w,h)
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 40), 1)
		# cv2.imshow("SHOWCONTOURS",image)
		# cv2.waitKey(1)
		

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
	# print("Length of unfiltered contours: ",len(contours0))
	# print("Length of filtered contours: ",len(contours))
	return contours

def otsu_preprocess(image):
	gray = image
	print(len(image.shape))
	if(len(image.shape) > 2):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	prepimage = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	# cv2.imshow("IN OTSU PREPREOCESS IMAGE",prepimage)
	# cv2.waitKey(1)
	_, contours, hierarchy = cv2.findContours(prepimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	return prepimage,contours

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

def overlapX(boxA,boxB):
	Ax1,Ay1,Ax2,Ay2 = boxA
	Bx1,By1,Bx2,By2 = boxB

	end = Ax2
	start = Bx1
	if Bx1 < Ax1: # if B is before A
		end = Bx2
		start = Ax1

	return start <= end

def overlapY(boxA,boxB):
	Ax1,Ay1,Ax2,Ay2 = boxA
	Bx1,By1,Bx2,By2 = boxB

	end = Ay2
	start = By1
	if By1 < Ay1: # if B is before A
		end = By2
		start = Ay1

	return start <= end

def remove_coinsides_letter(contours):
	import itertools
	remove = []
	indices = range(len(contours))
	for index1,index2 in itertools.combinations(indices,2):
		contour1 = contours[index1]
		contour2 = contours[index2]

		c1_x1,c1_y1,w1,h1 = cv2.boundingRect(contour1)
		c2_x1,c2_y1,w2,h2 = cv2.boundingRect(contour2)
		c1_x2,c1_y2 = c1_x1 + w1, c1_y1 + h1 
		c2_x2,c2_y2 = c2_x1 + w2, c2_y1 + h2

		boxA = (c1_x1,c1_y1,c1_x2,c1_y2)
		boxB = (c2_x1,c2_y1,c2_x2,c2_y2)
		if overlapX(boxA,boxB) or overlapY(boxA,boxB):
			areaA = w1 * h1 
			areaB = w2 * h2 
			smaller = index1 # first box is smaller (assume)
			if areaB < areaA: # second box is smaller
				smaller = index2
			remove.append(smaller)

	return [contour for i,contour in enumerate(contours) if i not in remove]


def preprocess(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (3, 3), 0)
	filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 77, 3)

	# Some morphology to clean up image
	kernel = np.ones((3,3), np.uint8)
	opening = cv2.morphologyEx(filtered,cv2.MORPH_OPEN, kernel, iterations = 1)
	kernel2 = np.ones((3,3),np.uint8)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2, iterations =1)
	# cv2.namedWindow('closing',cv2.WINDOW_NORMAL)
	# cv2.namedWindow('res',cv2.WINDOW_NORMAL)
	# cv2.namedWindow('filtered',cv2.WINDOW_NORMAL)
	# cv2.namedWindow('opening',cv2.WINDOW_NORMAL)
	# cv2.imshow("filtered",filtered)
	# cv2.imshow("closing",closing)
	# cv2.imshow("opening",opening)

	_, contours0, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	return contours0,closing


def sort_LR(contours):
	contours.sort(key=lambda b: b[0])
	return contours
def resize_fixed(img):
	h,w = img.shape[:2]
	ar = w / h 
	nw = 300
	nh = int(nw / ar)        
	nimage = cv2.resize(img,(nw,nh))
	
	return nimage

def resize_img(thresh):
	h,w = thresh.shape[:2]
	ar = w / h 
	nw = 800
	nh = int(nw / ar)        
	nimage = cv2.resize(thresh,(nw,nh))
	
	return nimage
	
def get_area_real_contours(contours):
	areas = []
	for con1 in contours:
		x, y, w, h = cv2.boundingRect(con1)
		current_area = w*h
		# print("current area: ",current_area)
		areas.append(current_area)
	print("all areas of the contour: ",areas)
	return areas

def get_area_of_contours(contours):
	areas = []
	for con1 in contours:
		x, y, w, h = con1[0],con1[1],con1[2],con1[3]
		current_area = w*h
		# print("current area: ",current_area)
		areas.append(current_area)
	print("all areas of the contour: ",areas)
	return areas


def get_bounding_area(contours):
	temp = 0
	for c in contours:
		x, y, w, h = c[0],c[1],c[2],c[3] 
		current_area = w*h
		# print("current area: ",current_area)
		temp = temp + current_area
	ave_area = temp/len(contours)
	print("average area: ",ave_area)
	return ave_area

def homomorphic_filtering(img):
	# Number of rows and columns
	# img = cv2.resize(img,None,fx=0.3,fy=0.3)
 
	rows = img.shape[0]
	cols = img.shape[1]
 
	# Remove some columns from the beginning and end
	img = img[:, 59:cols-20]
 
	# Number of rows and columns
	rows = img.shape[0]
	cols = img.shape[1]
 
	# Convert image to 0 to 1, then do log(1 + I)
	imgLog = np.log1p(np.array(img, dtype="float") / 255)
 
	# Create Gaussian mask of sigma = 10
	M = 2*rows + 1
	N = 2*cols + 1
	sigma = 10
	(X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
	centerX = np.ceil(N/2)
	centerY = np.ceil(M/2)
	gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2
 
	# Low pass and high pass filters
	Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
	Hhigh = 1 - Hlow
 
	# Move origin of filters so that it's at the top left corner to
	# match with the input image
	HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
	HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())
 
	# Filter the image and crop
	If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
	Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
	Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))
 
	# Set scaling factors and add
	gamma1 = 0.3
	gamma2 = 1.5
	Iout = gamma1*Ioutlow[0:rows,0:cols] + gamma2*Iouthigh[0:rows,0:cols]
 
	# Anti-log then rescale to [0,1]
	Ihmf = np.expm1(Iout)
	Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
	Ihmf2 = np.array(255*Ihmf, dtype="uint8")
 
	# Threshold the image - Anything below intensity 65 gets set to white
	Ithresh = Ihmf2 < 65
	Ithresh = 255*Ithresh.astype("uint8")

	return Ithresh




def i_and_j_preprocessor(num,name,image,contours3,global_ave_area,ix,line_begin_idx):
	i=0
	# print(len(contours3))    
	# print("pre-contours3: ",contours3)   
	# sys.exit(0)
	# ave_area_indie = get_ave_area_indie(contours3)
	ave_area_indie = global_ave_area
	# print("true labels length: ",len(true_labels))
	# print("precontours3 length: ",len(contours3))
	area_of_contours = get_area_of_contours(contours3)
	
	# Meaning may letter i or j. Group the dot and body.
	for indie_area in area_of_contours:
		if(indie_area < 0.5*ave_area_indie):
	# while(len(contours3) > len(true_labels)):
			print("Went to indie area")
			prev_flag = False
			next_flag = False
			for index_c,cont in enumerate(contours3):

				x,y,w,h =  cont[0],cont[1],cont[2],cont[3]
				if(w*h < (0.5*ave_area_indie)): # Check if the dot is close to the previous or next contour
					print("my average: ",w*h)
					print("That contour is: ",cont)
					print("Index c: ",index_c)
					print("Cont3 len: ",len(contours3))
					print("Contours3: ",contours3)
					if(index_c == len(contours3)-1): #If wala na next 
						new_x,new_y,new_w,new_h,remove_index = closer_to_prev(contours3,index_c,x,y,w,h)
						prev_flag = True
						print("Fuck this")
						break
					prev_diff = abs(contours3[index_c-1][0] - x) 
					prev_diff = abs(contours3[index_c-1][0] - x) 
					try:
						next_diff = abs(contours3[index_c+1][0] - x)
					except IndexError as ie:
						new_x,new_y,new_w,new_h,remove_index = closer_to_prev(contours3,index_c,x,y,w,h)
						prev_flag = True
						break
					print(prev_diff)
					print(next_diff)
					print("ans:",prev_diff - next_diff)

					if(prev_diff < next_diff): # mas malapit si prev diff kay dot
						print("Closer to prev contour")
						new_x,new_y,new_w,new_h,remove_index = closer_to_prev(contours3,index_c,x,y,w,h)
						prev_flag = True
						break
					else:
						print("closer to next contour")
						new_x,new_y,new_w,new_h,remove_index = closer_to_next(contours3,index_c,x,y,w,h)
						next_flag = True
						break
			# If potential body contour we had in first loop was wrong. correct it.
			if (new_x < 0 or new_y < 0 or new_w < 1 or new_h < 1):
				if(next_flag  == True):
					print("<next_flag>Fixing: ",new_x,new_y,new_w,new_h)
					new_x,new_y,new_w,new_h,remove_index = closer_to_prev(contours3,index_c,x,y,w,h)
				elif (prev_flag == True):
					print("<prev_flag>Fixing: ",new_x,new_y,new_w,new_h)
					try:
						next_diff = abs(contours3[index_c+1][0] - x)
					except IndexError as ie:
						new_x,new_y,new_w,new_h,remove_index = closer_to_prev(contours3,index_c,x,y,w,h)
						prev_flag = True
						break
				else:
					continue
			if(remove_index < len(contours3)):
				print("remove_index: ",remove_index)
				print("contours3 length: ",len(contours3))
				contours3[remove_index] = new_x,new_y,new_w,new_h

				# except UnboundLocalError as nonMeanError:
					# continue
				del contours3[index_c] 

	print("modified contours3: ",contours3)
	for c in contours3: 
		if(i < len(contours3)):
			try:
				x,y,w,h = c[0],c[1],c[2],c[3]
				# print(x,y,w,h)
				# cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 2)
				letter = image[y:y+h,x:x+w]
				# letter = cv2.threshold(letter, 0, 255,
				# cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
				# path = "/home/shebna/USERS/"+username+"/training_cutter/"
				cv2.imwrite(name + "_m_"+str(num)  + ".jpg", letter)
				print("Letter coords: ",x,y,w,h)
				# cv2.imshow("letter",letter)
				# cv2.waitKey(1)
				print("i: ",i)
				print("num in i_j: ")
				num = num +1
			except cv2.error as e:
				continue

		i = i + 1
	line_begin_idx = ix #update value of line_begin_idx

	return line_begin_idx,num