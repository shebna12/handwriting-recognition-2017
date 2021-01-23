import cv2
import numpy as np 	
import math
import matplotlib.pyplot as pyplot
from sklearn import datasets
from sklearn import svm
from utils import remove_coinsides, otsu_preprocess, resize_img, sort_LR, remove_noise
import sys

import os

def get_ave_area_indie(contours):
	temp = 0
	for c in contours:
		x, y, w, h = c[0],c[1],c[2],c[3]
		current_area = w*h
		# print("current area: ",current_area)
		temp = temp + current_area
	ave_area = temp/len(contours)
	print("zaverage area: ",ave_area)
	return ave_area

image = cv2.imread("intern.png")
contours3=[]
closing,contours0 = otsu_preprocess(image)
contours = remove_noise(contours0)
contours = contours
contours2 = remove_coinsides(contours)
# contours3 = contours2
for c in contours2:
	x,y,w,h = cv2.boundingRect(c)
	contours3.append([x,y,w,h])
print("contours3: ",contours3)
contours3 = sort_LR(contours3)
print("sorted contours3: ",contours3)
ave_area_indie = get_ave_area_indie(contours3)

for index_c,cont in enumerate(contours3):
			x,y,w,h =  cont[0],cont[1],cont[2],cont[3]
			if(w*h < (0.5*ave_area_indie)): # Check if the dot is close to the previous or next contour
				print("That contour is: ",cont)
				prev_diff = abs(contours3[index_c-1][0] - x) 
				next_diff = abs(contours3[index_c+1][0] - x)
				print(prev_diff)
				print(next_diff)
				print("ans:",prev_diff - next_diff)
				if(prev_diff < next_diff): # mas malapit si next_diff kay dot
					print("Closer to next contour")
					contours_new = contours3
					new_x = contours3[index_c-1][0]
					new_y = contours3[index_c-1][1]
					new_w = contours3[index_c-1][2]
					new_h = contours3[index_c-1][3]
					# if(x < new_x):
					if(new_x < x):
						print("here")
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
					break
				else:
					print("closer to prev contour")
					new_x = contours3[index_c+1][0]
					new_y = contours3[index_c+1][1]
					new_w = contours3[index_c+1][2]
					new_h = contours3[index_c+1][3]
					# if(new_x < x):
					# 	print("here")
					# 	new_w = new_w
					# 	new_x = new_x
					# 	new_h = new_y + (new_y-y)
					# 	new_y = y
					# 	remove_index = index_c+1
					# else:
					print("oops")
					new_w = new_w + ((w + x) - (new_x + new_w))
					new_x = new_x
					new_h = new_h +(new_y - y)
					new_y = y
					remove_index = index_c+1
					break
contours3[remove_index] = new_x,new_y,new_w,new_h
del contours3[index_c] 
print("modified contours3: ",contours3)




for c in contours3:
	x, y, w, h = c[0],c[1],c[2],c[3]
	print(x,y,w,h)

	cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 2)
	
deleteafter_img = resize_img(image)
cv2.imshow("No coinsides",deleteafter_img)
cv2.waitKey(0)
import sys
sys.exit(0)