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
import cv2
import numpy as np 	
import math
import matplotlib.pyplot as pyplot
from sklearn import datasets
from sklearn import svm


image = cv2.imread('G_1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
 
# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

rows,cols = thresh.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2),15,1)
dst = cv2.warpAffine(thresh,M,(cols,rows))

backorig = cv2.threshold(dst, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cv2.imwrite("G_new.jpg",backorig)

h,w = dst.shape[:2]
ar = w / h
nw = 500
nh = int(nw / ar)
dst = cv2.resize(dst,(nw,nh))

h,w = thresh.shape[:2]
ar = w / h
nw = 500
nh = int(nw / ar)
thresh = cv2.resize(thresh,(nw,nh))

h,w = backorig.shape[:2]
ar = w / h
nw = 500
nh = int(nw / ar)
backorig = cv2.resize(backorig,(nw,nh))


# cv2.namedWindow('gray',cv2.WINDOW_NORMAL)
# cv2.namedWindow('thresh',cv2.WINDOW_NORMAL)
cv2.imshow("thresh",thresh)
# cv2.imshow("GRAY",gray)
cv2.imshow("dst",dst)
cv2.imshow("tbackorig",backorig)
cv2.waitKey(0)