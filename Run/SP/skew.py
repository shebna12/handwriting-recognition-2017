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

blur = cv2.GaussianBlur(gray, (3, 3), 0)

filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 77, 3)


# Some morphology to clean up image,originally 7,7
kernel = np.ones((3,3), np.uint8)

opening = cv2.morphologyEx(filtered,cv2.MORPH_OPEN, kernel, iterations = 2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations =2)
kernel2 = np.ones((17,17),np.uint8)
closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel2, iterations =1)
cv2.namedWindow('grayy',cv2.WINDOW_NORMAL)
cv2.namedWindow('blurr',cv2.WINDOW_NORMAL)
cv2.namedWindow('closing',cv2.WINDOW_NORMAL)
cv2.namedWindow('res',cv2.WINDOW_NORMAL)
cv2.namedWindow('filtered',cv2.WINDOW_NORMAL)
cv2.namedWindow('opening',cv2.WINDOW_NORMAL)
cv2.imshow("filtered",filtered)
cv2.imshow("GRAY",gray)
cv2.imshow("BLUR",blur)
cv2.imshow("closing",closing)
cv2.imshow("opening",opening)
cv2.waitKey(0)