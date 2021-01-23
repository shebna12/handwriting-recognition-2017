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


img = cv2.imread('n.jpg',0)
rows,cols = img.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2),15,1)
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow("REsult",dst)
cv2.waitKey(0)