import cv2
import numpy as np

img = cv2.imread('bookpage.jpg')
#	first attempt //diretso threshold
#retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)

#	second attempt //grayscale the image before thresholding
# grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# retval, threshold = cv2.threshold(grayscaled, 10, 255, cv2.THRESH_BINARY)

#	third attempt //using adaptive thresholding which varies the threshold 
grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
threshold = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

#	fourth attempt //using OTSU's threshold
# grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# retval2,threshold = cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('original',img)
# cv2.imshow('Otsu threshold',threshold)


cv2.imshow('original',img)
cv2.imshow('threshold',threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()