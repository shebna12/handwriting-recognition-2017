import numpy as np 
import cv2
import imutils

image = cv2.imread("text.png")




#-----Converting image to LAB Color model----------------------------------- 
# lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# cv2.imshow("lab",lab)

# #-----Splitting the LAB image to different channels-------------------------
# l, a, b = cv2.split(lab)
# cv2.imshow('l_channel', l)
# cv2.imshow('a_channel', a)
# cv2.imshow('b_channel', b)

# #-----Applying CLAHE to L-channel-------------------------------------------
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# cl = clahe.apply(l)
# cv2.imshow('CLAHE output', cl)

# #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
# limg = cv2.merge((cl,a,b))
# cv2.imshow('limg', limg)

# #-----Converting image from LAB Color model to RGB model--------------------
# final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
# cv2.imshow('final', final)

# #_____END_____#



#image = imutils.resize(image, height = 500)
# gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# cv2.imshow("hola", gray)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

_,cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
digitCnts = []

for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)

		if w >= 15 and (h >= 30 and h <= 40):
				digitCnts.append(c)
				cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

print ('Number of contours are: %d -> ' %len(cnts))

cv2.imwrite("product.jpg", image)
cv2.imshow("original", image)
cv2.waitKey(0)
