import cv2
import sys
import imutils
from imutils.perspective import four_point_transform
import numpy as np
import os
def crop_margins(image):
	h,w,_ = image.shape
	print("image shape: ",image.shape)
	margin = w*h
	y = 0
	x = 0
	# new_y = y + 100
	# new_h = h - 200
	# new_x = x + 100
	# new_w = w - 200

	add_y = int(h*0.07)
	add_x = int(w*0.07)
	# print("add_y: ",add_y)
	# print("add_x: ",add_x)

	new_y = y + add_y
	new_h = h - 2*add_y
	new_x = x + add_x
	new_w = w - 2*add_x





	roi = image[new_y:new_y+new_h,new_x:new_x+new_w]


	return roi
def document_detect(username):
	folder_path = "/home/lincy/workflow_structure/USERS/" + username + "/training_raw/"
	numfiles = 0
	for root, dirs, files in os.walk(folder_path):
		for image_path in files:  
			print("Currently processing: ",image_path)
			img = cv2.imread(folder_path+image_path)


			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			print(img.shape)

			_,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

			_,contours,_ = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
			for c in contours:
				print("contour area:",cv2.contourArea(c))

				if(cv2.contourArea(c) < 150*100):
					continue
				(x1,y1,w1,h1) = cv2.boundingRect(c)
				x2 = x1 - 10
				y2 = y1 - 10
				w2 = w1 + 20
				h2 = h1 + 20      
				cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(0,255,0),8)


			temp = cv2.resize(img,None,fx=0.1,fy=0.1)
			# cv2.imshow("x",temp)
			# cv2.waitKey(0)

			img = img[y2:y2+h2, x2:x2+w2]                
			temp = cv2.resize(img,None,fx=0.1,fy=0.1)
			print(img.shape)
			# cv2.imshow("cropped", temp)
			# cv2.waitKey(0)
			# sys.exit(0)
			img = cv2.bilateralFilter(img, 9, 75, 75)

			# image = resize_img(image)
			# cv2.imshow("ORIG",image)
			orig = img.copy()
			gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			# gray = resize_img(gray)
			# cv2.imshow("GRAY",gray)
			blurred = cv2.GaussianBlur(gray, (5, 5), 0)

			th, bw = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


			temp2 = cv2.resize(bw,None,fx=0.1,fy=0.1)

			# cv2.imshow("Threshold",temp2)


			# edged = auto_canny(gray,0.33)
			# edged = cv2.Canny(gray, 75, 200)
			edged = cv2.Canny(img, th/2, th)
			temp3 = cv2.resize(edged,None,fx=0.3,fy=0.3)

			# cv2.imshow("EDGED",temp3)

			kernel = np.ones((15,15), np.uint8)
			dilation = cv2.dilate(edged,kernel,iterations = 2)
			temp8 = cv2.resize(dilation,None,fx=0.3,fy=0.3)
			# cv2.imshow("dilate",temp8)
			# cv2.waitKey(0)
			# sys.exit(0)

			edged = dilation

			# find contours in the edge map, then initialize
			# the contour that corresponds to the document
			cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
			cnts = cnts[0] if imutils.is_cv2() else cnts[1]
			docCnt = None
			 
			# ensure that at least one contour was found
			if len(cnts) > 0:
				# sort the contours according to their size in
				# descending order
				cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
			 
				# loop over the sorted contours
				for c in cnts:
					# approximate the contour
					peri = cv2.arcLength(c, True)
					approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			 
					# if our approximated contour has four points,
					# then we can assume we have found the paper
					if len(approx) == 4:
						docCnt = approx
						break



			# cv2.drawContours(img,[docCnt],0,(0,0,255),3)
			# temp4 = cv2.resize(img,None,fx=0.3,fy=0.3)
			# cv2.imshow("Contours",temp4)
			# cv2.waitKey(0)
			# sys.exit(0)
			# apply a four point perspective transform to both the
			# original image and grayscale image to obtain a top-down
			# birds eye view of the paper
			paper = four_point_transform(img, docCnt.reshape(4, 2))

			paper = crop_margins(paper)
			dest_path = "/home/lincy/workflow_structure/USERS/" + username + "/training_clean/"
			cv2.imwrite(dest_path + image_path,paper)
			# cv2.imshow("Paper",paper)
			print("Done processing IMAGE: ",image_path)
			# cv2.waitKey(0)


	return