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
def test_document_detect(username):
	folder_path = "/home/lincy/workflow_structure/USERS/" + username + "/testing/raw/"
	
	numfiles = 0
	for root, dirs, files in os.walk(folder_path):
		for image_path in files:  
			print("Currently processing: ",image_path)
			img = cv2.imread(folder_path+image_path)


			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			print(img.shape)

			_,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

			_,contours,_ = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
			

			cont_areas = []
			for c in contours:
				h,w = thresh.shape
				cont_areas.append(cv2.contourArea(c))


			for c in contours:
				# print("contour area:",cv2.contourArea(c))
				h,w = thresh.shape
				if(cv2.contourArea(c) < max(cont_areas)):
					continue
				(x1,y1,w1,h1) = cv2.boundingRect(c)
				print("BOUNDING RECT:",x1,y1,w1,h1)
				if(x1 > 10):
					x2 = x1 - 10
					w2 = w1 + 20
				if(y1 > 10):
					y2 = y1 - 10
					h2 = h1 + 20 
				# x2,y2,w2,h2 = x1,y1,w1,h1     
				cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(0,255,0),8)


			temp = cv2.resize(img,None,fx=0.1,fy=0.1)

			img = img[y2:y2+h2, x2:x2+w2]                
			temp = cv2.resize(img,None,fx=0.1,fy=0.1)
			print(img.shape)

			img = cv2.bilateralFilter(img, 9, 75, 75)


			orig = img.copy()
			gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

			blurred = cv2.GaussianBlur(gray, (5, 5), 0)

			th, bw = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


			temp2 = cv2.resize(bw,None,fx=0.1,fy=0.1)


			edged = cv2.Canny(img, th/2, th)
			temp3 = cv2.resize(edged,None,fx=0.3,fy=0.3)


			kernel = np.ones((15,15), np.uint8)
			dilation = cv2.dilate(edged,kernel,iterations = 2)
			temp8 = cv2.resize(dilation,None,fx=0.3,fy=0.3)
			

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



			# apply a four point perspective transform to both the
			# original image and grayscale image to obtain a top-down
			# birds eye view of the paper
			paper = four_point_transform(img, docCnt.reshape(4, 2))

			paper = crop_margins(paper)
			dest_path = "/home/lincy/workflow_structure/USERS/" + username + "/testing/images/"
			
			cv2.imwrite(dest_path + image_path,paper)
			print("Done processing IMAGE: ",image_path)
			# cv2.waitKey(0)


	return