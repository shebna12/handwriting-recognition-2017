import cv2
import numpy as np 
import imutils

image = cv2.imread("tanga.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)


_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
'''
for c in contours:
    area = cv2.contourArea(c)
    [x, y, w, h] = cv2.boundingRect(c)

    # if (area > 20 and area < 2000):
    #     [x, y, w, h] = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
'''
for contour in contours:
    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(contour)

    # discard areas that are too large
    if h>300 and w>300:
        continue

    # discard areas that are too small
    if h<50 or w<50:
        continue

    # draw rectangle around contour on original image
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)



print ('Number of contours are: %d -> ' %len(contours))

cv2.imshow("RESULT",image)
cv2.waitKey(0)
cv2.destroyAllWindows()