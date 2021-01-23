import numpy as np 
import cv2

img = cv2.imread('droid.jpg', cv2.IMREAD_COLOR)


# img[55,55] = [255,255,255]
# px = img[55,55] #location of the pixel
#img[58:341, 151:439] = [255,255,255] # img[row:row+height,col + col+width]
# print(px)
watch_face = img[87:268, 237:422]
#watch_face = [0,0,0]
img[0:181, 0:185] = watch_face
cv2.imshow('NEW',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#ROI(Region of Image) => sub-image of image
