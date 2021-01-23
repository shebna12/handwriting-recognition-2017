import cv2
from utils import remove_noise,get_ave_area

def resize_img(thresh):
	h,w = thresh.shape[:2]
	ar = w / h 
	nw = 1300
	nh = int(nw / ar)        
	nimage = cv2.resize(thresh,(nw,nh))
	
	return nimage

image = cv2.imread("testinglang.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
_,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY) # threshold
nimage = resize_img(thresh)
cv2.imshow("THRESH",nimage)
thresh = cv2.bitwise_not(thresh)
nimage = resize_img(thresh)
cv2.imshow("DILATED:",nimage)
_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours

contours = remove_noise(contours)
average_area = get_ave_area(contours)
print(len(contours))
print (average_area)
for contour in contours:
    [x,y,w,h] = cv2.boundingRect(contour)

    if((w*h) <= average_area):
    	cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
image = resize_img(image)
cv2.imshow("imahe",image)

cv2.waitKey(0)
# write original image with added contours to disk  

cv2.imwrite("contoured.jpg", image)