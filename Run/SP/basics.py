import numpy as np
import cv2

img = cv2.imread('simple.jpg',0)
cv2.imwrite('graysimple.png',img)
new_img = cv2.imread('graysimple.png',1)
cv2.imshow('A simple scene', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()