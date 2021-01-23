import cv2
import numpy
import matplotlib.pyplot as py

img = cv2.imread('droid.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('Android Watch',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('droid_graysale.jpg',img)