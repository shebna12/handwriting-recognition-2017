import numpy as np
import cv2

img = cv2.imread('droid.jpg', cv2.IMREAD_COLOR)
cv2.line(img, (0,0), (150,150), (255,0,0), 5)
cv2.rectangle(img, (124,56), (440,378), (0,255,0), 3)
cv2.circle(img, (335,172), 101, (0,0,225), 5)
pts = np.array([[272,184],[282,161],[304,144],[315,161],[321,184]], np.int32)
cv2.polylines(img, [pts], True, (0,255,255), 5)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Tadaaa!', (0,130), font, 5, (200,255,255), 3, cv2.LINE_AA)

cv2.imshow('watch with a line', img)
cv2.waitKey(0)
cv2.destroyAllWindows()