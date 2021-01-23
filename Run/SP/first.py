import cv2

image = cv2.imread("clouds.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Clouds in the eyes of humans", image)
cv2.imshow("Clouds in the eyes of dogs", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()