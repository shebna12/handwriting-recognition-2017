import sklearn
import cv2
import numpy
import sys
import scipy.ndimage
from  scipy import ndimage
im = cv2.imread("A_39878.jpg")
# im = numpy.zeros((20, 20))
# im[2:6, 2:14] = 1
print(type(im))
# sys.exit(0)
# Determine Centre of Mass

com = ndimage.measurements.center_of_mass(im)

print(com)

# Translation distances in x and y axis

x_trans = int(im.shape[0]//2-com[0])
y_trans = int(im.shape[1]//2-com[1])

# Pad and remove pixels from image to perform translation

if x_trans > 0:
    im2 = numpy.pad(im, ((x_trans, 0), (0, 0)), mode='constant')
    im2 = im2[:im.shape[0]-x_trans, :]
else:
    im2 = numpy.pad(im, ((0, -x_trans), (0, 0)), mode='constant')
    im2 = im2[-x_trans:, :]

if y_trans > 0:
    im3 = numpy.pad(im2, ((0, 0), (y_trans, 0)), mode='constant')
    im3 = im3[:, :im.shape[0]-y_trans]

else:
    im3 = numpy.pad(im2, ((0, 0), (0, -y_trans)), mode='constant')
    im3 = im3[:, -y_trans:]

cv2.imshow("NEW",im3)

print(ndimage.measurements.center_of_mass(im3))