from pyimagesearch.transform import four_point_transform
from pyimagesearch import imutils
from skimage.filters import threshold_adaptive
import numpy as numpy
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",required = True, help = "Path to the image to be scanned")
args = vars(ap.parse_args())