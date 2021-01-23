
import glob
import cv2
import numpy as np 	
import math

import os




def count_instances(folder_path):
	nvm = 0
	numfiles = 0
	num = 0
	all_files = []
	for root, dirs, files in os.walk(folder_path):
		for file in files:
			all_files.append(file)

	
	# print("ALL FILES: ",all_files)
	instaces_count = [file[0] for file in all_files]
	# print("Contents: ",instaces_count)
	alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
	for letter in alphabet:
		count = instaces_count.count(letter)
		print(letter + ": ",instaces_count.count(letter))





			
			

# Replace folder name (must be in the same directory as the code)
count_inst = count_instances("TRIAL9")