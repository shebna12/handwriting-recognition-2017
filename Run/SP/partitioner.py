# outputs a text file containing filenames of images and their corresponding label.
# The text file will then be used for parts.sh

import os
import glob
import math
from random import shuffle
def store_output(out, kind, user_folder):
	file = open(user_folder + "/" +kind + ".txt","a+")
	out = out.split("/")[1]
	print (out)
	file.write(out + "\n")
	file.close()
	
def getname_tool(training_final, n, partition, kind, user_folder):
	count = 0
	# n = 0
	
	for root,dirs,files in os.walk(training_final):
		# Uncomment this if you are working with ordered categories such as alphabets,digits, etc.
		dirs.sort()
		train = math.ceil(len(files)*partition)
		
		if (len(files)!=0):
			shuffle(files)
			for f in files:				
				print (f)
				if count < train:
					parent = root.split("/")[1]
					output = (parent + "/" + f + " "+ str(n))
					store_output(output, kind, user_folder)
					count = count + 1
			n = n+1
			count = 0

# if __name__ == '__main__':
def partition(user_folder):
	lowercase = 'abcdefghijklmnopqrstuvwxyz'
	uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

	training_final = user_folder + '/training_caffe_final'

	os.mkdir(training_final + "/lowercase")
	os.mkdir(training_final + "/uppercase")

	for letter in lowercase:
		os.system("mv "+training_final+"/"+letter+ " " +training_final+"/lowercase/")

	for letter in uppercase:
		os.system("mv "+training_final+"/"+letter+ " " +training_final+"/uppercase/")


	# the images must be sorted into folders (ex. a, b , c)
	# # TRAIN
	getname_tool(training_final+'/lowercase/', 0, 0.8, 'train', user_folder)
	getname_tool(training_final+'/uppercase/', 26, 0.8, 'train', user_folder)

	# # VAL
	getname_tool(training_final+'/lowercase/', 0, 0.2, 'val', user_folder)
	getname_tool(training_final+'/uppercase/', 26, 0.2, 'val', user_folder)

	