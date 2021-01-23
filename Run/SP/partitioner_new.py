# If folder contains mixed images of uppercase and lowercase letters 
# use this code to create a filename, to be used for partition.sh

import os
import glob
import math
from random import shuffle
import subprocess

def store_output(out, user):
	text_file = user + '_partition.txt'
	file = open("../USERS/"+user+"/"+text_file,"a+")
	out = out.split("/")[1]
	# print out
	file.write(out + "\n")
	file.close()

	text_file = '../USERS/' + user + '/' + text_file
	print("TEXT FILE: ", text_file)

	# subprocess.call(['bash', 'partition.sh', text_file, user])
	# return text_file
	
	
def getname_tool(folder_path, user):
	# n=0

	for root,dirs,files in os.walk(folder_path):
		# Uncomment this if you are working with ordered categories such as alphabets,digits, etc.
		dirs.sort()
		lettermap = { 0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n', 14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25:'z', 26:'A', 27:'B', 28:'C', 29:'D', 30:'E', 31:'F', 32:'G', 33:'H', 34:'I', 35:'J', 36:'K', 37:'L', 38:'M', 39:'N', 40:'O', 41:'P', 42:'Q', 43:'R', 44:'S', 45:'T', 46:'U', 47:'V', 48:'W', 49:'X', 50:'Y', 51:'Z' }
		
		if (len(files)!=0):
			for f in files:
				# print (f)
				parent = root.split("/")[1]
				# print "parent: ", parent
				output = (parent + "/" + f)
				store_output(output, user)

			# n = n+1
			# count = 0

# if __name__ == '__main__':
	# getname_tool('Shebna_test_final/')
	# the images must be sorted into folders (ex. a, b , c)