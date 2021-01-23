import os
import glob
import math
from random import shuffle
def store_output(out, user_folder):
	
	file = open(user_folder+"/final_partition.txt","a+")
	out = out.split("/")[1]
	# print out
	file.write(out + "\n")
	file.close()
	
def getname_tool(training_temp, user_folder):
	count = 0
	# n = 0

	print("folder user_folder: ", training_temp)
	for root,dirs,files in os.walk(training_temp):
		# Uncomment this if you are working with ordered categories such as alphabets,digits, etc.
		dirs.sort()
		
		if (len(files)!=0):
			shuffle(files)
			for f in files:				
				# print f
				# if count < total:
				parent = root.split("/")[1]
				output = (parent + "/" + f )
				store_output(output, user_folder)
				count = count + 1
			# n = n+1
			count = 0


def randomize(user_folder):
	# the images must be sorted into folders (ex. a, b , c)
 
	# total = 200
	
	training_temp = user_folder + '/training_temp'

	# lowercase = 'abcdefghijklmnopqrstuvwxyz'
	# uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

	# os.mkdir(final_user_folder + "/lowercase")
	# os.mkdir(final_user_folder + "/uppercase")

	# for letter in lowercase:
		# os.system("mv "+final_user_folder+"/"+letter+ " " +final_user_folder+"/lowercase/")

	# for letter in uppercase:
		# os.system("mv "+final_user_folder+"/"+letter+ " " +final_user_folder+"/uppercase/")



	getname_tool(training_temp, user_folder)

