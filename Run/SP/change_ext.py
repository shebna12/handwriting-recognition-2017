import os
import glob

def change_ext(folder_path,imtype):
	files=[]
	n = 0
	print("The following are files found in " + folder_path + " with ." + imtype + " format:")
	# print(glob.glob(folder_path+'/'+'*/*.' + imtype))
	print("Renaming files...")
	for file in glob.glob(folder_path+'/'+'*/*.' + imtype):
		# n = n + 1
		files.append(file.replace('\\','/'))
		# print(x)
		# print(x.split('_')[2])
		# print(files)
	for x in files:
		old_name = (x.split('/')[2])
		new_name = (folder_path + '/' + x.split('/')[1] + '/' + x.split('/')[1] + '_' + x.split('.')[1])
		print("Old name: ",x)
		print("New name: ",new_name)
		# os.rename(x,new_name)
	


change_ext("dataset2k18_2","jpg")