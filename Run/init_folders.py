import os


def create_folders(name):
	path = "/home/lincy/workflow_structure/USERS/"
	print("in_path: ",path+name)
	if os.path.exists(path+name):

		os.mkdir(path+name+"/training_raw") # unzipped files goes here
		os.mkdir(path+name+"/training_clean")
		os.mkdir(path+name+"/training_cutter")
		os.mkdir(path+name+"/training_skews")
		os.mkdir(path+name+"/training_final")
		os.mkdir(path+name+"/training_caffe_final")
		os.mkdir(path+name+"/examples")
		os.mkdir(path+name+"/val")
		os.mkdir(path+name+"/logs")
		os.mkdir(path+name+"/train")
		os.mkdir(path+name+"/training_temp")
		os.mkdir(path+name+"/testing")
		os.mkdir(path+name+"/testing/images")
		os.mkdir(path+name+"/testing/raw")
		os.mkdir(path+name+"/models")

