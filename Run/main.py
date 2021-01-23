import sys
sys.path.append('/home/lincy/workflow_structure/Run/SP')
from cutter import cut_ij_dataset
from cutter_general import cut_dataset
from skew_auto import initialize_skew
from preprocess_dataset import preprocess_dataset_ij
from preprocess_dataset_general import preprocess_dataset_general
from train_models import train_models
from document_detection import document_detect
from testing_document_detection import test_document_detect
from workflow import workflow_main
from init_folders import create_folders
import os
import shutil

def get_ij_images(username):
	root_path = "/home/lincy/workflow_structure/USERS/" + username+ "/"
	folder_path = root_path + "training_clean/"
	if not os.path.exists(folder_path+"ij"):
		os.mkdir(folder_path+"ij")
	if not os.path.exists(folder_path+"others"):
		os.mkdir(folder_path+"others")


	# numfiles = 0
	# num = 0
	for root, dirs, files in os.walk(folder_path):
		for image_path in files:   
			# numfiles = numfiles + 1
			image_name = image_path[0]
			print("image_name: ",image_name)
			if image_name == 'i' or image_name == 'j':
				try:
					os.rename(folder_path  + image_path, folder_path + "ij/" + image_path)
				except FileNotFoundError:
					continue

			else:
				try:
					os.rename(folder_path  + image_path, folder_path + "others/" + image_path)
				except FileNotFoundError:
					continue
				


def main(name):


	document_detect(name)
	get_ij_images(name)
	cut_dataset(name)
	cut_ij_dataset(name)
	initialize_skew(name)
	preprocess_dataset_general(name)
	preprocess_dataset_ij(name)
	

	