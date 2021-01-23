import os
from flask import Flask, render_template, request, flash, redirect, url_for, session, Session
from werkzeug import secure_filename
import zipfile
import init_folders as init 
import main
from partitioner_new import getname_tool
import subprocess
from partitioner import partition
from randomizer import randomize
from document_detection import document_detect
from testing_document_detection import test_document_detect
from train_models import train_models
from workflow import workflow_main
import workflow as work
import math



import sys
sys.path.append('/home/lincy/workflow_structure/Run/SP')

app = Flask(__name__)
sess = Session()
# app.secret_key = "super secret key"

EXTENSIONS = set(['zip', 'rar'])
IMAGE_EXTENSIONS = set(['jpg', 'png', 'jpeg'])
UPLOAD_FOLDER = ''
home = os.path.expanduser("~/")

username=0
userfolder=0
filename=0
current_users = home + 'workflow_structure/USERS'
curUsers = next(os.walk(current_users))[1]
option=0
labels=0
num=0
thr=0



app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def load_indexNew():
	return render_template('homeNew.html', curUsers=curUsers)

@app.route('/home')
def load_home():

	return render_template('homeNew.html', curUsers=curUsers)
	

@app.route('/uploadtest_success/<user>/successful', methods=['GET','POST'])
def uploadtest_success(user):

	if request.method == 'POST':
		uploadtest(user)

	else:
		return render_template('existingFormNew.html', user=user, message="You successfully trained your model!")
	# return "upload test"




@app.route('/uploadtest/<user>', methods=['GET','POST'])
def uploadtest(user):
	print("==============================================")
	

	if request.method == 'POST':


		global userfolder
		userfolder = home+'workflow_structure/USERS/'+ username

		UPLOAD_FOLDER = home+'workflow_structure/USERS/'+ username +'/testing/raw'

		# check if the post request has the file part
		if 'file' not in request.files:
			flash('no file part')
			error = 'no file part'
			return render_template('existingFormNew.html',error=error)
	
		f = request.files['file']

		

		# if user does not select file, browser also submit a empty part without filename
		if f.filename == '':
			flash('no selected file')
			error = 'no selected file'
			return render_template('existingFormNew.html',error=error)
		
		if f and allowed_files(f.filename):
			global filename
			filename = secure_filename(f.filename)
			f.save(os.path.join(UPLOAD_FOLDER , filename))

			# ===========================================
			#  gets option Labeled of Unlabeled
			# ===========================================
			global option
			option = request.form['label']
		
			if option == "labeled":
				global num
				num = request.form['test_images']
			test_document_detect(username)
			# document_detect_test(user)

			workflow_main(username, num)

			img_labels = work.labels
			print("*******************************")
			print("IMAGE LABEL labels: ", work.labels)
			print("*******************************")


		


				# print("Real words: ", work.real_words)
				# print("Accuracy of word segementation: ", work.word_seg_acc)
				# print("Accuracy of word recognition: ", work.word_recog_acc)

			image = "/static/" + username +'/'+ filename

			os.system("mkdir -p " + home + '/workflow_structure/Run/static/' + username)
			os.system("cp -p " + userfolder + "/testing/raw/"+ filename + ' ' + home + '/workflow_structure/Run/static/'+username+'/')

			print("=========================================")
			print("=========================================")
			print("IMG LABELS: ", img_labels)

			print("=========================================")
			print("=========================================")
			print("REAL WORDS: ", work.real_words)

			print("NUM: ", num)
			
			os.system("rm " + UPLOAD_FOLDER + "/" + filename)
			os.system("rm " + home + '/workflow_structure/USERS/'+ username  + "/testing/images/" + filename)


			if option == "labeled":
				work.word_seg_acc = work.word_seg_acc * 100
				work.word_recog_acc = work.word_recog_acc * 100

				word_seg = "{0:.2f}".format( work.word_seg_acc)
				word_recog = "{0:.2f}".format( work.word_recog_acc)

				print("word_seg: ", word_seg)
				print("word_recog: ", word_recog)

				#  ===============================================================================================
				# if Labeled button is clicked
				return render_template('result.html', words=work.real_words, seg=word_seg, recog=word_recog, image=image, option=option, labels=work.labels)
		
			# if Unlabeled button is cliked
			else:
				return render_template('result.html', words=work.real_words, option=option, image=image, seg='', recog='', labels='')
				# return render_template('result.html', words=work.real_words, seg=word_seg, recog=word_recog, image=image, option=option, labels=work.labels)
				
			#  ===============================================================================================
	else:
		# print("URL: ",request.url)


		# # print("user:", user )
		global username
		username = user


		print("USER: ", user)
		print("USERNAME: ", username)
		return render_template('existingFormNew.html', user=username, message='')
	# return "upload test"


# 1
@app.route('/upload', methods=['GET','POST'])
def upload_file():

	if request.method == 'POST':

		# take user's name and create user's folder
		user = request.form['name']

		if user == "":
			error = "user did not enter name"
			return render_template("indexNew.html", error=error)

		global username
		username = user

		print("USER: ", user)

		UPLOAD_FOLDER = home+'workflow_structure/USERS/'+ user
		

		if not os.path.exists(UPLOAD_FOLDER):
			os.mkdir(UPLOAD_FOLDER)
		
		# create folders using init_folders.py's create_folders
			init.create_folders(user)
		

		# check if the post request has the file part
		if 'file' not in request.files:
			flash('no file part')
			error = 'no file part'
			return render_template('indexNew.html',error=error)
		
		f = request.files['file']
		
		# if user does not select file, browser also submit a empty part without filename
		if f.filename == '':
			flash('no selected file')
			error = 'no selected file'
			return render_template('indexNew.html',error=error)
		if f and allowed_files_database(f.filename):

			filename = secure_filename(f.filename)
			f.save(os.path.join(app.config['UPLOAD_FOLDER'] , filename))
			
			unzip_file(filename, UPLOAD_FOLDER)		
			# task = unzip_file.delay(filename, UPLOAD_FOLDER)


		# ================================================
			# return render_template('loading.html')
			# =========== upload successful ==============
			return redirect(url_for('uploadtest_success', user=user))

			
	else:
		
		return render_template('indexNew.html', curUsers=curUsers)


def allowed_files_database(filename):
	return '.' in filename and \
		filename.rsplit('.',1)[1].lower() in EXTENSIONS

def allowed_files(filename):
	return '.' in filename and \
		filename.rsplit('.',1)[1].lower() in IMAGE_EXTENSIONS

def unzip_file(filename, user_folder):

	
	user = request.form['name']
	global username
	username = user

	global userfolder
	userfolder = user_folder
	
	zip_file = zipfile.ZipFile(filename, 'r')

	zip_file.extractall(user_folder + '/training_raw') # change this to '/training_raw'
	zip_file.close()

	# task = start_workflow.delay(user, user_folder)
	# return render_template('loading.html')
	start_workflow(user, user_folder)

# =====================================================
	# return redirect(url_for('uploadtest', user=user))
# @celery.task
def start_workflow(user, user_folder):

	# call main - responsible for the cutting and preprocessing
	main.main(user)

	
	print("PATH: ", user_folder)


	print("================================================================")
	print("================================================================")
	print("================================================================")
	print("================================================================")
	

	text_file = user_folder+'/'+user+'_partition.txt'
	src = user_folder + '/training_temp'



	# creates a textfile containing all the cut images from the dataset of the user
	getname_tool(user_folder + '/training_temp', user)

	dest = user_folder + '/training_temp'
	
	# transfers the cut images to their specific folders
	subprocess.call(['bash', 'SP/partition.sh', text_file, src, dest])


	randomize(user_folder)


	dest = user_folder + '/training_caffe_final'
	
	# partition the images to their specific folders in training_caffe_final
	subprocess.call(['bash', 'SP/partition.sh', user_folder+'/final_partition.txt', src, dest])
	
	dest = user_folder + '/training_final'

	# copies images to training_final folder with subfolder of letters
	subprocess.call(['bash', 'SP/partition.sh', user_folder+'/final_partition.txt', src, dest])


	# train SVM and random forest
	print("========== training SVM and Random Forest ============")
	train_models(user)


	# call partitioner.py that outputs a text file containing images and their corresponding labels
	partition(user_folder)


	print("=============== partition train =============")
	dest = user_folder + '/train'
	# separate train from val
	subprocess.call(['bash', 'SP/partition.sh', user_folder+'/train.txt', src, dest])

	print("=============== partition val =============")
	dest = user_folder + '/val'
	subprocess.call(['bash', 'SP/partition.sh', user_folder+'/val.txt', src, dest])


	caffe_folder = home + 'workflow_structure/caffe/'

	examples = user_folder + '/examples'
	data = user_folder
	tools = caffe_folder + 'build/tools'

	train_data_root = user_folder + '/train/'
	val_data_root = user_folder + '/val/'

	print("================ create LMDB file ===============")
	# create LMDB file; MAKE SURE TO FIX PATHS
	subprocess.call(['bash', 'SP/create_lmdb.sh', examples, data, tools, train_data_root, val_data_root])
	
	print("================ create mean binary proto ===============")
	# create mean binary proto
	subprocess.call(['bash', 'SP/make_mean.sh', examples, data, tools])

	print("===================================================================")
	print("Editing lenet_train_val.prototxt")
	print("===================================================================")
 
	# edit source of lmdb file in lenet_train_val.prototxt
	with open(home + 'workflow_structure/Run/SP/lenet_train_val.prototxt') as infile, open (user_folder + "/lenet_train_val.prototxt", "w") as outfile:
		for i, line in enumerate(infile):
			if i == 13:
				# line 14
				outfile.write("source: \""+user_folder+"/examples/sp_train_lmdb\"\n")
			elif i == 30:
				# line 31
				outfile.write("source: \""+user_folder+"/examples/sp_val_lmdb\"\n")
			else:
				outfile.write(line)

	print("===================================================================")
	print("Editing solver.prototxt")
	print("===================================================================")

	test_iter = math.ceil(len([name for name in os.listdir(user_folder + '/train') if os.path.isfile(os.path.join(user_folder + '/train', name))]) / 64)
	test_interval = math.ceil(len([name for name in os.listdir(user_folder + '/val') if os.path.isfile(os.path.join(user_folder + '/val', name))]) / 64) 

	print("========= test iter: ", test_iter)
	print("========= test interval: ", test_interval)

	# edit snapshot and net prefix in solver.prototxt
	with open(home + 'workflow_structure/Run/SP/solver.prototxt') as infile, open (user_folder + "/solver.prototxt", "w") as outfile:
		for i, line in enumerate(infile):
			if i == 0:
				# line 1:
				outfile.write("net: \""+user_folder+"/lenet_train_val.prototxt\"\n")
			elif i == 1:
				# line 2
				outfile.write("test_iter: "+ str(test_iter)+"\n")
			elif i == 2:
				# line 3
				outfile.write("test_interval: "+str(test_interval)+"\n")
			elif i == 12:
				# line 13
				outfile.write("snapshot_prefix: \""+user_folder+"/models\"\n")
			else:
				outfile.write(line)

	print("================ copying lenet_deploy.prototxt to user folder ===============")
	os.system("cp " + home + "/workflow_structure/Run/SP/lenet_deploy.prototxt " + user_folder)

	train_caffe = tools + "/caffe"

	print("================== training neural net ==================")
	# script for training
	subprocess.call(['bash', 'SP/train.sh', train_caffe, user_folder])

	

if __name__ == '__main__':
	
	app = Flask(__name__)
	# app.config['CELERY_BROKER_URL'] = 'redis://localhost:5000'
	# app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:5000'

	# celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
	# celery.conf.update(app.config)

	# 