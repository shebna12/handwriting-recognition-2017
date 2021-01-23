import os
print ("root prints out directories only from what you specified")
print ("dirs prints out sub-directories from root")
print ("files prints out all files from root and directories")
print ("*" * 20)
for root, dirs, files in os.walk("backlater"):
    print ("root:",root)
    print ("dirs: ",dirs)
    print ("files :",files)
    for f in files:
    	print("F: ",f)