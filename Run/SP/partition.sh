#!/usr/bin/env sh

SRC=$2
DEST=$3

echo "1st argument: $1"
echo "2nd argument: $2"


# # open text file to get the file name of the images
cat $1 | while read line; do 				
	filename="$(echo $line | cut -d' ' -f1 )"
	letter="$(echo $filename | cut -d'_' -f1)"

	folder="$(echo $DEST | cut -d'/' -f7)"

	echo "FOLDER: "$folder

	if [ $folder = "training_caffe_final" ] 
	then
		mkdir -p $DEST/$letter
		# copy to training_final
		cp "$SRC/$letter/$filename" "$DEST/$letter"
	
	elif [ $folder = "train" ] || [ $folder = "val" ] || [ $folder = "training_final" ]
	then
		# copy to train
		cp "$SRC/$letter/$filename" "$DEST"
	
	else
		mkdir -p $DEST/$letter
		# copies the image file to its own folder - training_temp
		mv "$SRC/$filename" "$DEST/$letter"
	fi

	

	echo "$filename"
done

echo "----- DONE -----"