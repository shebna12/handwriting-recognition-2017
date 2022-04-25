# Offline Handwriting Block Text Word Recognition using One-User Only Approach 
This is a backup copy of me and Ms. Lincy Legada's undergraduate thesis codes. Final version copy and hardbound of the paper are under the care of the Faculty of Computer Science, University of the Philippines Visayas. An abstract of this work is provided in the README file of this repo. 

## ABSTRACT

Handwritten notes and documents are tedious to deal with especially if they pile up over the years. The digitized or soft copy of the document is easier to manage compared to the hard copy. The current state-of-the-art handwriting recognition systems require a large computational and hardware requirement. The researchers aimed to address the problem of this requirement by creating a system that converts the image containing words written in block letters into text.

  

The block-handwritten word recognition system was developed using Prototyping Development approach. The system is mainly composed of two modules, training and testing. In training, the user is required to upload or her own dataset which is used in order to create a model for SVM, Random Forest and LeNet. An ensemble was created by combining the three classifiers. The model in turn will be used in testing to classify each letter, and them combine the letters into words. The testing experiments were done using 20 images with 12-15 words each image with an average of 79% segmentation accuracy.

  

According to the results, it was found out that the ensemble returned the highest accuracy for word recognition. Using individual classifiers on SVM had 58.7% accuracy, Random Forest had 33% accuracy, LeNet had 38.2% accuracy. Combining the three classifiers into one ensemble model yields a 59.9% accuracy.

  

The results of the ensemble model using only a certain user’s handwriting proved to be more accurate as compared to using a general population’s handwriting dataset. Using a certain user’s handwriting on the ensemble model returned a 59.9% accuracy while using a general population’s handwriting only had 53.7%.

This study also showed the importance of applying post-processing on word recognition. Without applying post-processing using the ensemble model only had a 36.6% accuracy. Applying post-processing improved the previous accuracy into 59.9%.


## Methodology

![flow chart image](https://lh3.googleusercontent.com/pw/ACtC-3e9teMYCxQ2xCwQDC4f3_5V9nUdv77ZabAColF2mt3gz9-U2R4o8QhT3dqHtMYnBlruS5AB6NYVnepFjphvuabNHX4UtWTyPtgANq-NYWR4mA2VvUIawQI2qgpG6NwjVQEW66Jr6j-u8-W36rGV4-zdaA=w781-h423-no?authuser=0)


## To Run

 1. Install the required libraries. (Make sure you use Python 3.5)
 2. Open your console/terminal.
 3. Change current working directory to the `RUN` directory.
 4. Type `export FLASK_APP=index.py`, press *Enter*.
 5. Type `export FLASK_DEBUG=1`, press *Enter*.
 6. Type `python -m flask run`, press *Enter*.
 7. Open your browser and go to `localhost:5000` or `127.0.0.1:5000`


## Required libraries

Python 3.5
Flask 1.0.2

## Note
Feel free to create an issue if you have any questions. I'll try to get back at it as soon as I can. 

