# Face classification and beauty scoring
The project's aim is to build a human face classification model using Neural Network. The input face would be classified 
into one of the four classes: Asian Male - Asian Female - European Male - European Female. The model would then rate the
face's beauty score in a value range of [1; 5] with 1 is the less beautifulness and 5 is highest.  
Code written in Python using Keras.  The dataset used is the SCUT-FBP5500 Database (https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)  
The trained model was uploaded under the name model_31_July.h5  
Run the file webcam.py and the program would automatically detect any face in front of the webcam (and draw a blue retangle around the face) 
then classify and score it. Press the 'q' button to stop the program.  
Download the model and remember to customize webcam.py with the correct directory path to the model.  
\
My face as input:  Male - Asian - 2.89/5
![apt get](https://github.com/lmhoang45/Beauty-Scoring/blob/master/Screenshot%20from%202018-08-19%2020-34-06.png)  
\
Gal Gadot as input:  Female - Asian - 3.48/5 (my model underrated her a lot haha:) )
![apt get](https://github.com/lmhoang45/Beauty-Scoring/blob/master/Screenshot%20from%202018-08-19%2020-35-48.png)  
\
Barack Obama as input:  Male - European - 3.27/5
![apt get](https://github.com/lmhoang45/Beauty-Scoring/blob/master/Screenshot%20from%202018-08-19%2020-40-53.png)    

# Prerequisites
Required libraries and frameworks: opencv - numpy - skimage  
# Note
The used model is DenseNet121 (https://keras.io/applications/#densenet).  
The model is trained in Google Colab and then saved in drive under the .h5 file type.
