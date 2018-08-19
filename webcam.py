from __future__ import print_function
import cv2
import numpy as np
#from scipy import misc
#import pylab as pl
#from matplotlib import pyplot as plt
#from PIL import Image
import os
from skimage.util import invert
from loss_accuracy import custom_loss, gender_accuracy, race_accuracy, beauty_loss, beauty_accuracy
from keras.models import load_model
import keras.losses
keras.losses.custom_loss = custom_loss

model = load_model('/home/lmhoang45/Downloads/model_31_July.h5')


face_cascade = cv2.CascadeClassifier('/home/lmhoang45/anaconda3/pkgs/libopencv-3.4.1-h1a3b859_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while(True):
    (ret, image) = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#    cv2.imshow("Hello", image)
    if len(faces) > 0:
        a = '+ '
#        cv2.putText(image, text = a, org = (int(20),int(20)), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0, 0, 0))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            human = image[y:y+h, x:x+w]
            human = cv2.resize(human, (224,224))
            cv2.imshow("Input Face", human)
            human = np.reshape(human, [1,224,224,3])
            human = human/255
        result = model.predict(human)
        if result[0][0] > result[0][1]:
            a += 'Male '
        else:
            a += 'Female '

        if result[0][2] > result[0][3]:
            a += 'European: '
        else:
            a += 'Asian: '
        a += str(round(4000*result[0][4]+1000)/1000)
        a += '/5'
#        for i in range(4):
#            print(result[0][i])
#        print(result)
#        print('\n')
        cv2.putText(image, text = a, org = (int(20),int(20)), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0, 0, 0))
    cv2.imshow("Overview", image)    
    
#    cv2.waitKey(0)
#    d = c & 0xFF    
#    if d == ord('z'):
#    cv2.destroyAllWindows()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()




#img = cv2.imread('/home/lmhoang45/Desktop/face.jpg')
#img = cv2.resize(img,(224,224))
#img = np.reshape(img,[1,224,224,3])
#img = preprocess_input(img)

#classes = model.predict(img)

#print(classes)
#print(type(classes))




#import numpy as np
#import cv2

#cap = cv2.VideoCapture(0)

#while(True):
#    # Capture frame-by-frame
#    ret, frame = cap.read()

#    # Our operations on the frame come here
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#    # Display the resulting frame
#    cv2.imshow('frame', gray)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

## When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()
