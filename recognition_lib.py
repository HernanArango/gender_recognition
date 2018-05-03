#!/usr/bin/env python
# -*- coding: utf-8 -*-
from local_binary_patterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import pickle
#opencv
import cv2


 
def recognition_images():
	# initialize the local binary patterns descriptor along with
	# the data and label lists
	desc = LocalBinaryPatterns(24, 8)

	model = pickle.load(open("models/model.save", 'rb'))

	# loop over the testing images
	for imagePath in paths.list_images("faces/testing"):
		# load the image, convert it to grayscale, describe it,
		# and classify it
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		hist = desc.describe(gray)
		prediction = model.predict([hist])[0]
	 
		# display the image and the prediction
		cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			1.0, (0, 0, 255), 3)
		cv2.imshow("Image", image)
		cv2.waitKey(0)



def recognition_image(image):
	desc = LocalBinaryPatterns(24, 8)

	model = pickle.load(open("models/model.save", 'rb'))

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	prediction = model.predict([hist])[0]
 
	return prediction



def recognition_camera():
	cam = cv2.VideoCapture(0)

	#se carga el modelo
	face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


	while(True):
	    # Capture frame-by-frame
	    ret, frame = cam.read()

	    # Our operations on the frame come here
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#detectar ccaras en el video   
	    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	    for(x,y,w,h) in faces:
	        cv2.rectangle(frame, (x, y), (x+w, y+h), (30, 144, 255), 2)

	        
	        imagen_rostro = frame[y:y+h, x:x+w]

	        
	        cv2.putText(frame, recognition_image(imagen_rostro), (x+h, y+h), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
	        

	    if cv2.waitKey(1) == 27:
	    	break  # esc to quit

	    # Display the resulting frame
	    cv2.imshow('Reconocimiento de género',frame)


	# When everything done, release the capture
	cam.release()

	cv2.destroyAllWindows()