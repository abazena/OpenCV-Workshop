import time
import numpy as np
import cv2
# init a cascade classifier
classifier = cv2.CascadeClassifier('../../data/haarcascades/haarcascade_frontalface_alt2.xml')
# start a video stream from camera 0
camera_stream = cv2.VideoCapture(0)
prefix = "Detected_face_"
while(True):
	# read data from video stream
	ret, frame = camera_stream.read()
	if ret == False:
		print("Error: Couldn't read from camera stream")
		break
	# convert image to gray scale
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the current frame using the cascade classifier
	faces =  classifier.detectMultiScale(gray_frame)
    # loop through the faces detected in the frame
	for(x,y,w,h)in faces:
		# for each detected face define the regon of interest
		roi = frame[y:y+h , x:x+w]
		img_path = '../../output/' + prefix + str(time.time()) + ".jpg"
		cv2.imwrite(img_path, roi)
    # exit when 'q' key is pressed
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
	cv2.imshow('OpenCV Face Detection', frame)
# close the video stream
camera_stream.release()
# close all windows opend by opencv
cv2.destroyAllWindows()
