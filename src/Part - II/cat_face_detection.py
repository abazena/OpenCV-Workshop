import numpy as np
import cv2
# init a cascade classifier
classifier = cv2.CascadeClassifier('../../data/haarcascades/haarcascade_frontalcatface_extended.xml')
# start a video stream from camera 0
camera_stream = cv2.VideoCapture(0)

while(True):
	# read data from video stream
	ret, frame = camera_stream.read()
	if ret == False:
		print("Error: Couldn't read from camera stream")
		break
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the current frame using the cascade classifier
	cat_faces =  classifier.detectMultiScale(gray_frame)
	# loop through the faces detected in the frame
	for(x,y,w,h)in cat_faces:
		# for each detected face define the regon of interest
		roi_color = gray_frame[y:y+h , x:x+w]
		# draw a rectangle over the current face
		color = (255,0,0) #color in BGR format 0-255
		stroke = 2 # width of line
		width = x+w # end cord of  width
		height = y+h # end cord of height
		# draw the rect
		cv2.rectangle(frame , (x,y),(width , height), color, stroke)
	# show current frame with all faces and rects
	cv2.imshow('Cat Face Detection by OpenCV', frame)
	# exit when 'q' key is pressed
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
# close the video stream
camera_stream.release()
# close all windows opend by opencv
cv2.destroyAllWindows()
