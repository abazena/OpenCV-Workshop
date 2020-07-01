import pickle as plk
import cv2
import numpy as np
#import rpi_gpio as GPIO

# init a cascade classifier
classifier = cv2.CascadeClassifier('../../../data/haarcascades/haarcascade_frontalface_alt2.xml')
# init an LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.createLBPHFaceRecognizer()
# load saved model data
recognizer.read("../../../output/faces.yml")
ids_labels = {}
# load saved labels
with open("../../../output/face-labels.plk", 'rb')as f:
    tmp_labels = plk.load( f)
    '''list comprehension: swap the keys and values in the list that was loaded from file'''
    ids_labels= {v:k for k,v in tmp_labels.items()}
# start a video stream from camera 0
video_stream = cv2.VideoCapture(0)
# call the setup function from our GPIO util


while(True):
    # read data from video stream
    ret, frame = video_stream.read()
     # quit if couldn't read from the camera
    if ret == False:
        print("[Error] Could not read from camera")
        break
    # convert to gray scale
    gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    # detect faces in the current frame using the cascade classifier
    faces =  classifier.detectMultiScale(gray_frame)
    # loop through the faces detected in the frame
    for(x,y,w,h)in faces:
        # for each detected face define the regon of interest
        roi_color = frame[y:y+h , x:x+w]
        roi_gray = gray_frame[y:y+h , x:x+w]
        # predict the detected face 
        _id, conf = recognizer.predict(roi_gray)
        # name of the predicted face
        name = ids_labels[_id]
        print(conf)
        # text font 
        font = cv2.FONT_HERSHEY_SIMPLEX
        # color of text
        color = (0,0,0)
        stroke = 2
        if name == "unknown" or conf >= 80 or conf < 40:
            name = "unknown"
            color = (0,0,255)
        else:
            # draw a rectangle over the current face
            color = (0,255,0) #color in BGR format 0-255
        # output the text using the afore selected font, color and stroke
        cv2.putText(frame, name,(x,y), font,1,color,stroke,cv2.LINE_AA)
        width = x+w # end cord of  width
        height = y+h # end cord of height
        # draw the rect
        cv2.rectangle(frame , (x,y),(width , height), color, stroke)
    # show current frame with all faces and rects
    cv2.imshow('FR By OpenCV', frame)
    # exit when 'q' key is pressed 
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# close the video stream 
video_stream.release()
# set GPIOs to low
# close all windows opened by opencv 
cv2.destroyAllWindows()
