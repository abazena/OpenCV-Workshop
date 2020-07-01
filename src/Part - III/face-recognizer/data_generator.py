import numpy as np
import cv2
# start a video stream from camera 0
camera_stream = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('../../../data/haarcascades/haarcascade_frontalface_alt2.xml')
# image path prefix
path_prefix = "../../../data/imgs/faces/Bazena/IMG_"
#image id for each captured image 
image_index = 0

while(True):
    # read data from video stream
    ret, frame = camera_stream.read()
    # quit if couldn't read from the camera
    if ret == False:
        print("[Error] Could not read from camera")
        break
    # display the captured frame
    cv2.imshow('Video by OpenCV', frame)
    # exit when 'q' key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    # if user pressed 'c' capture image
    if cv2.waitKey(20) & 0xFF == ord('c'):
        # convert to gray scale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # check if frame contains faces
        faces =  classifier.detectMultiScale(gray_frame)
        # if there was no faces in the frame print msg to user
        if len(faces) < 1:
            print("[Error] Couldn't detect faces in the current frame")
        for(x,y,w,h)in faces:
            # for each detected face define the regon of interest
            roi = gray_frame[y:y+h , x:x+w]
            # concatenate final img path using the path_prefix and image_index vars
            img_path = path_prefix + str(image_index) + '.jpg'
            # increment image_index by 1 for next iteration
            image_index+=1
            # save face 
            cv2.imwrite(img_path, roi)
            # print msg to user with the path of the saved face 
            print("[INFO] Face image saved:" ,img_path)
# close the video stream
camera_stream.release()
# close all windows opened by opencv
cv2.destroyAllWindows()