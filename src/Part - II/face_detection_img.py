import numpy as np
import cv2

# init a cascade classifier
classifier = cv2.CascadeClassifier('../../data/haarcascades/haarcascade_frontalface_alt2.xml')
# load image to be searched for faces
img = cv2.imread('../../data/img.jpg', -1)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# detect faces in the current frame using the cascade classifier
faces =  classifier.detectMultiScale(gray_img)
# loop through the faces detected in the frame
for(x,y,w,h)in faces:
    # for each detected face define the regon of interest
    roi_color = gray_img[y:y+h , x:x+w]
    # draw a rectangle over the current face
    color = (255,0,0) #color in BGR format 0-255
    stroke = 2 # width of line
    width = x+w # end cord of  width
    height = y+h # end cord of height
    # draw the rect
    cv2.rectangle(img , (x,y),(width , height), color, stroke)

# display the caputerd image
cv2.imshow('Face Detection by OpenCV', img)
# wait for the user to press Space Bar before closing all windows
cv2.waitKey(0)
# close all windows opend by opencv
cv2.destroyAllWindows()
