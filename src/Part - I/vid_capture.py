import numpy as np
import cv2
# start a video stream from camera 0
camera_stream = cv2.VideoCapture(0)
while(True):
    # read data from video stream
    ret, frame = camera_stream.read()
    # display the captured image
    cv2.imshow('Video by OpenCV', frame)
    # exit when 'q' key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# close the video stream
camera_stream.release()
# close all windows opend by opencv
cv2.destroyAllWindows()
