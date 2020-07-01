import numpy as np
import cv2
# start a video stream from camera 0
camera_stream = cv2.VideoCapture(0)
# create a VideoWriter object 
videoWriter = cv2.VideoWriter('./output/video-name.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (640,480))
while(True):
    # read data from video stream
    ret, frame = camera_stream.read()
    # add the current frame the video
    videoWriter.write(frame)
    # display the captured image
    cv2.imshow('Video by OpenCV', frame)
    # exit when 'q' key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# close video stream
camera_stream.release()
#  close video video writer
videoWriter.release()
# close all windows opend by opencv
cv2.destroyAllWindows()