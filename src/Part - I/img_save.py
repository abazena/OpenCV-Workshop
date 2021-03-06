import numpy as np
import cv2
# start a video stream from camera 0
camera_stream = cv2.VideoCapture(0)
# read data from video stream
ret, frame = camera_stream.read()
# display the captured image
cv2.imshow('Image by OpenCV', frame)
# save the image to '../../output/img.jpg'
cv2.imwrite('img.jpg', frame)
# wait for the user to press Space Bar before closing all windows
cv2.waitKey(0)
# close the video stream
camera_stream.release()
# close all windows opend by opencv
cv2.destroyAllWindows()