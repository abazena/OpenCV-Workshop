import numpy as np
import cv2
# start a video stream from video.avi
video_stream = cv2.VideoCapture('./output/video-name.avi')
while(video_stream.isOpened()):
    # read data from video stream
    ret, frame = video_stream.read()
    # display the captured frame
    cv2.imshow('Video by OpenCV', frame)
    # exit when 'q' key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# close video stream
video_stream.release()
# close all windows
cv2.destroyAllWindows()