import numpy as np
import cv2
# load image (./image.jpg)
img = cv2.imread('img.jpg', 0)
# display the captured image
cv2.imshow('Image by OpenCV', img)
# wait for the user to press Space Bar before closing all windows
cv2.waitKey(0)
# close all windows opend by opencv
cv2.destroyAllWindows()
