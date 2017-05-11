#!/bin/python
import numpy as np
import cv2

# path to vidoe file
folder = '../../data/individual_keys/'
file = 'C4.mp4'
path_to_vidoe_file = folder + file
# create window
window_name = 'window'
cv2.namedWindow( window_name, cv2.WINDOW_NORMAL);
cv2.resizeWindow(window_name, 400, 550);

# create video capture object
cap = cv2.VideoCapture(path_to_vidoe_file)

# read the frames
while(cap.isOpened()):
    ret, frame = cap.read()

    cv2.imshow(window_name,frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()