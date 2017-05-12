#!/bin/python
import numpy as np
import cv2
import keripiav_helper_functions as helper

# path to vidoe file
folder = '../data/individual_keys/'
file = 'C4.mp4'
path_to_vidoe_file = folder + file
# create window
window_name = 'window'
cv2.namedWindow( window_name, cv2.WINDOW_NORMAL);
cv2.resizeWindow(window_name, 400, 550);

# create video capture object
cap = cv2.VideoCapture(path_to_vidoe_file)

if(not cap.isOpened()):
    print "error reading video\n"

# crop zone. x is vertical, y is horizontal for numpy. reverse for opencv ?
cropped_origin = (500,50)
cropped_end = (1600,850)
cropped_size = (cropped_end[0]-cropped_origin[0],cropped_end[1]-cropped_origin[1])

# read the frames
while(cap.isOpened()):
    # read the frame
    ret, frame = cap.read()
    frame = frame[cropped_origin[0]:cropped_end[0],cropped_origin[1]:cropped_end[1],:]

    # converto to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # to find the black pixels in the image, we look for low saturation and low value
    low_treshold_hsv_black = np.array([0,0,0])
    high_treshold_hsv_black = np.array([255,70,30])
    black_pixels = cv2.inRange(hsv_frame, low_treshold_hsv_black, high_treshold_hsv_black)
    # to find the white pixels, we look for low saturation and high value
    low_treshold_hsv_white = np.array([0,0,200])
    high_treshold_hsv_white = np.array([255,35,255])
    white_pixels = cv2.inRange(hsv_frame, low_treshold_hsv_white, high_treshold_hsv_white)

    # image composed of the white and black pixels
    black_white_image = black_pixels + white_pixels

    # cv2.imshow(window_name,black_white_image)
    # # wait for a key press, and exit if Esc is pressed
    # if cv2.waitKey() & 0xFF == 27 :
    #     break

    # close and open the image

    closing_kernel_size = 13;
    closing_kernel = np.ones((closing_kernel_size, closing_kernel_size))
    # closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, closing_kernel)
    closed_image = cv2.morphologyEx(black_white_image, cv2.MORPH_CLOSE, closing_kernel)
    # cv2.imshow(window_name,closed_image)
    # # wait for a key press, and exit if Esc is pressed
    # if cv2.waitKey() & 0xFF == 27 :
    #     break

    # opening_kernel_size = 7;
    # opening_kernel = np.ones((opening_kernel_size, opening_kernel_size))
    # # opened_image = cv2.morphologyEx(black_white_image, cv2.MORPH_OPEN, opening_kernel)
    # opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, opening_kernel)
    # cv2.imshow(window_name,opened_image)
    # # wait for a key press, and exit if Esc is pressed
    # if cv2.waitKey() & 0xFF == 27 :
    #     break

    # src = opened_image
    # src = closed_image
    # print closed_image
    src = cv2.cvtColor(closed_image, cv2.COLOR_GRAY2BGR)
    # src[:,:,0] = 0

    cv2.imshow(window_name,src)
    # wait for a key press, and exit if Esc is pressed
    if cv2.waitKey() & 0xFF == 27 :
        break

    # find the edges and lines
    # edges = np.array(src.shape)
    edges = cv2.Canny(src, 2500, 2700, 0, 5)
    cv2.imshow(window_name,edges)
    # wait for a key press, and exit if Esc is pressed
    if cv2.waitKey() & 0xFF == 27 :
        break

    nImtersectionPoints = 120
    minLineLength = 180
    maxLineGap = 35
    lines_tmp = cv2.HoughLinesP(edges,1,np.pi/180,nImtersectionPoints,0,minLineLength,maxLineGap)
    # print lines_tmp
    lines = np.squeeze(lines_tmp)
    leftmost_vertical_line = [cropped_size[1],0,cropped_size[1],1]
    rightmost_vertical_line = [0,0,0,1]
    bottom_line = [0,0,1,0]
    for x1,y1,x2,y2 in lines:
        # print x1, " ", y1, " ", x2, " ", y2
        if(helper.isVerticalLine(x1,y1,x2,y2)):
            # print "left line : ", leftmost_vertical_line, "\ncurrent line : ", [x1,y1,x2,y2] 
            leftmost_vertical_line = helper.resultingVerticalLeftLine(leftmost_vertical_line,[x1,y1,x2,y2])
            rightmost_vertical_line = helper.resultingVerticalRightLine(rightmost_vertical_line,[x1,y1,x2,y2])
        else:
            bottom_line = helper.resultingHorizontalBottomLine(bottom_line,[x1,y1,x2,y2])
            # print "left line : ", leftmost_vertical_line, "\n\n\n" 
        # cv2.line(src,(x1,y1),(x2,y2),np.array([0, 0, 255]),3)
        # cv2.line(src,(leftmost_vertical_line[0],leftmost_vertical_line[1]),(leftmost_vertical_line[2],leftmost_vertical_line[3]),np.array([0, 255, 0]),3)

        # cv2.imshow(window_name,src)
        # # wait for a key press, and exit if Esc is pressed
        # if cv2.waitKey() & 0xFF == 27 :
        #     break

    cv2.line(src,(leftmost_vertical_line[0],leftmost_vertical_line[1]),(leftmost_vertical_line[2],leftmost_vertical_line[3]),np.array([0, 255, 0]),3)
    cv2.line(src,(rightmost_vertical_line[0],rightmost_vertical_line[1]),(rightmost_vertical_line[2],rightmost_vertical_line[3]),np.array([0, 255, 0]),3)
    cv2.line(src,(bottom_line[0],bottom_line[1]),(bottom_line[2],bottom_line[3]),np.array([0, 255, 0]),3)
    cv2.imshow(window_name,src)
    # wait for a key press, and exit if Esc is pressed
    if cv2.waitKey() & 0xFF == 27 :
        break


cap.release()
cv2.destroyAllWindows()