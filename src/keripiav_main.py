#!/usr/bin/env python
import numpy as np
import cv2
import keripiav_helper_functions as helper

# path to vidoe file
folder = '../data/individual_keys/'
file = 'C4.mp4'
path_to_vidoe_file = folder + file


# TODO : load calibration parameters


# create window
window_name = 'window'
cv2.namedWindow( window_name, cv2.WINDOW_NORMAL);
cv2.resizeWindow(window_name, 400, 550);

# create video capture object
cap = cv2.VideoCapture(path_to_vidoe_file)

if(not cap.isOpened()):
    print "error reading video\n"

# TODO : make manual
# crop zone. x is vertical, y is horizontal for numpy. reverse for opencv ?
cropped_origin = (500,50)
cropped_end = (1600,850)
cropped_size = (cropped_end[0]-cropped_origin[0],cropped_end[1]-cropped_origin[1])
cropped_height = cropped_size[0]
cropped_width = cropped_size[1]

# TODO : find a baseline


# read the frames
while(cap.isOpened()):
    # read the frame
    ret, frame = cap.read()
    frame = frame[cropped_origin[0]:cropped_end[0],cropped_origin[1]:cropped_end[1],:]

    # cv2.imshow(window_name,frame)
    # # wait for a key press, and exit if Esc is pressed
    # if cv2.waitKey() & 0xFF == 27 :
    #     break

    # converto to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # cv2.imshow(window_name,hsv_frame)
    # # wait for a key press, and exit if Esc is pressed
    # if cv2.waitKey() & 0xFF == 27 :
    #     break

    # to find the black pixels in the image, we look for low saturation and low value
    low_treshold_hsv_black = np.array([0,0,0])
    high_treshold_hsv_black = np.array([255,80,35])
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

    # close the image
    closing_kernel_size = 20;
    closing_kernel = np.ones((closing_kernel_size, closing_kernel_size))
    closed_image = cv2.morphologyEx(black_white_image, cv2.MORPH_CLOSE, closing_kernel)

    src = cv2.cvtColor(closed_image, cv2.COLOR_GRAY2BGR)

    # cv2.imshow(window_name,src)
    # # wait for a key press, and exit if Esc is pressed
    # if cv2.waitKey() & 0xFF == 27 :
    #     break

    # find the edges and lines
    edges = cv2.Canny(src, 2500, 2700, 0, 5)
    # cv2.imshow(window_name,edges)
    # # wait for a key press, and exit if Esc is pressed
    # if cv2.waitKey() & 0xFF == 27 :
    #     break

    nImtersectionPoints = 120
    minLineLength = 180
    maxLineGap = 35
    lines_tmp = cv2.HoughLinesP(edges,1,np.pi/180,nImtersectionPoints,0,minLineLength,maxLineGap)
    lines = np.squeeze(lines_tmp)
    left_line = [cropped_size[1],0,cropped_size[1],1]
    right_line = [0,0,0,1]
    bottom_line = [0,0,1,0]
    # find the left, right and bottom lines
    for x1,y1,x2,y2 in lines:
        # print x1, " ", y1, " ", x2, " ", y2
        if(helper.isVerticalLine(x1,y1,x2,y2)):
            # print "left line : ", left_line, "\ncurrent line : ", [x1,y1,x2,y2] 
            left_line = helper.resultingVerticalLeftLine(left_line,[x1,y1,x2,y2])
            right_line = helper.resultingVerticalRightLine(right_line,[x1,y1,x2,y2])
        else:
            bottom_line = helper.resultingHorizontalBottomLine(bottom_line,[x1,y1,x2,y2])

    # cv2.line(src,(left_line[0],left_line[1]),(left_line[2],left_line[3]),np.array([0, 255, 0]),3)
    # cv2.line(src,(right_line[0],right_line[1]),(right_line[2],right_line[3]),np.array([0, 255, 0]),3)
    # cv2.line(src,(bottom_line[0],bottom_line[1]),(bottom_line[2],bottom_line[3]),np.array([0, 255, 0]),3)

    # cv2.imshow(window_name,src)
    # # wait for a key press, and exit if Esc is pressed
    # if cv2.waitKey() & 0xFF == 27 :
    #     break

    # remove everything outside of the keyboard
    poly1 = helper.findLeftPolygon(left_line, cropped_height, cropped_width)
    poly2 = helper.findRightPolygon(right_line, cropped_height, cropped_width)
    poly3 = helper.findBottomPolygon(bottom_line, cropped_height, cropped_width)

    cv2.fillConvexPoly(src, poly1, np.array([0,0,0]))    
    cv2.fillConvexPoly(src, poly2, np.array([0,0,0]))    
    cv2.fillConvexPoly(src, poly3, np.array([0,0,0]))    
    cv2.imshow(window_name,src)
    # wait for a key press, and exit if Esc is pressed
    if cv2.waitKey() & 0xFF == 27 :
        break

    
    # find the edges and lines again
    edges = cv2.Canny(src, 2500, 2700, 0, 5)
    # cv2.imshow(window_name,edges)
    # # wait for a key press, and exit if Esc is pressed
    # if cv2.waitKey() & 0xFF == 27 :
    #     break

    nImtersectionPoints = 50
    minLineLength = 20
    maxLineGap = 75
    lines_tmp = cv2.HoughLinesP(edges,1,np.pi/180,nImtersectionPoints,0,minLineLength,maxLineGap)
    lines = np.squeeze(lines_tmp)
    top_line = [cropped_width-100 ,cropped_height, cropped_width, cropped_height]
    # find the left, right and bottom lines
    for x1,y1,x2,y2 in lines:
        if(not helper.isVerticalLine(x1,y1,x2,y2)):
            top_line = helper.resultingHorizontalTopLine(top_line,[x1,y1,x2,y2])

    # cv2.line(src,(top_line[0],top_line[1]),(top_line[2],top_line[3]),np.array([0, 255, 0]),3)

    # cv2.imshow(window_name,src)
    # # wait for a key press, and exit if Esc is pressed
    # if cv2.waitKey() & 0xFF == 27 :
    #     break

    # find the corners of the keyboard 
    # TODO : filter these corners
    corners = helper.findCorners(top_line,bottom_line,left_line,right_line)

    for point in corners:
        cv2.circle(frame, (point[0],point[1]), 15, np.array([0,0,255]), 5)

    cv2.imshow(window_name,frame)
    # wait for a key press, and exit if Esc is pressed
    if cv2.waitKey() & 0xFF == 27 :
        break


cap.release()
cv2.destroyAllWindows()