#!/usr/bin/env python
import numpy as np
import cv2
import keripiav_helper_functions as helper
import Filter

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

# TODO : find a baseline online
for i in range(20):
    ret, frame = cap.read()

baseline = frame[cropped_origin[0]:cropped_end[0],cropped_origin[1]:cropped_end[1],:]
baseline_copy = baseline.copy()
cv2.imshow(window_name,baseline)
cv2.waitKey()
hsv_baseline = cv2.cvtColor(baseline_copy, cv2.COLOR_BGR2HSV)

# to find the black pixels in the image, we look for low saturation and low value
low_treshold_hsv_black = np.array([0,0,0])
high_treshold_hsv_black = np.array([255,80,35])
black_pixels = cv2.inRange(hsv_baseline, low_treshold_hsv_black, high_treshold_hsv_black)
# to find the white pixels, we look for low saturation and high value
low_treshold_hsv_white = np.array([0,0,200])
high_treshold_hsv_white = np.array([255,35,255])
white_pixels = cv2.inRange(hsv_baseline, low_treshold_hsv_white, high_treshold_hsv_white)

# image composed of the white and black pixels
black_white_image = black_pixels + white_pixels

# close the image
closing_kernel_size = 20;
closing_kernel = np.ones((closing_kernel_size, closing_kernel_size))
closed_image = cv2.morphologyEx(black_white_image, cv2.MORPH_CLOSE, closing_kernel)

src = cv2.cvtColor(closed_image, cv2.COLOR_GRAY2BGR)

# find the edges and lines
edges = cv2.Canny(src, 2500, 2700, 0, 5)

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

# remove everything outside of the keyboard
polyleft = helper.findLeftPolygon(left_line, cropped_height, cropped_width)
polyright = helper.findRightPolygon(right_line, cropped_height, cropped_width)
polybottom = helper.findBottomPolygon(bottom_line, cropped_height, cropped_width)

cv2.fillConvexPoly(src, polyleft, np.array([0,0,0]))    
cv2.fillConvexPoly(src, polyright, np.array([0,0,0]))    
cv2.fillConvexPoly(src, polybottom, np.array([0,0,0]))    

# find the edges and lines again
edges = cv2.Canny(src, 2500, 2700, 0, 5)

nImtersectionPoints = 45
minLineLength = 10
maxLineGap = 75
lines_tmp = cv2.HoughLinesP(edges,1,np.pi/180,nImtersectionPoints,0,minLineLength,maxLineGap)
lines = np.squeeze(lines_tmp)
top_line = [cropped_width-100 ,cropped_height, cropped_width, cropped_height]
# find the left, right and bottom lines
for x1,y1,x2,y2 in lines:
    if(not helper.isVerticalLine(x1,y1,x2,y2)):
        top_line = helper.resultingHorizontalTopLine(top_line,[x1,y1,x2,y2])

polytop = helper.findTopPolygon(top_line, cropped_height, cropped_width)

corners = helper.findCorners(top_line,bottom_line,left_line,right_line)
for point in corners:
    cv2.circle(baseline_copy, (point[0],point[1]), 15, np.array([0,0,255]), 5)

cv2.imshow(window_name,baseline_copy)
# wait for a key press, and exit if Esc is pressed
cv2.waitKey() 

print "corners : ", corners

fp = Filter.ProximityFilter(corners, 100)
fb = Filter.ButterworthFilter(corners,0.1,2)

# read the frames
while(cap.isOpened()):
    # read the frame
    ret, frame = cap.read()
    frame = frame[cropped_origin[0]:cropped_end[0],cropped_origin[1]:cropped_end[1],:]
    cframe = frame.copy()

    cv2.imshow(window_name,frame)
    # wait for a key press, and exit if Esc is pressed
    if cv2.waitKey() & 0xFF == 27 :
        break

    # converto to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow(window_name,hsv_frame)
    # wait for a key press, and exit if Esc is pressed
    if cv2.waitKey() & 0xFF == 27 :
        break

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

    cv2.imshow(window_name,black_white_image)
    # wait for a key press, and exit if Esc is pressed
    if cv2.waitKey() & 0xFF == 27 :
        break

    # close the image
    closing_kernel_size = 20;
    closing_kernel = np.ones((closing_kernel_size, closing_kernel_size))
    closed_image = cv2.morphologyEx(black_white_image, cv2.MORPH_CLOSE, closing_kernel)

    src = cv2.cvtColor(closed_image, cv2.COLOR_GRAY2BGR)

    cv2.imshow(window_name,src)
    # wait for a key press, and exit if Esc is pressed
    if cv2.waitKey() & 0xFF == 27 :
        break

    # find the edges and lines
    edges = cv2.Canny(src, 2500, 2700, 0, 5)
    cv2.imshow(window_name,edges)
    # wait for a key press, and exit if Esc is pressed
    if cv2.waitKey() & 0xFF == 27 :
        break

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

    cv2.line(src,(left_line[0],left_line[1]),(left_line[2],left_line[3]),np.array([0, 255, 0]),3)
    cv2.line(src,(right_line[0],right_line[1]),(right_line[2],right_line[3]),np.array([0, 255, 0]),3)
    cv2.line(src,(bottom_line[0],bottom_line[1]),(bottom_line[2],bottom_line[3]),np.array([0, 255, 0]),3)

    cv2.imshow(window_name,src)
    # wait for a key press, and exit if Esc is pressed
    if cv2.waitKey() & 0xFF == 27 :
        break

    # remove everything outside of the keyboard
    polyleft = helper.findLeftPolygon(left_line, cropped_height, cropped_width)
    polyright = helper.findRightPolygon(right_line, cropped_height, cropped_width)
    polybottom = helper.findBottomPolygon(bottom_line, cropped_height, cropped_width)

    cv2.fillConvexPoly(src, polyleft, np.array([0,0,0]))    
    cv2.fillConvexPoly(src, polyright, np.array([0,0,0]))    
    cv2.fillConvexPoly(src, polybottom, np.array([0,0,0]))    
    cv2.imshow(window_name,src)
    # wait for a key press, and exit if Esc is pressed
    if cv2.waitKey() & 0xFF == 27 :
        break

    
    # find the edges and lines again
    edges = cv2.Canny(src, 2500, 2700, 0, 5)
    cv2.imshow(window_name,edges)
    # wait for a key press, and exit if Esc is pressed
    if cv2.waitKey() & 0xFF == 27 :
        break

    nImtersectionPoints = 45
    minLineLength = 10
    maxLineGap = 75
    lines_tmp = cv2.HoughLinesP(edges,1,np.pi/180,nImtersectionPoints,0,minLineLength,maxLineGap)
    lines = np.squeeze(lines_tmp)
    top_line = [cropped_width-100 ,cropped_height, cropped_width, cropped_height]
    # find the left, right and bottom lines
    for x1,y1,x2,y2 in lines:
        if(not helper.isVerticalLine(x1,y1,x2,y2)):
            top_line = helper.resultingHorizontalTopLine(top_line,[x1,y1,x2,y2])

    polytop = helper.findTopPolygon(top_line, cropped_height, cropped_width)
    cv2.line(src,(top_line[0],top_line[1]),(top_line[2],top_line[3]),np.array([0, 255, 0]),3)

    cv2.imshow(window_name,src)
    # wait for a key press, and exit if Esc is pressed
    if cv2.waitKey() & 0xFF == 27 :
        break

    # find the corners of the keyboard 
    # TODO : filter these corners
    corners = helper.findCorners(top_line,bottom_line,left_line,right_line)
    # print "corners : ", corners
    tmp = fp.updateFilter(corners)
    # print "tmp : ", tmp
    corners = fb.updateFilter(tmp)
    # print "filtered corners : ", corners

    for point in corners:
        cv2.circle(frame, (int(point[0]),int(point[1])), 15, np.array([0,0,255]), 5)

    cv2.imshow(window_name,frame)
    # wait for a key press, and exit if Esc is pressed
    if cv2.waitKey() & 0xFF == 27 :
        break
    
    # Make image differences in hsv space
    # pos_diff = cv2.absdiff(baseline,cframe)
    pos_diff = helper.posDiff(baseline,cframe)
    neg_diff = helper.posDiff(-baseline,-cframe)
    # neg_diff = cv2.absdiff(-cframe,-baseline)
    
    cv2.imshow(window_name,pos_diff)
    if cv2.waitKey() & 0xFF == 27 :
        break

    pos_diff_treshold_low = np.array([0,0,120])
    pos_diff_treshold_high = np.array([255,255,255])
    neg_diff_treshold_low = np.array([0,0,200])
    neg_diff_treshold_high = np.array([255,20,255])
    pos_diff = cv2.cvtColor(pos_diff, cv2.COLOR_BGR2HSV)
    pos_diff = cv2.inRange(pos_diff, pos_diff_treshold_low, pos_diff_treshold_high)
    neg_diff = cv2.cvtColor(neg_diff, cv2.COLOR_BGR2HSV)
    neg_diff = cv2.inRange(neg_diff, neg_diff_treshold_low, neg_diff_treshold_high)

    cv2.fillConvexPoly(pos_diff, polyleft, np.array([0,0,0])) 
    cv2.fillConvexPoly(pos_diff, polytop, np.array([0,0,0])) 
    cv2.fillConvexPoly(neg_diff, polyleft, np.array([0,0,0])) 
    cv2.fillConvexPoly(neg_diff, polytop, np.array([0,0,0])) 

    if(np.max(pos_diff) > 0):
        print "positive difference"
        cv2.imshow(window_name,pos_diff)
        if cv2.waitKey() & 0xFF == 27 :
            break

    white_pixels = np.argwhere(pos_diff)

    # if(np.max(neg_diff) > 0):
    #     print "negative difference"
    #     cv2.imshow(window_name,neg_diff)
    #     if cv2.waitKey() & 0xFF == 27 :
    #         break
    


cap.release()
cv2.destroyAllWindows()