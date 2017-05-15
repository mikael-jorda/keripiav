import numpy as np
import cv2

# retuens true if the line is globally vertical (slope more than 1)
def isVerticalLine(x1,y1,x2,y2):
	if(x1==x2 and y1==y2):
		print "\nWARNING : not a line, the two points are the same\n\n"
		return
	return(abs(y2-y1) > abs(x2-x1))

def isVerticalLineArray(l):
	return isVerticalLine(l[0],l[1],l[2],l[3])

# compute the line formed by the two leftmost points of the ends of the two lines given as arguments
def resultingVerticalLeftLine(l1,l2):
	# makes a left line out of the two left most points
	if(not (isVerticalLine(l1[0],l1[1],l1[2],l1[3]) and isVerticalLine(l2[0],l2[1],l2[2],l2[3]))):
		print "\nWARNING : Not vertical lines in resultingVerticalLeftLine\n\n"
		return
	eq1 = verticalLineEquation(l1[0],l1[1],l1[2],l1[3])
	eq2 = verticalLineEquation(l2[0],l2[1],l2[2],l2[3])
	# order the points of the first line
	if(l1[1] < l1[3]):
		top_point_1 = (l1[0], l1[1])
		bottom_point_1 = (l1[2],l1[3])
	else:
		top_point_1 = (l1[2], l1[3])
		bottom_point_1 = (l1[0],l1[1])
	# order the points of the second line
	if(l2[1] < l2[3]):
		top_point_2 = (l2[0], l2[1])
		bottom_point_2 = (l2[2],l2[3])
	else:
		top_point_2 = (l2[2], l2[3])
		bottom_point_2 = (l2[0],l2[1])

	# find the resulting top point
	if(top_point_1[1] < top_point_2[1]):
		tmpx = computeXOnLine(eq2, top_point_1[1])
		if(tmpx < top_point_1[0]):
			resulting_top = (tmpx, top_point_1[1])
		else:
			resulting_top = top_point_1
	else:
		tmpx = computeXOnLine(eq1, top_point_2[1])
		if(tmpx < top_point_2[0]):
			resulting_top = (tmpx, top_point_2[1])
		else:
			resulting_top = top_point_2

	# find the resulting bottom point
	if(bottom_point_1[1] > bottom_point_2[1]):
		tmpx = computeXOnLine(eq2, bottom_point_1[1])
		if(tmpx < bottom_point_1[0]):
			resulting_bottom = (tmpx, bottom_point_1[1])
		else:
			resulting_bottom = bottom_point_1
	else:
		tmpx = computeXOnLine(eq1, bottom_point_2[1])
		if(tmpx < bottom_point_2[0]):
			resulting_bottom = (tmpx, bottom_point_2[1])
		else:
			resulting_bottom = bottom_point_2

	return [int(resulting_top[0]),int(resulting_top[1]),int(resulting_bottom[0]),int(resulting_bottom[1])]

# compute the line formed by the two rightmost points of the ends of the two lines given as arguments
def resultingVerticalRightLine(l1,l2):
	# makes a left line out of the two left most points
	if(not (isVerticalLine(l1[0],l1[1],l1[2],l1[3]) and isVerticalLine(l2[0],l2[1],l2[2],l2[3]))):
		print "\nWARNING : Not vertical lines in resultingVerticalRightLine\n\n"
		return
	eq1 = verticalLineEquation(l1[0],l1[1],l1[2],l1[3])
	eq2 = verticalLineEquation(l2[0],l2[1],l2[2],l2[3])
	# order the points of the first line
	if(l1[1] < l1[3]):
		top_point_1 = (l1[0], l1[1])
		bottom_point_1 = (l1[2],l1[3])
	else:
		top_point_1 = (l1[2], l1[3])
		bottom_point_1 = (l1[0],l1[1])
	# order the points of the second line
	if(l2[1] < l2[3]):
		top_point_2 = (l2[0], l2[1])
		bottom_point_2 = (l2[2],l2[3])
	else:
		top_point_2 = (l2[2], l2[3])
		bottom_point_2 = (l2[0],l2[1])

	# find the resulting top point
	if(top_point_1[1] < top_point_2[1]):
		tmpx = computeXOnLine(eq2, top_point_1[1])
		if(tmpx > top_point_1[0]):
			resulting_top = (tmpx, top_point_1[1])
		else:
			resulting_top = top_point_1
	else:
		tmpx = computeXOnLine(eq1, top_point_2[1])
		if(tmpx > top_point_2[0]):
			resulting_top = (tmpx, top_point_2[1])
		else:
			resulting_top = top_point_2

	# find the resulting bottom point
	if(bottom_point_1[1] > bottom_point_2[1]):
		tmpx = computeXOnLine(eq2, bottom_point_1[1])
		if(tmpx > bottom_point_1[0]):
			resulting_bottom = (tmpx, bottom_point_1[1])
		else:
			resulting_bottom = bottom_point_1
	else:
		tmpx = computeXOnLine(eq1, bottom_point_2[1])
		if(tmpx > bottom_point_2[0]):
			resulting_bottom = (tmpx, bottom_point_2[1])
		else:
			resulting_bottom = bottom_point_2

	return [int(resulting_top[0]),int(resulting_top[1]),int(resulting_bottom[0]),int(resulting_bottom[1])]

# compute the line formed by the two bottommost points of the ends of the two lines given as arguments
def resultingHorizontalBottomLine(l1,l2):
	# makes a left line out of the two left most points
	if(isVerticalLine(l1[0],l1[1],l1[2],l1[3]) or isVerticalLine(l2[0],l2[1],l2[2],l2[3])):
		print "\nWARNING : Not horizontal lines in resultingHorinzontalBottomLine\n\n"
		return
	eq1 = horizontalLineEquation(l1[0],l1[1],l1[2],l1[3])
	eq2 = horizontalLineEquation(l2[0],l2[1],l2[2],l2[3])
	# order the points of the first line
	if(l1[0] < l1[2]):
		left_point_1 = (l1[0], l1[1])
		right_point_1 = (l1[2],l1[3])
	else:
		left_point_1 = (l1[2], l1[3])
		right_point_1 = (l1[0],l1[1])
	# order the points of the second line
	if(l2[0] < l2[2]):
		left_point_2 = (l2[0], l2[1])
		right_point_2 = (l2[2],l2[3])
	else:
		left_point_2 = (l2[2], l2[3])
		right_point_2 = (l2[0],l2[1])

	# find the resulting left point
	if(left_point_1[0] < left_point_2[0]):
		tmpy = computeYOnLine(eq2, left_point_1[0])
		if(tmpy > left_point_1[1]):
			resulting_left = (left_point_1[0], tmpy)
		else:
			resulting_left = left_point_1
	else:
		tmpy = computeYOnLine(eq1, left_point_2[0])
		if(tmpy > left_point_2[1]):
			resulting_left = (left_point_2[0],tmpy)
		else:
			resulting_left = left_point_2

	# find the resulting right point
	if(right_point_1[0] > right_point_2[0]):
		tmpy = computeYOnLine(eq2, right_point_1[0])
		if(tmpy > right_point_1[1]):
			resulting_right = (right_point_1[0], tmpy)
		else:
			resulting_right = right_point_1
	else:
		tmpy = computeYOnLine(eq1, right_point_2[0])
		if(tmpy > right_point_2[1]):
			resulting_right = (right_point_2[0], tmpy)
		else:
			resulting_right = right_point_2

	return [int(resulting_left[0]),int(resulting_left[1]),int(resulting_right[0]),int(resulting_right[1])]

# compute the line formed by the two bottommost points of the ends of the two lines given as arguments
def resultingHorizontalTopLine(l1,l2):
	# makes a left line out of the two left most points
	if(isVerticalLine(l1[0],l1[1],l1[2],l1[3]) or isVerticalLine(l2[0],l2[1],l2[2],l2[3])):
		print "\nWARNING : Not horizontal lines in resultingHorinzontalTopLine\n\n"
		return
	eq1 = horizontalLineEquation(l1[0],l1[1],l1[2],l1[3])
	eq2 = horizontalLineEquation(l2[0],l2[1],l2[2],l2[3])
	# order the points of the first line
	if(l1[0] < l1[2]):
		left_point_1 = (l1[0], l1[1])
		right_point_1 = (l1[2],l1[3])
	else:
		left_point_1 = (l1[2], l1[3])
		right_point_1 = (l1[0],l1[1])
	# order the points of the second line
	if(l2[0] < l2[2]):
		left_point_2 = (l2[0], l2[1])
		right_point_2 = (l2[2],l2[3])
	else:
		left_point_2 = (l2[2], l2[3])
		right_point_2 = (l2[0],l2[1])

	# find the resulting left point
	if(left_point_1[0] < left_point_2[0]):
		tmpy = computeYOnLine(eq2, left_point_1[0])
		if(tmpy < left_point_1[1]):
			resulting_left = (left_point_1[0], tmpy)
		else:
			resulting_left = left_point_1
	else:
		tmpy = computeYOnLine(eq1, left_point_2[0])
		if(tmpy < left_point_2[1]):
			resulting_left = (left_point_2[0],tmpy)
		else:
			resulting_left = left_point_2

	# find the resulting right point
	if(right_point_1[0] > right_point_2[0]):
		tmpy = computeYOnLine(eq2, right_point_1[0])
		if(tmpy < right_point_1[1]):
			resulting_right = (right_point_1[0], tmpy)
		else:
			resulting_right = right_point_1
	else:
		tmpy = computeYOnLine(eq1, right_point_2[0])
		if(tmpy < right_point_2[1]):
			resulting_right = (right_point_2[0], tmpy)
		else:
			resulting_right = right_point_2

	return [int(resulting_left[0]),int(resulting_left[1]),int(resulting_right[0]),int(resulting_right[1])]

# return the line equation 
def lineEquation(x1,y1,x2,y2):
	if(isVerticalLine(x1,y1,x2,y2)):
		return verticalLineEquation(x1,y1,x2,y2)
	else:
		return horizontalLineEquation(x1,y1,x2,y2)

# return the line equation when the inpus is an array such that l = [x1,y1,x2,y2] 
def lineEquationArray(l):
	return lineEquation(l[0],l[1],l[2],l[3])

# return the equation of a vertical line
def verticalLineEquation(x1,y1,x2,y2):
	a = 1.
	b = -(float(x1)-float(x2))/(float(y1)-float(y2))
	c = -(float(x1)+float(x2)+b*(float(y1)+float(y2)))/2.

	return np.array([a,b,c])

# return the equation of a horizontal line
def horizontalLineEquation(x1,y1,x2,y2):
	a = -(float(y1)-float(y2))/(float(x1)-float(x2))
	b = 1.
	c = -(float(y1)+float(y2)+a*(float(x1)+float(x2)))/2.

	return np.array([a,b,c])

# computes the y coordinate corresponding to the given line and x
def computeYOnLine(line_equation, x):
	a = line_equation[0]
	b = line_equation[1]
	c = line_equation[2]
	if(b == 0):
		print "\nWARNING : vertical line, impossible to compute y in computeYOnLine\n\n"
		return
	return (float) (-c-a*x)/b

# computes the x coordinate corresponding to the given line and y
def computeXOnLine(line_equation, y):
	a = line_equation[0]
	b = line_equation[1]
	c = line_equation[2]
	if(a == 0):
		print "\nWARNING : horizontal line, impossible to compute x in computeXOnLine\n\n"
		return
	return (float) (-c-b*y)/a

# finds the polygon to the left of the vertical line given as argument in the image
def findLeftPolygon(line, img_height, img_width):
	if(not isVerticalLineArray(line)):
		print "\nWARNING : line not vertical in findLeftPolygon\n\n"
	eq = lineEquationArray(line)
	xtop = computeXOnLine(eq,0)
	xbottom = computeXOnLine(eq, img_height)
	p1 = np.array([xtop,0], np.int32)
	p2 = np.array([xbottom,img_height], np.int32)
	if(xtop > 0):
		p0 = np.array([0,0], np.int32)
		if(xbottom > 0):
			p3 = np.array([0,img_height], np.int32)
			return np.array([p0,p1,p2,p3])
		return np.array([p0,p1,p2])
	else:
		if(xbottom > 0):
			p3 = np.array([0,img_height])
			return np.array([p1,p2,p3]) 
		else:
			print "\nWARNING : should not reach here in findLeftPolygon\n\n"

# finds the polygon to the right of the vertical line given as argument in the image
def findRightPolygon(line, img_height, img_width):
	if(not isVerticalLineArray(line)):
		print "\nWARNING : line not vertical in findRightPolygon\n\n"
	eq = lineEquationArray(line)
	xtop = computeXOnLine(eq,0)
	xbottom = computeXOnLine(eq, img_height)
	p1 = np.array([xtop,0], np.int32)
	p2 = np.array([xbottom,img_height], np.int32)
	if(xtop < img_width):
		p0 = np.array([img_width,0], np.int32)
		if(xbottom < img_width):
			p3 = np.array([img_width,img_height], np.int32)
			return np.array([p0,p1,p2,p3])
		return np.array([p0,p1,p2])
	else:
		if(xbottom < img_width):
			p3 = np.array([img_width,img_height])
			return np.array([p1,p2,p3]) 
		else:
			print "\nWARNING : should not reach here in findRightPolygon\n\n"

def findBottomPolygon(line, img_height, img_width):
	if(isVerticalLineArray(line)):
		print "\nWARNING : line not horizontal in findBottomPolygon\n\n"
	eq = lineEquationArray(line)
	yleft = computeYOnLine(eq,0)
	yright = computeYOnLine(eq, img_width)
	p1 = np.array([0,yleft], np.int32)
	p2 = np.array([img_width, yright], np.int32)
	if(yleft < img_height):
		p0 = np.array([0,img_height], np.int32)
		if(yright < img_height):
			p3 = np.array([img_width,img_height], np.int32)
			return np.array([p0,p1,p2,p3])
		return np.array([p0,p1,p2])
	else:
		if(yright < img_height):
			p3 = np.array([img_width,img_height])
			return np.array([p1,p2,p3]) 
		else:
			print "\nWARNING : should not reach here in findBottomPolygon\n\n"

def findTopPolygon(line, img_height, img_width):
	if(isVerticalLineArray(line)):
		print "\nWARNING : line not horizontal in findTopPolygon\n\n"
	eq = lineEquationArray(line)
	yleft = computeYOnLine(eq,0)
	yright = computeYOnLine(eq, img_width)
	p1 = np.array([0,yleft], np.int32)
	p2 = np.array([img_width, yright], np.int32)
	if(yleft > 0):
		p0 = np.array([0,0], np.int32)
		if(yright > 0):
			p3 = np.array([img_width,0], np.int32)
			return np.array([p0,p1,p2,p3])
		return np.array([p0,p1,p2])
	else:
		if(yright > 0):
			p3 = np.array([img_width,0])
			return np.array([p1,p2,p3]) 
		else:
			print "\nWARNING : should not reach here in findTopPolygon\n\n"

# find the corners of a polygon defined by 4 intersecting lines
def findCorners(tl,bl,ll,rl):
	eqt = lineEquationArray(tl)
	eqb = lineEquationArray(bl)
	eql = lineEquationArray(ll)
	eqr = lineEquationArray(rl)

	tlp = np.cross(eqt,eql)
	p1 = np.array([tlp[0]/tlp[2],tlp[1]/tlp[2]], np.int32)

	trp = np.cross(eqt,eqr)
	p2 = np.array([trp[0]/trp[2],trp[1]/trp[2]], np.int32)

	blp = np.cross(eqb,eql)
	p3 = np.array([blp[0]/blp[2],blp[1]/blp[2]], np.int32)

	brp = np.cross(eqb,eqr)
	p4 = np.array([brp[0]/brp[2],brp[1]/brp[2]], np.int32)

	return np.array([p1,p2,p3,p4])