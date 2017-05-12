import numpy as np
import cv2

def isVerticalLine(x1,y1,x2,y2):
	if(x1==x2 and y1==y2):
		print "\nWARNING : not a line, the two points are the same\n\n"
		return
	return(abs(y2-y1) > abs(x2-x1))

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

def lineEquation(x1,y1,x2,y2):
	if(isVerticalLine(x1,y1,x2,y2)):
		return verticalLineEquation(x1,y1,x2,y2)
	else:
		return horizontalLineEquation(x1,y1,x2,y2)

def verticalLineEquation(x1,y1,x2,y2):
	a = 1.
	b = -(float(x1)-float(x2))/(float(y1)-float(y2))
	c = -(float(x1)+float(x2)+b*(float(y1)+float(y2)))/2.

	return np.array([a,b,c])

def horizontalLineEquation(x1,y1,x2,y2):
	a = -(float(y1)-float(y2))/(float(x1)-float(x2))
	b = 1.
	c = -(float(y1)+float(y2)+a*(float(x1)+float(x2)))/2.

	return np.array([a,b,c])

def computeYOnLine(line_equation, x):
	a = line_equation[0]
	b = line_equation[1]
	c = line_equation[2]
	if(b == 0):
		print "\nWARNING : vertical line, impossible to compute y in computeYOnLine\n\n"
		return
	return (float) (-c-a*x)/b

def computeXOnLine(line_equation, y):
	a = line_equation[0]
	b = line_equation[1]
	c = line_equation[2]
	if(a == 0):
		print "\nWARNING : horizontal line, impossible to compute x in computeXOnLine\n\n"
		return
	return (float) (-c-b*y)/a