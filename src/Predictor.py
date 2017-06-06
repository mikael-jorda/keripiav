#!/usr/bin/env python
import numpy as np
import cv2
import keripiav_helper_functions as helper
import project_keyboard
import corners
from project_keyboard import imshow, homogenize, dehomogenize
import Filter
from collections import Counter

# predictor class that, given a video, is able to predict which key is played in the frames.
# It has a baseline to which we compare the images, it is able to compute the transform between the position 
# of the keyboard in the current image and the position of the virtual keyboard, and, given this trnsformation,
# to predict the key that is played in the image.
class Predictor:

	# initialize the predictor class.
	# 
	# Parameters 
	# 	camera_side : the side of the camera (left or right)
	# 	video_file : the path to the video file
	def __init__(self, video_file, camera_side=None):

		# define all the parameters for the different cameras
		if camera_side == 'right':
			self.pos_camera = 'right'
			# threshold to detect the white and black pixels in hsv space
			self.low_treshold_hsv_white = np.array([0,0,200])
			self.high_treshold_hsv_white = np.array([255,35,255])
			self.low_treshold_hsv_black = np.array([0,0,0])
			self.high_treshold_hsv_black = np.array([255,80,35])
			# kernel size for opening operation. The higher, the more aggressive if the closing
			self.opening_kernel_size = 10;
			# kernel size for closing operation. The higher, the more aggressive if the closing
			self.closing_kernel_size = 20;
			# parameters for first Hough line detection
			self.nIntersectionPointsFirst = 120
			self.minLineLengthFirst = 180
			self.maxLineGapFirst = 35
			# parameters for second Hough line detection
			self.nImtersectionPointsSecond = 45
			self.minLineLengthSecond = 10
			self.maxLineGapSecond = 75
			
		elif camera_side == 'left':
			self.pos_camera = 'left'
			# threshold to detect the white and black pixels in hsv space
			self.low_treshold_hsv_white = np.array([0,0,200])
			self.high_treshold_hsv_white = np.array([255,40,255])
			self.low_treshold_hsv_black = np.array([10,10,10])
			self.high_treshold_hsv_black = np.array([100,30,150])
			# kernel size for opening operation. The higher, the more aggressive if the closing
			self.opening_kernel_size = 10;
			# kernel size for closing operation. The higher, the more aggressive if the closing
			self.closing_kernel_size = 5;
			# parameters for first Hough line detection
			self.nIntersectionPointsFirst = 70
			self.minLineLengthFirst = 130
			self.maxLineGapFirst = 55
			# parameters for second Hough line detection
			self.nImtersectionPointsSecond = 45
			self.minLineLengthSecond = 10
			self.maxLineGapSecond = 75

		# TODO: remove - camera side can be determined by findCorners
		# else:
		#     print 'please chose \'left\' or \'right\' for the camera_side parameter\n'
		#     quit()

		self.video_stream = cv2.VideoCapture(path_to_video_file)
		if(not self.video_stream.isOpened()):
			print "error reading video\n"
			quit()

		# ret, self.frame = self.video_stream.read()

		# self.getCroppedZone(frame)
		# self.baseline = frame[self.cropped_origin[0]:self.cropped_end[0], self.cropped_origin[1]:self.cropped_end[1]]
		self.pos_camera = None
		self.advanceFrame(update_projection=True)
		self.baseline = self.frame
		self.baseline_hsv = cv2.cvtColor(self.baseline, cv2.COLOR_BGR2HSV)

		self.window_name = 'Predictor_' + self.pos_camera
		cv2.namedWindow( self.window_name, cv2.WINDOW_NORMAL);
		cv2.resizeWindow(self.window_name, 400, 550);

		imshow(self.frame_marked, scale_down=3, window=self.window_name+"_baseline")

		self.fp = Filter.ProximityFilter(self.corners, 100)
		self.fb = Filter.ButterworthFilter(self.corners,0.1,2)

		# TODO: tune diff thresholds
		self.pos_diff_threshold_low = np.array([0,0,120])
		self.pos_diff_threshold_high = np.array([255,255,255])
		self.neg_diff_threshold_low = np.array([0,0,120])
		self.neg_diff_threshold_high = np.array([255,255,255])
	
	def advanceFrame(self, update_projection=False, project_image=True):
		ret, self.frame = self.video_stream.read()
		if not ret:
			return False
		self.frame_marked = self.frame.copy()

		if update_projection:
			# Find corners
			self.corners, self.pos_camera = corners.find_corners(self.frame_marked, self.pos_camera, mark_img=True, show_img=False)
			# TODO: filter corners

			# Find projection matrix and update key map/mask
			self.T_img_to_virtual, self.T_virtual_to_img = project_keyboard.find_projection(self.corners)
			self.key_map = project_keyboard.key_map(self.frame.shape, self.T_virtual_to_img, self.pos_camera)
			self.key_mask = (self.key_map > 0)[:,:,np.newaxis]

		# Project virtual image (only for visualization)
		if project_image:
			self.img_virtual = project_keyboard.project_image(self.frame, self.key_mask, self.T_img_to_virtual)

		return True

	def countKeyDiffs(self, show_img=False):
		# Use HSV
		frame_hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

		# Find differences
		pos_diff = (self.baseline_hsv.astype(np.int32) - frame_hsv.astype(np.int32)) * self.key_mask
		neg_diff = (frame_hsv.astype(np.int32) - self.baseline_hsv.astype(np.int32)) * self.key_mask
		pos_diff_v = pos_diff[:,:,2]
		neg_diff_v = neg_diff[:,:,2]

		# Clip negative differences in V channel
		pos_diff_v[pos_diff_v < 0] = 0
		neg_diff_v[neg_diff_v < 0] = 0

		# Take abs of negative differences in H, S channels
		pos_diff = np.abs(pos_diff).astype(np.uint8)
		neg_diff = np.abs(neg_diff).astype(np.uint8)
		pos_diff = cv2.inRange(pos_diff, self.pos_diff_threshold_low, self.pos_diff_threshold_high)
		neg_diff = cv2.inRange(neg_diff, self.neg_diff_threshold_low, self.neg_diff_threshold_high)

		# Find pixel coordinates
		pixels_pos_diff = np.argwhere(pos_diff)[:,::-1]
		pixels_neg_diff = np.argwhere(neg_diff)[:,::-1]

		# Count detected keys for diff pixels
		counts = Counter()

		for pixel in pixels_pos_diff:
			key = project_keyboard.key_label(self.key_map, pixel)
			if key is None or project_keyboard.is_black(key):
				continue
			counts[key] += 1

			# Plot positive differences
			if show_img:
				for i in range(pixels_pos_diff.shape[0]):
					cv2.circle(self.frame_marked, tuple(pixels_pos_diff[i,:2].astype(np.int32)), 1, (0,255,0), 3)
				pixels_pos_diff = homogenize(pixels_pos_diff)
				pixels_pos_diff_virtual = dehomogenize(pixels_pos_diff.dot(self.T_img_to_virtual.T))[:,::-1]
				for i in range(pixels_pos_diff_virtual.shape[0]):
					cv2.circle(self.img_virtual, tuple(pixels_pos_diff_virtual[i,:2].astype(np.int32)), 1, (0,255,0), 3)

		for pixel in pixels_neg_diff:
			key = project_keyboard.key_label(self.key_map, pixel)
			if key is None or project_keyboard.is_white(key):
				continue
			counts[key] += 1

			# Plot negative differences
			if show_img:
				for i in range(pixels_neg_diff.shape[0]):
					cv2.circle(self.frame_marked, tuple(pixels_neg_diff[i,:2].astype(np.int32)), 1, (0,0,255), 3)
				pixels_neg_diff = homogenize(pixels_neg_diff)
				pixels_neg_diff_virtual = dehomogenize(pixels_neg_diff.dot(self.T_img_to_virtual.T))[:,::-1]
				for i in range(pixels_neg_diff_virtual.shape[0]):
					cv2.circle(self.img_virtual, tuple(pixels_neg_diff_virtual[i,:2].astype(np.int32)), 1, (0,0,255), 3)

		if show_img and counts:
			imshow(self.frame_marked, scale_down=3, window=self.window_name, wait=1)
			imshow(self.img_virtual, window=self.window_name+"_virtual", wait=1)

		return counts

	# manually select the zone of interest in the image
	# TODO for manual selection on the image
	def getCroppedZone(self, frame, show_image = False):
		if self.side == 'right':
			self.cropped_origin = (500,50)
			self.cropped_end = (1600,850)
			self.cropped_size = (self.cropped_end[0]-self.cropped_origin[0],self.cropped_end[1]-self.cropped_origin[1])
			self.cropped_height = self.cropped_size[0]
			self.cropped_width = self.cropped_size[1]
		elif self.side == 'left':
			self.cropped_origin = (550,0)
			self.cropped_end = (1900,1000)
			self.cropped_size = (self.cropped_end[0]-self.cropped_origin[0],self.cropped_end[1]-self.cropped_origin[1])
			self.cropped_height = self.cropped_size[0]
			self.cropped_width = self.cropped_size[1]
		if show_image:
			cv2.imshow(self.window_name,frame[self.cropped_origin[0]:self.cropped_end[0], self.cropped_origin[1]:self.cropped_end[1]])
			# wait for a key press, and exit if Esc is pressed
			if cv2.waitKey() & 0xFF == 27 :
				quit()

	# update the baseline
	def updateBaseline(self, new_baseline, show_image = False):
		self.baseline = new_baseline[self.cropped_zone_x, self.cropped_zone_y]
		if show_image:
			cv2.imshow(self.window_name,self.baseline)
			if cv2.waitKey() & 0xFF == 27 :
				quit()

	# given a frame, find the corners of the keyboard
	def findCorners(self, frame, show_image = False, show_all_subimages = False):

		if show_all_subimages:
			cv2.imshow(self.window_name,frame)
			if cv2.waitKey() & 0xFF == 27 :
				quit()

		# 1. transform to HSV color space
		hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		if show_all_subimages:
			cv2.imshow(self.window_name,hsv_frame)
			if cv2.waitKey() & 0xFF == 27 :
				quit()

		# 2. binary treshold to keep only the black and white values
		# to find the black pixels in the image, we look for low saturation and low value
		black_pixels = cv2.inRange(hsv_frame, self.low_treshold_hsv_black, self.high_treshold_hsv_black)
		# to find the white pixels, we look for low saturation and high value
		white_pixels = cv2.inRange(hsv_frame, self.low_treshold_hsv_white, self.high_treshold_hsv_white)

		# image composed of the white and black pixels
		black_white_image = black_pixels + white_pixels
		if show_all_subimages:
			cv2.imshow(self.window_name,black_white_image)
			if cv2.waitKey() & 0xFF == 27 :
				quit()

		# 3. open and close the image
		opening_kernel = np.ones((self.opening_kernel_size, self.opening_kernel_size))
		opened_image = cv2.morphologyEx(black_white_image, cv2.MORPH_CLOSE, opening_kernel)
		closing_kernel = np.ones((self.closing_kernel_size, self.closing_kernel_size))
		closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, closing_kernel)

		src = cv2.cvtColor(closed_image, cv2.COLOR_GRAY2BGR)

		if show_all_subimages:
			cv2.imshow(self.window_name,src)
			if cv2.waitKey() & 0xFF == 27 :
				quit()

		# 4. find the edges and lines
		edges = cv2.Canny(src, 2500, 2700, 0, 5)
		if show_all_subimages:
			cv2.imshow(self.window_name,edges)
			if cv2.waitKey() & 0xFF == 27 :
				quit()

		lines_tmp = cv2.HoughLinesP(edges,1,np.pi/180,self.nIntersectionPointsFirst,0,self.minLineLengthFirst,self.maxLineGapFirst)
		lines = np.squeeze(lines_tmp)
		left_line = [self.cropped_size[1],0,self.cropped_size[1],1]
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
			if show_all_subimages:
				cv2.line(src,(x1,y1),(x2,y2),np.array([0, 0, 255]),3)
				cv2.line(src,(left_line[0],left_line[1]),(left_line[2],left_line[3]),np.array([0, 255, 0]),3)
				cv2.imshow(self.window_name,src)
				if cv2.waitKey() & 0xFF == 27 :
					quit()

		if show_all_subimages:
			cv2.line(src,(left_line[0],left_line[1]),(left_line[2],left_line[3]),np.array([0, 255, 0]),3)
			cv2.line(src,(right_line[0],right_line[1]),(right_line[2],right_line[3]),np.array([0, 255, 0]),3)
			cv2.line(src,(bottom_line[0],bottom_line[1]),(bottom_line[2],bottom_line[3]),np.array([0, 255, 0]),3)
			cv2.imshow(self.window_name,src)
			if cv2.waitKey() & 0xFF == 27 :
				quit()

		# remove everything outside of the keyboard
		polyleft = helper.findLeftPolygon(left_line, self.cropped_height, self.cropped_width)
		polyright = helper.findRightPolygon(right_line, self.cropped_height, self.cropped_width)
		polybottom = helper.findBottomPolygon(bottom_line, self.cropped_height, self.cropped_width)

		cv2.fillConvexPoly(src, polyleft, np.array([0,0,0]))    
		cv2.fillConvexPoly(src, polyright, np.array([0,0,0]))    
		cv2.fillConvexPoly(src, polybottom, np.array([0,0,0]))    
		if show_all_subimages:
			cv2.imshow(self.window_name,src)
			if cv2.waitKey() & 0xFF == 27 :
				quit()

		# find the edges and lines again
		edges = cv2.Canny(src, 2500, 2700, 0, 5)
		if show_all_subimages:
			cv2.imshow(self.window_name,edges)
			if cv2.waitKey() & 0xFF == 27 :
				quit()

		lines_tmp = cv2.HoughLinesP(edges,1,np.pi/180,self.nImtersectionPointsSecond,0,self.minLineLengthSecond,self.maxLineGapSecond)
		lines = np.squeeze(lines_tmp)
		top_line = [self.cropped_width-100 ,self.cropped_height, self.cropped_width, self.cropped_height]
		# find the top line
		for x1,y1,x2,y2 in lines:
			if(not helper.isVerticalLine(x1,y1,x2,y2)):
				top_line = helper.resultingHorizontalTopLine(top_line,[x1,y1,x2,y2])

		polytop = helper.findTopPolygon(top_line, self.cropped_height, self.cropped_width)
		if show_all_subimages:
			cv2.line(src,(top_line[0],top_line[1]),(top_line[2],top_line[3]),np.array([0, 255, 0]),3)
			cv2.imshow(self.window_name,src)
			if cv2.waitKey() & 0xFF == 27 :
				quit()

		# find the corners of the keyboard 
		if self.side == 'right':
			corners = helper.findCorners(top_line,bottom_line,left_line,right_line)
		elif self.side == 'left':
			corners = helper.findCorners(bottom_line,top_line,right_line,left_line)

		# print corners
		if show_image:
			for point in corners:
				cv2.circle(frame, (int(point[0]),int(point[1])), 15, np.array([0,0,255]), 5)
			cv2.imshow(self.window_name,frame)
			if cv2.waitKey() & 0xFF == 27 :
				quit()

		return corners


if __name__ == "__main__":

	# path to video file
	# folder = '../data/Right_Mikael_camera/individual_keys/'
	# file = 'C4m.mp4'
	folder = '../data/Left_Toki_camera/individual_keys/'
	filename = 'C4t.mp4'
	path_to_video_file = folder + filename

	# pright = Predictor(path_to_video_file)
	pleft = Predictor(path_to_video_file)
	while pleft.advanceFrame():
		counts = pleft.countKeyDiffs(show_img=True)
		print(counts)


