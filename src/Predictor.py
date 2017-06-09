#!/usr/bin/env python
import numpy as np
import cv2
import keripiav_helper_functions as helper
import project_keyboard
import corners
from project_keyboard import imshow, homogenize, dehomogenize
import Filter
from collections import Counter
import os
import corners
import time

# import urllib
# import cv2
# from win32api import GetSystemMetrics

#the [x, y] the last right-click event will be stored here
# right_click = list()

#this function will be called whenever the mouse is right-clicked


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
	def __init__(self, video_file, camera_side=None, calibration_file=None, use_fixed_parameters=True):

		# open the video
		self.video_stream = cv2.VideoCapture(video_file)
		if(not self.video_stream.isOpened()):
			print "error reading video\n"
			quit()

		# read the first frame, use it to find the side of the camera, the zone of interest and initialize the baseline
		ret, self.frame = self.video_stream.read()
		ret, self.frame = self.video_stream.read()
		self.corners, self.pos_camera = corners.find_corners(self.frame)
		self.updateBaseline(self.frame)


		# Load calibration parameters
		if self.pos_camera == "right":
			self.calibration = "../data/calibration/nexus5.mp4.npz"
		elif self.pos_camera == "left":
			self.calibration = "../data/calibration/galaxy_s7-7.mp4.npz"
		if self.calibration:
			npz_calibration = np.load(self.calibration)
			self.camera_matrix = npz_calibration["camera_matrix"]
			self.dist_coefs = npz_calibration["dist_coefs"]

		self.window_name = 'Predictor_' + self.pos_camera

		if use_fixed_parameters:
			if self.pos_camera == 'right':
				npz_parameters = np.load('../data/fixed_parameters/Transformations_right.npz')
				self.corners = npz_parameters["corners"]
				self.T_img_to_virtual = npz_parameters["T_img_to_virtual"]
				self.T_virtual_to_img = npz_parameters["T_virtual_to_img"]
				self.frame = cv2.imread('../data/fixed_parameters/baseline_right.png')
			else:
				if 'individual_keys' in video_file:
					npz_parameters = np.load('../data/fixed_parameters/Transformations_left_ind_keys.npz')
					self.corners = npz_parameters["corners"]
					self.T_img_to_virtual = npz_parameters["T_img_to_virtual"]
					self.T_virtual_to_img = npz_parameters["T_virtual_to_img"]
					self.frame = cv2.imread('../data/fixed_parameters/baseline_left_ind_keys.png')
				else:
					npz_parameters = np.load('../data/fixed_parameters/Transformations_left.npz')
					self.corners = npz_parameters["corners"]
					self.T_img_to_virtual = npz_parameters["T_img_to_virtual"]
					self.T_virtual_to_img = npz_parameters["T_virtual_to_img"]
					self.frame = cv2.imread('../data/fixed_parameters/baseline_left.png')
			self.updateBaseline(self.frame)
			self.key_map = project_keyboard.key_map(self.frame.shape, self.T_virtual_to_img, self.pos_camera)
			self.key_mask = (self.key_map > 0)[:,:,np.newaxis]
		else:
			# get a suitable frame for the baseline
			cv2.namedWindow( self.window_name+"_baseline", cv2.WINDOW_NORMAL);
			cv2.resizeWindow(self.window_name+"_baseline", 400, 550);
			print "\n******************************************************************************************"
			print   "Select baseline. Press Enter if the image is satisfying, and space to go to the next image"
			print   "******************************************************************************************"
			while True:
				self.advanceFrame(filter_corners = False, update_projection=True)
				imshow(self.frame_marked, window=self.window_name+"_baseline")
				if cv2.waitKey() == 10:
					break
			self.updateBaseline(self.frame)
			self.corners, pos_camera = corners.find_corners(self.frame)
		
			cv2.namedWindow( self.window_name, cv2.WINDOW_NORMAL);
			cv2.resizeWindow(self.window_name, 400, 550);
			cv2.setMouseCallback(self.window_name, self.click_callback)


		self.fp = Filter.ProximityFilter(self.corners, 50)
		self.fb = Filter.ButterworthFilter(self.corners,0.3,2)

		# # find the hsv treshold for the hand


		if self.pos_camera == 'right':
			self.hand_threshold_low = np.array([0, 10, 50])
			self.hand_threshold_high = np.array([50, 255, 255])
			# self.hand_threshold_low = np.array([0, 30, 120])
			# self.hand_threshold_high = np.array([40, 150, 255])
		elif self.pos_camera == 'left':
			self.hand_threshold_low = np.array([0, 50, 80])
			self.hand_threshold_high = np.array([200, 255, 255])

		# 	self.skin_pixel_location = list()
		# 	print "\n***********************"
		# 	print   "Right click on the skin"
		# 	print   "***********************"	
		# 	cv2.setMouseCallback(self.window_name+"_baseline", self.click_callback)
		# 	imshow(self.frame, window=self.window_name+"_baseline")
		# 	cv2.waitKey()

		# 	low_h_skin = max(self.baseline_hsv[self.skin_pixel_location[0],self.skin_pixel_location[1],0]-25,0)
		# 	high_h_skin = min(self.baseline_hsv[self.skin_pixel_location[0],self.skin_pixel_location[1],0]+25,255)
		# 	low_s_skin = max(self.baseline_hsv[self.skin_pixel_location[0],self.skin_pixel_location[1],1]-50,0)
		# 	high_s_skin = min(self.baseline_hsv[self.skin_pixel_location[0],self.skin_pixel_location[1],1]+50,255)
		# 	low_v_skin = max(self.baseline_hsv[self.skin_pixel_location[0],self.skin_pixel_location[1],2]-50,0)
		# 	high_v_skin = min(self.baseline_hsv[self.skin_pixel_location[0],self.skin_pixel_location[1],2]+50,255)

		# 	self.hand_threshold_low = np.array([low_h_skin, low_s_skin, low_v_skin])
		# 	self.hand_threshold_high = np.array([high_h_skin, high_s_skin, high_v_skin])

		# print "low skin hsv threshold : ", self.hand_threshold_low
		# print "high skin hsv threshold : ", self.hand_threshold_high

		# print self.baseline_hsv[self.skin_pixel_location[0],self.skin_pixel_location[1]]



		# TODO: tune diff thresholds
		self.pos_diff_threshold_low = np.array([0,0,120])
		self.pos_diff_threshold_high = np.array([255,255,255])
		self.neg_diff_threshold_low = np.array([0,0,120])
		self.neg_diff_threshold_high = np.array([255,255,255])


	
	def advanceFrame(self, filter_corners = True, update_projection=False, project_image=True):

		# start_time = time.time()

		ret, self.frame = self.video_stream.read()
		if not ret:
			return False

		# Undistort frame
		if self.calibration:
			h, w = self.frame.shape[:2]
			newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coefs, (w, h), 1, (w, h))
			self.frame = cv2.undistort(self.frame, self.camera_matrix, self.dist_coefs, None, newcameramtx)
			x, y, w, h = roi
			self.frame = self.frame[y:y+h,x:x+w]

		# Create copy of frame for marking
		self.frame_marked = self.frame.copy()

		if update_projection:
			# Find corners
			self.corners, pos_camera = corners.find_corners(self.frame_marked, self.pos_camera, mark_img=True, show_img=False)
			# TODO: filter corners
			# print "\nunfiltered corner 1 : ", self.corners[0]
			if filter_corners:
				self.corners = self.fb.updateFilter(self.fp.updateFilter(self.corners))
				# print "filtered corner 1 : ", self.corners[0]

			self.corners = self.corners.astype(np.int32)
			for c in self.corners:
				cv2.circle(self.frame_marked, tuple(c[:2]), 10, (0,0,255), 3)

			# Find projection matrix and update key map/mask
			self.T_img_to_virtual, self.T_virtual_to_img = project_keyboard.find_projection(self.corners)
			self.key_map = project_keyboard.key_map(self.frame.shape, self.T_virtual_to_img, self.pos_camera)
			self.key_mask = (self.key_map > 0)[:,:,np.newaxis]
			# print "T 1 : \n", self.T_img_to_virtual

		# Project virtual image (only for visualization)
		if project_image:
			self.img_virtual = project_keyboard.project_image(self.frame, self.key_mask, self.T_img_to_virtual)
			# print "img virtual : ", self.img_virtual
			# imshow(self.img_virtual, window=self.window_name+"_virtual", wait=1)

		# print "elapsed time in advanceFrame : ", time.time() - start_time

		return True

	def countKeyDiffs(self, show_img=False):

		# start_time = time.time()

		# Use HSV
		frame_hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

		# find hand position
		hand_mask = cv2.inRange(frame_hsv, self.hand_threshold_low, self.hand_threshold_high)
		# imshow(hand_mask, self.window_name)
		# print "low skin hsv threshold : ", self.hand_threshold_low
		# print "high skin hsv threshold : ", self.hand_threshold_high
		# imshow(hand_mask, window=self.window_name, wait=1)
		# cv2.waitKey(50)

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

		# remove the hand
		# imshow(pos_diff, window=self.window_name, wait=1)
		# imshow(neg_diff, window=self.window_name, wait=1)
		pos_diff[np.argwhere(hand_mask)[:,0],np.argwhere(hand_mask)[:,1]] = 0
		neg_diff[np.argwhere(hand_mask)[:,0],np.argwhere(hand_mask)[:,1]] = 0
		# imshow(pos_diff, window=self.window_name, wait=1)
		# imshow(neg_diff, window=self.window_name, wait=1)
		# print np.argwhere(hand_mask)

		# Find pixel coordinates
		pixels_pos_diff = np.argwhere(pos_diff)[:,::-1]
		pixels_neg_diff = np.argwhere(neg_diff)[:,::-1]

		# Count detected keys for diff pixels
		counts = Counter()

		idx_pos_diff = np.zeros((pixels_pos_diff.shape[0],), np.bool)
		for i in range(pixels_pos_diff.shape[0]):
			key = project_keyboard.key_label(self.key_map, pixels_pos_diff[i])
			if key is None or project_keyboard.is_black(key):
				continue
			counts[key] += 1
			idx_pos_diff[i] = 1

		idx_neg_diff = np.zeros((pixels_neg_diff.shape[0],), np.bool)
		for i in range(pixels_neg_diff.shape[0]):
			key = project_keyboard.key_label(self.key_map, pixels_neg_diff[i])
			if key is None or project_keyboard.is_white(key):
				continue
			counts[key] += 1
			idx_neg_diff[i] = 1

		# if show_img and counts:
		if counts:
			pixels_pos_diff = homogenize(pixels_pos_diff[idx_pos_diff,:]).astype(np.int32)
			pixels_neg_diff = homogenize(pixels_neg_diff[idx_neg_diff,:]).astype(np.int32)
			self.frame_marked[pixels_pos_diff[:,1],pixels_pos_diff[:,0]] = np.array((0,255,0))
			self.frame_marked[pixels_neg_diff[:,1],pixels_neg_diff[:,0]] = np.array((0,0,255))

			pixels_pos_diff_virtual = dehomogenize(pixels_pos_diff.dot(self.T_img_to_virtual.T)).astype(np.int32)
			pixels_neg_diff_virtual = dehomogenize(pixels_neg_diff.dot(self.T_img_to_virtual.T)).astype(np.int32)
			pixels_pos_diff_virtual = np.clip(pixels_pos_diff_virtual, 0, np.array(self.img_virtual.shape[:2]) - 1)
			pixels_neg_diff_virtual = np.clip(pixels_neg_diff_virtual, 0, np.array(self.img_virtual.shape[:2]) - 1)
			self.img_virtual[pixels_pos_diff_virtual[:,0],pixels_pos_diff_virtual[:,1]] = np.array((0,255,0))
			self.img_virtual[pixels_neg_diff_virtual[:,0],pixels_neg_diff_virtual[:,1]] = np.array((0,0,255))

		if show_img:
			imshow(self.frame_marked, scale_down=3, window=self.window_name, wait=1)
			imshow(self.img_virtual, window=self.window_name+"_virtual", wait=1)

		# print "elapsed time in countKeyDiff : ", time.time() - start_time

		return counts

	# remove the keys that have too few occurences, and the keys next to each other
	def predictKeys(self, show_img=False):
		counts = self.countKeyDiffs(show_img)

		# start_time = time.time()

		total_key_counts = sum([counts[k] for k in counts])
		if total_key_counts > 0:
			threshold = int(float(total_key_counts)/2./float(len(counts)))
		else:
			return counts

		# new_counts = {k:v for k,v in counts.iteritems() if v > max(threshold, 100)}
		new_counts = {k:v for k,v in counts.iteritems() if v > threshold}

		# print "new counts : ", new_counts

		to_remove = [];

		list_keys = [project_keyboard.indexof(k) for k in new_counts]
		list_keys.sort()
		for i in range(len(list_keys)-1):
			ki1 = list_keys[i]
			ki2 = list_keys[i+1]
			k1 = project_keyboard.key(ki1)
			k2 = project_keyboard.key(ki2)
			# print "k1 : ", k1, new_counts[k1]
			c1 = new_counts[k1]
			c2 = new_counts[k2]
			if ki1 + 1 == ki2:
				if(self.pos_camera == "right"):
					if(float(c2) > 0.8*float(c1)):
						# print "removed ", k1, " from list"
						# del new_counts[k1]
						to_remove.append(k1)
				elif(self.pos_camera == "left"):
					if(float(c1) > 0.8*float(c2)):
						# print "removed ", k2, " from list"
						to_remove.append(k2)
						# del new_counts[k2]

		for k in to_remove:
			del new_counts[k]

		# print "elapsed time in predictKeys : ", time.time() - start_time


		return new_counts




	def click_callback(self, event, x, y, flags, params):

		#right-click event value is 2
		if event == 2:

			#store the coordinates of the right-click event
			self.skin_pixel_location = [y, x]

			#this just verifies that the mouse data is being collected
			#you probably want to remove this later
			frame_hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
			print frame_hsv[y,x]
			# print self.skin_pixel_location
			# print self.frame.shape

	# manually select the zone of interest in the image
	# TODO for manual selection on the image
	# def getCroppedZone(self, frame, show_image = False):
	# 	if self.side == 'right':
	# 		self.cropped_origin = (500,50)
	# 		self.cropped_end = (1600,850)
	# 		self.cropped_size = (self.cropped_end[0]-self.cropped_origin[0],self.cropped_end[1]-self.cropped_origin[1])
	# 		self.cropped_height = self.cropped_size[0]
	# 		self.cropped_width = self.cropped_size[1]
	# 	elif self.side == 'left':
	# 		self.cropped_origin = (550,0)
	# 		self.cropped_end = (1900,1000)
	# 		self.cropped_size = (self.cropped_end[0]-self.cropped_origin[0],self.cropped_end[1]-self.cropped_origin[1])
	# 		self.cropped_height = self.cropped_size[0]
	# 		self.cropped_width = self.cropped_size[1]
	# 	if show_image:
	# 		cv2.imshow(self.window_name,frame[self.cropped_origin[0]:self.cropped_end[0], self.cropped_origin[1]:self.cropped_end[1]])
	# 		# wait for a key press, and exit if Esc is pressed
	# 		if cv2.waitKey() & 0xFF == 27 :
	# 			quit()

	# update the baseline
	def updateBaseline(self, new_baseline, show_image = False):
		# self.baseline = new_baseline[self.cropped_zone_x, self.cropped_zone_y]
		self.baseline = new_baseline
		# self.baseline_hsv = cv2.cvtColor(self.baseline, cv2.COLOR_BGR2HSV)
		self.baseline_hsv = cv2.cvtColor(self.baseline, cv2.COLOR_BGR2HSV)
		if show_image:
			cv2.imshow(self.window_name,self.baseline)
			if cv2.waitKey() & 0xFF == 27 :
				quit()


if __name__ == "__main__":

	# path to video file

	# dir_data = os.path.join("..", "data", "Right_Mikael_camera")
	# dir_data = os.path.join("..", "data", "Left_Toki_camera")
	# dir_video = os.path.join(dir_data, "individual_keys")
	# dir_video = os.path.join(dir_data, "scales")
	# dir_video = os.path.join(dir_data, "twinkle_twinkle_little_star")
	# file_video = "ttlst.mp4"
	# file_video = "ttlsm.mp4"
	# file_video = "s2m.mp4"
	# file_video = "C4m.mp4"


	# path to video file
	# DIR = os.path.join("..", "data", "Right_Mikael_camera", "individual_keys")
	DIR = os.path.join("..", "data", "Left_Toki_camera", "individual_keys")
	FILES = [f for f in os.listdir(DIR) if f.endswith(".mp4")]
	FILES = FILES[1:]
	show_img = False

	# dir_data = os.path.join("..", "data", "Left_Toki_camera")
	# dir_video = os.path.join(dir_data, "individual_keys")
	# file_video = "C4t.mp4"

	# path_to_video_file = os.path.join(dir_video, file_video)
	# path_to_calibration_file = os.path.join(dir_data, file_calibration)

	# pright = Predictor(path_to_video_file, calibration=path_to_calibration_file)
	for file in FILES:
		path_to_video_file = DIR + '/' + file
		print path_to_video_file
		idx_frame = 0
		pleft = Predictor(path_to_video_file)
		while pleft.advanceFrame(filter_corners = False, update_projection=False, project_image=True):
			# counts = pleft.countKeyDiffs(show_img=True)
			counts = pleft.predictKeys(show_img)
			idx_frame += 1
			if counts:
				print(idx_frame, counts)


