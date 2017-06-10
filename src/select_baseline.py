import cv2, os
import corners
import numpy as np
import project_keyboard

def click_callback(event, x, y, flags, params):

	#right-click event value is 2
	if event == 2:

		#store the coordinates of the right-click event
		print [x, y]


save = False
# save = True

if save:
	save_folder = '../data/fixed_parameters/'

# DIR = os.path.join("..", "data", "Left_Toki_camera", "individual_keys")
DIR = os.path.join("..", "data", "Left_Toki_camera", "scales")
# DIR = os.path.join("..", "data", "Right_Mikael_camera", "individual_keys")

# file = 'C4m.mp4'
file = 's1t.mp4'

# calibration = "../data/calibration/nexus5.mp4.npz"
calibration = "../data/calibration/galaxy_s7-7.mp4.npz"

npz_calibration = np.load(calibration)
camera_matrix = npz_calibration["camera_matrix"]
dist_coefs = npz_calibration["dist_coefs"]

video_file =  DIR + '/' + file

cap = cv2.VideoCapture(video_file)
if(not cap.isOpened()):
	print "error reading video\n"
	quit()

cv2.namedWindow( 'baseline', cv2.WINDOW_NORMAL);
cv2.resizeWindow( 'baseline', 400, 550);
cv2.setMouseCallback('baseline', click_callback)


ret, frame = cap.read()

# while True:
for i in range(23):
	ret, frame = cap.read()

	# cv2.imshow('baseline', frame)
	# if(cv2.waitKey() == 10):
		# break

# Undistort frame
h, w = frame.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
frame = cv2.undistort(frame, camera_matrix, dist_coefs, None, newcameramtx)
x, y, w, h = roi
frame = frame[y:y+h,x:x+w]

# cv2.imshow('baseline', frame)
# cv2.waitKey()

# Create copy of frame for marking
frame_marked = frame.copy()

# Find corners
corners, pos_camera = corners.find_corners(frame_marked, pos_camera=None, mark_img=False, show_img=False)

# modify corners
# print pos_camera
# right camera
# corners[0] += [-10,-10,0]
# corners[3] += [0,-4,0]
# left camera indiv keys
# corners[0] += [-65,100,0]
# corners[1] += [-2,-2,0]
# corners[2] += [3,-2,0]
# corners[3] += [-55,110,0]
# left camera
# corners[0] += [0,2,0]
# corners[1] += [8,-2,0]
# corners[2] += [1,0,0]
# corners[3] += [0,0,0]

corners[0] += [0,2,0]
corners[1] += [0,-2,0]
corners[2] += [1,0,0]
corners[3] += [0,0,0]
print corners

corners = corners.astype(np.int32)
for c in corners:
	cv2.circle(frame_marked, tuple(c[:2]), 10, (0,0,255), 3)

# Find projection matrix and update key map/mask
T_img_to_virtual, T_virtual_to_img = project_keyboard.find_projection(corners)
key_map = project_keyboard.key_map(frame.shape, T_virtual_to_img, pos_camera)
key_mask = (key_map > 0)[:,:,np.newaxis]
# print T_img_to_virtual

# if project_image:
img_virtual = project_keyboard.project_image(frame, key_mask, T_img_to_virtual)
# print img_virtual

cv2.namedWindow( 'marked image', cv2.WINDOW_NORMAL);
cv2.resizeWindow( 'marked image', 400, 550);
# cv2.namedWindow( 'virtual image', cv2.WINDOW_NORMAL);
# cv2.resizeWindow( 'virtual image', 400, 550);

cv2.imshow('marked image', frame_marked)
# cv2.imshow('baseline', frame)
project_keyboard.imshow(img_virtual, window="virtual image", wait=1)
# cv2.imshow('virtual image', img_virtual)
cv2.waitKey()

if save:
	np.savez(save_folder + 'Transformations_left.npz',T_img_to_virtual=T_img_to_virtual, T_virtual_to_img=T_virtual_to_img, corners=corners)
	cv2.imwrite(save_folder + 'baseline_left.png',frame)
