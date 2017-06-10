#!/usr/bin/env python
# import numpy as np
# import cv2
# import keripiav_helper_functions as helper
# import project_keyboard
# from project_keyboard import imshow
import os, time
import Predictor
from collections import Counter
import project_keyboard

def predictFrom2Cameras(counts_left, counts_right):

	# start_time = time.time()

	# combined_counts = Counter()
	combined_counts = Counter(counts_left) + Counter(counts_right)

	total_key_counts = sum([combined_counts[k] for k in combined_counts])
	if total_key_counts > 0:
		threshold = int(float(total_key_counts)/3./float(len(combined_counts)))
	else:
		return []

	new_counts = {k:v for k,v in combined_counts.iteritems() if v > max(threshold,100)}

	to_remove = []
	for k in counts_left:
		if (project_keyboard.isInRightHalf(k)) and (k not in counts_right):
			to_remove.append(k)

	for k in counts_right:
		if (not project_keyboard.isInRightHalf(k)) and (k not in counts_left):
			to_remove.append(k)

	for k in to_remove:
		if k in new_counts:
			del new_counts[k]

	# print "elapsed time in predictFrom2cameras : ", time.time() - start_time


	if new_counts:
		return list(new_counts)
	else:
		return []

# remove a key if is is not played for at least 3 frames
# 
def postProcessListKeys(list_keys):
	processed_list = []
	for i in range(len(list_keys)-2):
		# k = list_keys[i]
		frame_keys = [k for k in list_keys[i] if (k in list_keys[i+1] and k in list_keys[i+2])]
		processed_list.append(frame_keys)

	return processed_list

def writeToFile(file, l, pl, vid1, vid2):
	file.write("%s\n" % vid1)
	file.write("%s\n" % vid2)
	for k in l:
		file.write("%s, " % k)
	file.write("\n")
	for k in pl:
		file.write("%s, " % k)
	file.write("\n\n")
	file.flush()


save_file_path = "../data/results/scale3.txt"
save_file = open(save_file_path, 'w')

# print project_keyboard.isInRightHalf('Bb4')


# path to video file
DIRL = os.path.join("..", "data", "Left_Toki_camera", "scales")
DIRR = os.path.join("..", "data", "Right_Mikael_camera", "scales")
# DIR = os.path.join("..", "data")
# FILESL = [f for f in os.listdir(DIRL) if f.endswith(".mp4")]
# FILESL = FILESL[1:]
# FILESL = ['s3t.mp4', 's6t.mp4', 'schromatic1t.mp4', 'schromatic4t.mp4', 'schromatic7t.mp4', 'schromatic5t.mp4']
FILESL = ['s3t.mp4']
# FILESR = [f for f in os.listdir(DIRR) if f.endswith(".mp4")]
# FILESR = FILESR[1:]
# show_img = True
show_img = True
# print(FILESL)
# print(FILESR)

# if(len(FILESR) != len(FILESL)):
	# print "ERROR : not the same number of files for the two cameras"
	# quit()

idx_frame = 0
for i in range(len(FILESL)):
	fl = FILESL[i]
	fr = fl[0:-5] + 'm' + fl[-4::]
	# fr[-5] = 'm'

	fr = DIRR + '/' + fr
	fl = DIRL + '/' + fl
	print "file ", i+1, " of ", len(FILESL)
	print fl
	print fr
	print ""
	pleft = Predictor.Predictor(fl, use_fixed_parameters=True)
	pright = Predictor.Predictor(fr, use_fixed_parameters=True)
	list_keys = []
	while pleft.advanceFrame(filter_corners = False, update_projection=False, project_image=True):
		pright.advanceFrame(filter_corners = False, update_projection=False, project_image=True)
		# counts = pleft.countKeyDiffs(show_img=True)
		countsl = pleft.predictKeys(show_img)
		countsr = pright.predictKeys(show_img)
		idx_frame += 1
		ccounts = predictFrom2Cameras(countsl, countsr)
		# print "ccounts : ", ccounts
		list_keys.append(ccounts)
		if ccounts != []:
			print(idx_frame, list_keys[-1])
	# print list_keys
	processed_list_keys = postProcessListKeys(list_keys)
	writeToFile(save_file, list_keys, processed_list_keys, fl, fr)

save_file.close()
# cv2.destroyAllWindows()