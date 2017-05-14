#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibrate.py <video path>
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import sys

if __name__ == '__main__':
    path_video = sys.argv[1]

    # create video capture object
    cap = cv2.VideoCapture(path_video)

    if(not cap.isOpened()):
        print("error reading video")
        exit()

    square_size = 1.0

    pattern_size = (9, 6)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0
    img_names_undistort = []
    # read the frames
    idx_frame = -1
    while(cap.isOpened()):
        # read the frame
        ret, img = cap.read()
        if not ret:
            break
        idx_frame += 1
        if idx_frame % 10 != 0:
            continue
        if idx_frame > 200:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is None:
            print("Failed to load", fn)
            continue

        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if not found:
            print('chessboard not found')
            continue

        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

        print(idx_frame, 'ok')

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())
    np.savez(path_video, camera_matrix=camera_matrix, dist_coefs=dist_coefs)

    # undistort the image with the calibration
    print('')
    for img_found in img_names_undistort:
        img = cv2.imread(img_found)

        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        outfile = img_found + '_undistorted.png'
        print('Undistorted image written to: %s' % outfile)
        cv2.imwrite(outfile, dst)

    cv2.destroyAllWindows()
