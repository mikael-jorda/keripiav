#!/usr/bin/env python
from __future__ import division
import numpy as np
import cv2
import keripiav_helper_functions as helper
import project_keyboard
from project_keyboard import imshow
import Filter
from collections import Counter
import os
import scipy

# Contrast limited adaptive histogram equalization object
clahe = cv2.createCLAHE(clipLimit=20, tileGridSize=(4, 2))

"""
Find the four corners of the keyboard

Parameters:
    img                       - RGB image
    pos_camera (optional)     - "left", "right"
        By default, this function detects the position of the camera based on
        the keyboard orientation.
    mark_img (optional)       - draw on passed in img
    show_img (optional)       - display intermediate steps
    img_white_keys (optional) - np.array to populate with a mask for the white keys.

Returns: (corners, pos_camera)
    corners    - [n x 3] np.array with the corners as homogenized points (x y 1)
                 in the order: [top_left; top_right; bottom_right; bottom_left]
    pos_camera - detected/given position of the camera ("left" or "right")
"""
def find_corners(img, pos_camera=None, mark_img=True, show_img=False, img_white_keys=None):
    # Convert to HSV color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Equalize V channel
    img_hsv[:,:,2] = clahe.apply(img_hsv[:,:,2])

    if show_img:
        imshow(np.hstack((img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2])), 3)

    # Threshold white keys
    img_w = cv2.inRange(img_hsv, np.array([20, 0, 200]), np.array([160, 60, 255]))
    img_h = cv2.inRange(img_hsv, np.array([20, 0, 0]), np.array([160, 255, 255]))
    img_s = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([255, 60, 255]))
    img_v = cv2.inRange(img_hsv, np.array([0, 0, 200]), np.array([255, 255, 255]))
    if show_img:
        imshow(np.hstack((img_h, img_s, img_v)), 3)

    # Fill white keys with watershed
    img_w[:,:10] = 255
    img_w[:,-10:] = 255
    img_w[:200,:] = 255
    img_w[-200:,:] = 255
    cv2.floodFill(img_w, np.zeros((img_w.shape[0]+2, img_w.shape[1]+2), dtype=np.uint8), (0,0), 0)
    if show_img:
        imshow(img_w, 3)
    img_fg = cv2.morphologyEx(img_w, cv2.MORPH_ERODE, np.ones((10, 10)))
    img_bg = cv2.morphologyEx(img_w, cv2.MORPH_DILATE, np.ones((100, 100)))
    img_bg[:,:10] = 0
    img_bg[:,-10:] = 0
    img_bg[:200,:] = 0
    img_bg[-200:,:] = 0
    img_bg = 255 - img_bg
    markers = np.zeros(img_w.shape, dtype=np.int32)
    markers[img_fg > 0] = 1
    markers[img_bg > 0] = 2
    markers = cv2.watershed(img, markers)
    img_w = (255 * (markers == 1)).astype(np.uint8)
    if show_img:
        imshow(img_w, 3)
    cv2.floodFill(img_w, np.zeros((img_w.shape[0]+2, img_w.shape[1]+2), dtype=np.uint8), (0,0), 0)

    # Find orientation of largest connected component
    _, contours, _ = cv2.findContours(img_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(c) for c in contours])
    vx, vy, x, y = cv2.fitLine(contours[areas.argmax()], cv2.DIST_L2, 0, 0.01, 0.01)
    V = np.vstack((vx, vy))

    # Determine camera position from keyboard orientation
    if pos_camera is None:
        if V[0] * V[1] > 0:
            pos_camera = "right"
        else:
            pos_camera = "left"

    # Dilate image with keyboard-aligned kernel and extract largest connected component
    theta = np.arctan2(V[0], V[1])
    kernel = 255 * np.round(scipy.ndimage.rotate(np.ones((150, 2)), theta * 180/np.pi)).astype(np.uint8)
    img_cc = cv2.morphologyEx(img_w, cv2.MORPH_DILATE, kernel)
    _, contours, _ = cv2.findContours(img_cc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(c) for c in contours])
    contour = np.squeeze(contours[areas.argmax()], axis=1)
    img_cc.fill(0)
    cv2.drawContours(img_cc, contours, areas.argmax(), 255, -1)
    img_w = 255 * np.logical_and(img_w>0, img_cc>0).astype(np.uint8)

    if show_img:
        imshow(img_cc, 3)

    # Find combined contour of keyboard segmentations
    _, contours, _ = cv2.findContours(img_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_w.fill(0)
    if mark_img:
        for i in range(len(contours)):
            cv2.drawContours(img, contours, i, (0,0,255), 3)
    for i in range(len(contours)):
        cv2.drawContours(img_w, contours, i, 255, -1)
    areas = np.array([cv2.contourArea(c) for c in contours])
    contours = [np.squeeze(contour, axis=1) for contour in contours]
    contour = np.vstack(contours)
    contour = np.column_stack((contour, np.ones(contour.shape[0])))

    if show_img:
        imshow(img_w, 3)

    # Find corners
    corner_left = contour[contour[:,0].argmin()].astype(np.int32)
    corner_right = contour[contour[:,0].argmax()].astype(np.int32)
    corner_bottom = contour[contour.shape[0]-1-contour[:,1][::-1].argmax()].astype(np.int32)
    corner_top = contour[contour[:,1].argmin()].astype(np.int32)

    # Push up bottom corner to the key's surface
    num_white = 0
    for i in range(40):
        vec_w = img_w[corner_bottom[1]-i,corner_bottom[0]-20:corner_bottom[0]+20]>0
        num_white_next = vec_w.sum()
        if num_white_next - num_white > 3:
            corner_bottom[1] -= i - 1
            break
        num_white = num_white_next

    # Refine corner with subpixel search
    corners = np.row_stack((corner_left[:2], corner_right[:2], corner_bottom[:2], corner_top[:2]))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(img_v,np.float32(corners.astype(np.float64)),(10,10),(-1,-1),criteria)
    corner_left = np.round(np.append(corners[0], 1)).astype(np.int32)
    corner_right = np.round(np.append(corners[1], 1)).astype(np.int32)
    corner_bottom = np.round(np.append(corners[2], 1)).astype(np.int32)
    corner_top = np.round(np.append(corners[3], 1)).astype(np.int32)

    # Determine front and back sides of the keyboard
    if pos_camera == "right":
        corner_back = corner_right
        corner_front = corner_left
        contour_back = img_w.shape[1] - 1 - np.argmax(img_w[corner_top[1]:corner_back[1]+1,::-1]>0, axis=1)
    else:
        corner_back = corner_left
        corner_front = corner_right
        contour_back = np.argmax(img_w[corner_top[1]:corner_back[1]+1]>0, axis=1)

    # Find back contour
    idx = np.logical_and(contour_back<img_w.shape[1]-1, contour_back>0)
    contour_back = np.column_stack((contour_back, \
                                    np.arange(corner_top[1], corner_back[1]+1),
                                    np.ones((contour_back.shape[0],), dtype=np.int32)))
    contour_back = contour_back[idx]
    contour_back_origin = contour_back - corner_back
    cv2.polylines(img, np.int32([contour_back[:,:2]]), False, (0,255,255), 5)
    corner_top = contour_back[contour_back[:,1].argmin()]

    # Rotate line from vertical position until it hits the back contour
    num_hit = 0
    if pos_camera == "right":
        sign_theta = 1
    else:
        sign_theta = -1
    for theta in np.linspace(0,np.pi/2,90):
        line_back = [sign_theta * np.cos(theta), -np.sin(theta), 0]
        num_hit_new = np.sum(np.dot(contour_back_origin, line_back)>0)
        # Stop when the gradient of hit pixels spikes
        if num_hit_new - num_hit > contour_back.shape[0] / 30 and theta > 0:
            break
        num_hit = num_hit_new
    line_back[-1] = -np.dot(corner_back, line_back)

    # Update contour to include only points close to the line
    contour_back = contour_back[np.abs(np.dot(contour_back, line_back))<10,:]

    if mark_img:
        cv2.polylines(img, np.int32([contour_back[:,:2]]), False, (0,255,0), 5)
        dir_line_back = np.array([line_back[1], -line_back[0]])
        points_line_back = np.array([2000, -2000])[:,np.newaxis] * dir_line_back[np.newaxis,:] + corner_back[np.newaxis,:2]
        cv2.line(img, tuple(points_line_back[0].astype(np.int32)), tuple(points_line_back[1].astype(np.int32)), (255,0,255), 5)

    # Fit least squares line to back contour
    # U, S, VT = np.linalg.svd(contour_back[:,:2] - corner_back[:2])
    U, S, VT = np.linalg.svd(contour_back[:,:2] - contour_back[:,:2].mean(axis=0))
    line_back = np.append(VT[-1], 0)
    line_back[-1] = -np.dot(corner_back, line_back)

    if mark_img:
        dir_line_back = np.array([line_back[1], -line_back[0]])
        points_line_back = np.array([2000, -2000])[:,np.newaxis] * dir_line_back[np.newaxis,:] + corner_back[np.newaxis,:2]
        cv2.line(img, tuple(points_line_back[0].astype(np.int32)), tuple(points_line_back[1].astype(np.int32)), (255,0,0), 5)

    # Find intersection between back and top lines
    corner_top_mid = corner_top
    line_top = np.cross(corner_top, corner_front).astype(np.float32)
    line_top /= np.linalg.norm(line_top)
    corner_top = np.cross(line_back, line_top)
    corner_top /= corner_top[-1]
    corner_top = np.round(corner_top).astype(np.int32)

    # Plot corners
    if mark_img:
        cv2.circle(img, tuple(corner_top[:2]), 10, (0,255,0), 3)
        cv2.circle(img, tuple(corner_right[:2]), 10, (255,255,0), 3)
        cv2.circle(img, tuple(corner_bottom[:2]), 10, (255,0,0), 3)
        cv2.circle(img, tuple(corner_left[:2]), 10, (255,0,255), 3)
        cv2.circle(img, tuple(corner_top_mid[:2]), 10, (255,0,255), 3)

    # Collect corners
    if pos_camera == "left":
        corners = np.row_stack((corner_left, corner_top, corner_right, corner_bottom))
    else:
        corners = np.row_stack((corner_top, corner_right, corner_bottom, corner_left))

    if img_white_keys is not None:
        img_white_keys[:,:] = img_w
    return corners, pos_camera

    # keypoints = surf.detect(img_v,None)
    # for kp in keypoints:
    #     x, y = kp.pt
    #     cv2.circle(img, (int(x), int(y)), 3, (255,255,0), 5)
    # imshow(img, 3)

    # dst = cv2.cornerHarris(img_v,2,3,0.04)
    # harris_corners_y, harris_corners_x = np.nonzero(dst>0.1*dst.max())
    # for i in range(harris_corners_y.shape[0]):
    #     cv2.circle(img_clean, (harris_corners_x[i], harris_corners_y[i]), 1, (0,0,255), 1)

    # Refine corners
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001)
    # corners = cv2.cornerSubPix(img_v,np.float32(corners[:,:2].astype(np.float64)),(10,10),(-1,-1),criteria)
    # corner_top[:2] = np.round(corners[0]).astype(np.int32)
    # corner_right[:2] = np.round(corners[1]).astype(np.int32)
    # corner_bottom[:2] = np.round(corners[2]).astype(np.int32)
    # corner_left[:2] = np.round(corners[3]).astype(np.int32)
    # return corners


DIR = os.path.join("..", "data", "individual_keys")
# DIR = os.path.join("..", "data")
FILES = [f for f in os.listdir(DIR) if f.endswith(".mp4")]
# FILES = FILES[1:]
show_img = True
print(FILES)

npz_calibration = np.load(os.path.join("..", "data", "calibration", "galaxy_s7-7.mp4.npz"))
# npz_calibration = np.load(os.path.join("..", "data", "calibration", "nexus5.mp4.npz"))
camera_matrix = npz_calibration["camera_matrix"]
dist_coefs = npz_calibration["dist_coefs"]

if __name__ == "__main__":
    for filename in FILES:
        # Create video capture object
        cap = cv2.VideoCapture(os.path.join(DIR, filename))
        if not cap.isOpened():
            raise IOError("Error reading video")

        # SIFT object
        sift = cv2.xfeatures2d.SIFT_create()
        # SURF object
        surf = cv2.xfeatures2d.SURF_create()

        num_iter = 0
        # cap.set(cv2.CAP_PROP_POS_FRAMES, num_iter)
        while cap.isOpened():
            # Read the frame
            _, img = cap.read()
            print("Iteration %d" % num_iter)
            num_iter += 1
            if img is None:
                break

            # Undistort image
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
            img = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
            x, y, w, h = roi
            img = img[y:y+h,x:x+w]
            img_clean = img.copy()

            img_w = np.zeros(img.shape[:2])
            corners, pos_camera = find_corners(img, img_white_keys=img_w)

            # Project keyboard
            img_virtual, T_img_to_virtual, T_virtual_to_img = project_keyboard.project_image(img_clean, corners)

            def find_black_key_corner(key):
                # Find overlapping white key
                if (pos_camera == "left" and key[1] == "b") or (pos_camera == "right" and key[1] == "#"):
                    key_white = key[0] + key[-1]
                else:
                    idx_key = "ABCDEFG".index(key[0]) - 1
                    key_white = "ABCDEFG"[idx_key] + key[-1]
                bbox_white = np.column_stack((project_keyboard.bounding_box(key_white), np.ones(4)))
                bbox_white = (np.min(bbox_white, axis=0)[:2].astype(np.int32), np.max(bbox_white, axis=0)[:2].astype(np.int32))
                img_white = np.zeros((bbox_white[1] - bbox_white[0]).astype(np.int32))
                X_white, Y_white = np.meshgrid(np.arange(bbox_white[0][0], bbox_white[1][0]),
                                               np.arange(bbox_white[0][1], bbox_white[1][1]))
                pixels_white_virtual = np.column_stack((X_white.flatten(), Y_white.flatten(), np.ones((X_white.shape[0]*X_white.shape[1],))))
                pixels_white_img = pixels_white_virtual.dot(T_virtual_to_img.T)
                pixels_white_img = np.round(pixels_white_img / pixels_white_img[:,-1,np.newaxis]).astype(np.int32)
                idx_white = img_w[pixels_white_img[:,1],pixels_white_img[:,0]] > 0
                pixels_white_virtual = pixels_white_virtual[:,:2] - bbox_white[0][np.newaxis,:]
                img_white[pixels_white_virtual[idx_white,0].astype(np.int32),pixels_white_virtual[idx_white,1].astype(np.int32)] = 255

                # Find corner of black key
                histogram = np.sum(img_white > 0, axis=1)
                idx = np.argmax(histogram > np.max(histogram[:int(histogram.shape[0]/2)]))
                pixel_black_virtual = np.array((idx, np.argmax(img_white[idx,:] > 0) + bbox_white[0][1], 1))
                pixel_black_img = T_virtual_to_img.dot(pixel_black_virtual)
                pixel_black_img /= pixel_black_img[-1]
                pixel_black_img = pixel_black_img[:2].astype(np.int32)
                return pixel_black_img

            # Find corners for closest three black keys
            points_img = []
            points_virtual = []
            if pos_camera == "left":
                for key in list(project_keyboard.black_keys())[:3]:
                    pixel_corner_img = find_black_key_corner(key)
                    bbox_virtual3d = project_keyboard.bounding_box(key)
                    pixel_corner_virtual = np.append(bbox_virtual3d[-2,:], 1)
                    points_virtual.append(pixel_corner_virtual)
                    points_img.append(pixel_corner_img)
            else:
                for key in list(project_keyboard.black_keys())[-3:]:
                    pixel_corner_img = find_black_key_corner(key)
                    bbox_virtual3d = project_keyboard.bounding_box(key)
                    pixel_corner_virtual = np.append(bbox_virtual3d[-1,:], 1)
                    points_virtual.append(pixel_corner_virtual)
                    points_img.append(pixel_corner_img)
            points_virtual = np.row_stack(points_virtual)
            points_img = np.row_stack(points_img)

            # Find 3d projection with black keys
            T_virtual3d_to_img = project_keyboard.perspective_transformation_3d(T_virtual_to_img, points_virtual, points_img)
            T_virtual3d_to_virtual = T_img_to_virtual.dot(T_virtual3d_to_img)
            points_img_flat = points_virtual.dot(T_virtual3d_to_img.T)
            points_img_flat /= points_img_flat[:,-1,np.newaxis]
            points_virtual_flat = points_img_flat.dot(T_img_to_virtual.T)
            points_virtual_flat /= points_virtual_flat[:,-1,np.newaxis]
            points_virtual_flat = points_virtual_flat.astype(np.int32)

            cv2.circle(img_virtual, tuple(points_virtual_flat[0,:2][::-1]), 10, (0,0,255), 3)
            cv2.circle(img_virtual, tuple(points_virtual_flat[1,:2][::-1]), 10, (0,0,255), 3)
            cv2.circle(img_virtual, tuple(points_virtual_flat[2,:2][::-1]), 10, (0,0,255), 3)

            # Create map of key indices
            print(T_virtual3d_to_img)
            # img_map = project_keyboard.key_map(img.shape, T_virtual3d_to_img, pos_camera)
            img_map = project_keyboard.key_map(img.shape, T_virtual_to_img, pos_camera)
            # if show_img:
            imshow(img_map, 3)

            # Plot black keys
            # TODO: Remove
            # if pos_camera == "left":
            #     black_keys = list(reversed(list(project_keyboard.black_keys())))
            # else:
            #     black_keys = list(project_keyboard.black_keys())
            # colors = [(0,0,255),(0,255,255),(0,255,0),(255,0,0),(255,0,255)]
            # idx = 0
            # for key in black_keys:
            #     bbox_black_virtual3d = project_keyboard.bounding_box(key)
            #     bbox_black_virtual3d = np.column_stack((bbox_black_virtual3d, np.ones(bbox_black_virtual3d.shape[0],)))
            #     bbox_black_img = bbox_black_virtual3d.dot(T_virtual3d_to_img.T)
            #     bbox_black_img = (bbox_black_img[:,:2] / bbox_black_img[:,2,np.newaxis]).astype(np.int32)
            #     hull_black_img = np.squeeze(cv2.convexHull(bbox_black_img))
            #     cv2.fillConvexPoly(img, hull_black_img, colors[idx])

            #     bbox_black_virtual = bbox_black_virtual3d.dot(T_virtual3d_to_virtual.T)
            #     bbox_black_virtual = (bbox_black_virtual[:,:2] / bbox_black_virtual[:,2,np.newaxis]).astype(np.int32)
            #     # bbox_black_virtual = project_keyboard.bounding_box(key)[:4,:2]
            #     hull_black_virtual = np.squeeze(cv2.convexHull(bbox_black_virtual))
            #     cv2.fillConvexPoly(img_virtual, hull_black_virtual[:,::-1], colors[idx])
            #     idx += 1
            #     if idx >= len(colors):
            #         idx = 0

            # if show_img:
            imshow(img, 3)

        break

