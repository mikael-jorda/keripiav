#!/usr/bin/env python
from __future__ import division
import numpy as np
import cv2
import keripiav_helper_functions as helper
import os
import math

# 23mm x 125mm white keys, 13.7mm x 80mm black keys
# 52 white keys, 36 black keys
# 1196 x 125mm total
NUM_WHITE_KEYS = 52
NUM_BLACK_KEYS = 36
DIM_WHITE_KEYS = (23, 125)
DIM_KEYBOARD = (NUM_WHITE_KEYS * DIM_WHITE_KEYS[0], DIM_WHITE_KEYS[1])

'''
Show image scaled down by scale_down factor and wait for key press
'''
def imshow(img, scale_down=1):
    cv2.imshow("image", cv2.resize(img, (int(img.shape[1] / scale_down), int(img.shape[0] / scale_down))).astype(np.uint8))
    cv2.waitKey(0)

'''
Return key label from (x,y) pixel coordinates
'''
def key_label(pixel):
    letters = "ABCDEFG"
    idx_octave = int(math.floor(pixel[0] / (7 * DIM_WHITE_KEYS[0]) + 5/7))
    idx_letter = int((pixel[0] % (7 * DIM_WHITE_KEYS[0])) / DIM_WHITE_KEYS[0])
    key = "%s%d" % (letters[idx_letter], idx_octave)
    return key

'''
Find perspective transformation from points1 to points2
    points1: (x y 1) [n x 3]
    points2: (x' y' 1) [n x 3]
'''
def perspective_transformation(points1, points2):
    # Projective transform
    # [ x' ]   [ a b c ] [ x ]
    # [ y' ] = [ d e f ] [ y ]
    # [ 1  ]   [ g h i ] [ 1 ]
    assert(points1.shape[1] == 3 and points2.shape[1] == 3)
    assert(all(points1[:,2] == 1) and all(points2[:,2] == 1))

    # Find centroids of corner points
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)

    # Find translations to move centroids to origin
    T_1_to_center = np.eye(3)
    T_1_to_center[:2,2] = -centroid1[:2]
    T_center_to_1 = np.linalg.inv(T_1_to_center)

    T_2_to_center = np.eye(3)
    T_2_to_center[:2,2] = -centroid2[:2]
    T_center_to_2 = np.linalg.inv(T_2_to_center)

    # Translate points around centroid
    points1 = points1.dot(T_1_to_center.T)
    points2 = points2.dot(T_2_to_center.T)

    # Flattened system of equations
    # [ x y 1 0 0 0 -xx' -yx' -x' ] * [ a b c d e f g h i ]' = [ 0 ]
    # [ 0 0 0 x y 1 -xy' -yy' -y' ]                            [ 0 ]
    M = np.asarray(np.bmat([[points1, np.zeros(points1.shape), -points2[:,0,np.newaxis] * points1],
                            [np.zeros(points1.shape), points1, -points2[:,1,np.newaxis] * points1]]))

    # Set i = 1
    A = M[:,:-1]
    b = -M[:,-1]
    x = np.append(np.linalg.solve(A, b), 1)

    # Collect terms into matrix
    T = np.reshape(x, (3,3))

    # Find perspective transformation from image points to virtual points
    T_1_to_2 = T_center_to_2.dot(T).dot(T_1_to_center)

    return T_1_to_2

if __name__ == "__main__":
    # Define four keyboard corners in image: upper left, upper right, bottom right, bottom left
    # points_img = np.array([[35,1581],[721,588],[856,597],[341,1689]])
    points_img = np.array([[34,1446],[648,543],[774,554],[305,1539]])
    # points_img = np.array([[34,1401],[639,520],[762,532],[301,1493]])
    # points_img = np.array([[30,1233],[573,469],[684,476],[270,1314]])
    # points_img = np.array([[37,1487],[673,559],[804,570],[318,1584]])
    # points_img = np.array([[42,1559],[707,582],[844,594],[333,1668]])
    points_img = np.hstack((points_img, np.ones((4,1))))

    # Load calibration data
    npz_calibration = np.load(os.path.join("..", "data", "calibration", "galaxy_s7.mp4.npz"))
    camera_matrix = npz_calibration["camera_matrix"]
    dist_coefs = npz_calibration["dist_coefs"]

    # Load image
    img = cv2.imread(os.path.join("..", "data", "image1.png"), cv2.IMREAD_COLOR)

    # Undistort image and save
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
    img = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
    x, y, w, h = roi
    img = img[y:y+h,x:x+w]
    cv2.imwrite(os.path.join("..", "data", "image1_calibrated.png"), img)

    # Define virtual keyboard corner points
    points_virtual = np.array([[0,0],[0,DIM_KEYBOARD[0]],[DIM_KEYBOARD[1],DIM_KEYBOARD[0]],[DIM_KEYBOARD[1],0]])
    points_virtual = np.hstack((points_virtual, np.ones((4,1))))

    # Plot corners
    img2 = img.copy()
    for i in range(points_img.shape[0]):
        cv2.circle(img2, tuple(points_img[i,:2].astype(np.int32)), 10, (0,255,0), 5)
    imshow(img2, 3)

    # Find projection matrix
    T_img_to_virtual = perspective_transformation(points_img, points_virtual)
    print("T_img_to_virtual:")
    print(T_img_to_virtual)

    # Find keyboard pixels
    mask_keyboard = np.zeros(img.shape[:2])
    cv2.fillConvexPoly(mask_keyboard, points_img[:,:2].astype(np.int32), (1,))
    pixels_img = np.nonzero(mask_keyboard)
    pixels_img = np.column_stack((pixels_img[1], pixels_img[0], np.ones(pixels_img[0].shape)))

    # Project keyboard pixels into virtual space
    pixels_virtual = pixels_img.dot(T_img_to_virtual.T)
    pixels_virtual /= pixels_virtual[:,-1,np.newaxis]

    # Trim pixels to be proper image indices
    pixels_img = pixels_img.astype(np.int32)
    pixels_virtual = pixels_virtual.astype(np.int32)
    pixels_img = np.clip(pixels_img[:,:2], 0, np.array(img.shape)[1::-1] - 1)
    pixels_virtual = np.clip(pixels_virtual[:,:2], 0, np.array(DIM_KEYBOARD)[1::-1] - 1)

    # Construct projected image
    img_virtual = np.zeros((DIM_KEYBOARD[1], DIM_KEYBOARD[0], 3))
    img_virtual[pixels_virtual[:,0],pixels_virtual[:,1],:] = img[pixels_img[:,1],pixels_img[:,0],:]
    for i in range(NUM_WHITE_KEYS):
        cv2.line(img_virtual, (DIM_WHITE_KEYS[0] * i, 0), (DIM_WHITE_KEYS[0] * i, DIM_WHITE_KEYS[1]), (0,0,255), 1)

    imshow(img_virtual)

    cv2.destroyAllWindows()
