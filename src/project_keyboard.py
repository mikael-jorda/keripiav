#!/usr/bin/env python
from __future__ import division
import numpy as np
import cv2
import keripiav_helper_functions as helper
import os
import math
from collections import Counter

# 23mm x 125mm white keys, 13.7mm x 80mm black keys
# 52 white keys, 36 black keys
# 1196 x 125mm total
NUM_WHITE_KEYS = 52
NUM_BLACK_KEYS = 36
DIM_WHITE_KEYS = (23, 140)
DIM_BLACK_KEYS = (14, DIM_WHITE_KEYS[1] - 56, DIM_WHITE_KEYS[1] - 49, 10) # W L_B L_T H
POS_BLACK_KEYS = (DIM_WHITE_KEYS[0] - 10, 2*DIM_WHITE_KEYS[0] - 4, 4*DIM_WHITE_KEYS[0] - 10, 5*DIM_WHITE_KEYS[0] - 7, 6*DIM_WHITE_KEYS[0] - 4)
DIM_KEYBOARD = (NUM_WHITE_KEYS * DIM_WHITE_KEYS[0], DIM_WHITE_KEYS[1])

'''
Show image scaled down by scale_down factor and wait for key press
'''
def imshow(img, scale_down=1, wait=0, window="image"):
    cv2.imshow(window, cv2.resize(img, (int(img.shape[1] / scale_down), int(img.shape[0] / scale_down))).astype(np.uint8))
    cv2.waitKey(wait)

def homogenize(points):
    if points.shape[1] >= 3 and np.all(points[:,-1] == 1):
        return points
    return np.column_stack((points, np.ones((points.shape[0],))))

def dehomogenize(points):
    if points.shape[1] == 2:
        return points
    return points[:,:-1] / points[:,-1,np.newaxis]

def indexof(key):
    idx_octave = int(key[-1])
    if key[1] == "#":
        idx_letter = "CDEFGAB".index(key[0])
        key = "CDEFGAB"[idx_letter+1] + "b" + key[-1]
    if key[1] == "b":
        idx_letter = "DEGAB".index(key[0])
        idx_key = NUM_WHITE_KEYS + 5 * idx_octave + idx_letter - 4
    else:
        idx_letter = "CDEFGAB".index(key[0])
        idx_key = 7 * idx_octave + idx_letter - 5
    return idx_key + 1

def key(idx_key):
    idx_key -= 1
    if idx_key < 0:
        return None
    if idx_key < NUM_WHITE_KEYS:
        idx_key += 5
        idx_octave = int(idx_key / 7)
        idx_letter = idx_key - 7 * idx_octave
        key = "%s%d" % ("CDEFGAB"[idx_letter], idx_octave)
    else:
        idx_key += 4 - NUM_WHITE_KEYS
        idx_octave = int(idx_key / 5)
        idx_letter = idx_key - 5 * idx_octave
        key = "%sb%d" % ("DEGAB"[idx_letter], idx_octave)
    return key

def sharp(key):
    if key[1] == "b":
        idx_letter = "CDEFGAB".index(key[0])
        key = "CDEFGAB"[idx_letter-1] + "#" + key[-1]
    return key

def flat(key):
    if key[1] == "#":
        idx_letter = "CDEFGAB".index(key[0])
        key = "CDEFGAB"[idx_letter+1] + "b" + key[-1]
    return key

def key_map(img_shape, T_virtual_to_img, pos_camera="left"):
    img = np.zeros(img_shape[:2], dtype=np.uint8)
    if T_virtual_to_img.shape[1] == 4:
        # 3d black key projection
        T_virtual2d_to_img = np.column_stack((T_virtual_to_img[:,:2], T_virtual_to_img[:,-1]))
        for key in white_keys():
            bbox_white_virtual = homogenize(bounding_box(key))
            bbox_white_img = np.round(dehomogenize(bbox_white_virtual.dot(T_virtual2d_to_img.T))).astype(np.int32)
            cv2.fillConvexPoly(img, bbox_white_img, indexof(key))

        black_key_list = list(black_keys())
        if pos_camera == "left":
            black_key_list = reversed(black_key_list)
        for key in black_key_list:
            bbox_black_virtual3d = homogenize(bounding_box(key))
            bbox_black_img = np.round(dehomogenize(bbox_black_virtual3d.dot(T_virtual_to_img.T))).astype(np.int32)
            hull_black_img = np.squeeze(cv2.convexHull(bbox_black_img))
            cv2.fillConvexPoly(img, hull_black_img, indexof(key))
    else:
        # 2d black key projection
        for key in white_keys():
            bbox_white_virtual = homogenize(bounding_box(key))
            bbox_white_img = np.round(dehomogenize(bbox_white_virtual.dot(T_virtual_to_img.T))).astype(np.int32)
            cv2.fillConvexPoly(img, bbox_white_img, indexof(key))
        for key in black_keys():
            if pos_camera == "left":
                white_key = flat(key)[0] + key[-1]
            else:
                white_key = sharp(key)[0] + key[-1]
            bbox_white_virtual = homogenize(bounding_box(white_key))
            bbox_white_virtual[-2:,0] = DIM_BLACK_KEYS[2]
            bbox_white_img = np.round(dehomogenize(bbox_white_virtual.dot(T_virtual_to_img.T))).astype(np.int32)
            cv2.fillConvexPoly(img, bbox_white_img, indexof(key))

    return img

'''
Return key label from (x,y) pixel coordinates
'''
def key_label(key_map, pixel):
    pixel = np.round(pixel).astype(np.int32)
    return key(key_map[pixel[1],pixel[0]])
    # letters = "ABCDEFG"
    # idx_octave = int(math.floor(pixel[1] / (7 * DIM_WHITE_KEYS[0]) + 5/7))
    # idx_letter = int((pixel[1] % (7 * DIM_WHITE_KEYS[0])) / DIM_WHITE_KEYS[0])
    # key = "%s%d" % (letters[idx_letter], idx_octave)
    # return key

def bounding_box(key_label):
    idx_octave = int(key_label[-1])
    if key_label[1] == "b":
        idx_letter = "CDEFGAB".index(key_label[0])
        key_label = "CDEFGAB"[idx_letter-1] + "#" + key_label[-1]
    if key_label[1] == "#":
        idx_letter = "CDFGA".index(key_label[0])
        origin = np.array([[0, DIM_WHITE_KEYS[0] * (7 * idx_octave - 5) + POS_BLACK_KEYS[idx_letter], 0],
                           [0, DIM_WHITE_KEYS[0] * (7 * idx_octave - 5) + POS_BLACK_KEYS[idx_letter], DIM_BLACK_KEYS[-1]]])
        bbox = np.array([[[0,                 0,                 0],
                          [0,                 DIM_BLACK_KEYS[0], 0],
                          [DIM_BLACK_KEYS[1], DIM_BLACK_KEYS[0], 0],
                          [DIM_BLACK_KEYS[1], 0, 0]],
                         [[0,                 0,                 DIM_BLACK_KEYS[-1]],
                          [0,                 DIM_BLACK_KEYS[0], DIM_BLACK_KEYS[-1]],
                          [DIM_BLACK_KEYS[2], DIM_BLACK_KEYS[0], DIM_BLACK_KEYS[-1]],
                          [DIM_BLACK_KEYS[2], 0,                 DIM_BLACK_KEYS[-1]]]])
        bbox += origin[:,np.newaxis,:]
        bbox = np.reshape(bbox, (-1,bbox.shape[-1]))
    else:
        idx_letter = "CDEFGAB".index(key_label[0])
        origin = np.array((0, DIM_WHITE_KEYS[0] * (7 * idx_octave - 5 + idx_letter)))
        bbox = np.array([[0, 0], [0, DIM_WHITE_KEYS[0]], [DIM_WHITE_KEYS[1], DIM_WHITE_KEYS[0]], [DIM_WHITE_KEYS[1], 0]])
        bbox += origin[np.newaxis,:]
    return bbox

def black_keys():
    letters = "DEGAB"
    octaves = list(range(1,8))
    yield "Bb0"
    for octave in octaves:
        for letter in letters:
            yield "%sb%d" % (letter, octave)

def white_keys():
    letters = "CDEFGAB"
    octaves = list(range(1,8))
    yield "A0"
    yield "B0"
    for octave in octaves:
        for letter in letters:
            yield "%s%d" % (letter, octave)
    yield "C8"

def is_black(key):
    return key[1] == "b" or key[1] == "#"

def is_white(key):
    return not is_black(key)

def majority_key_label(key_map, pixels):
    keys = [key_label(key_map, pixel) for pixel in pixels]
    counts = Counter()
    for key in keys:
        counts[key] += 1
    # print(counts)
    return counts.most_common(1)[0][0]

'''
Define virtual keyboard corner points
'''
def virtual_keyboard_corners(dim=2):
    points_virtual = np.array([[0,0],[0,DIM_KEYBOARD[0]],[DIM_KEYBOARD[1],DIM_KEYBOARD[0]],[DIM_KEYBOARD[1],0]])
    if dim == 3:
        points_virtual = np.hstack((points_virtual, np.zeros((4,1))))
    points_virtual = np.hstack((points_virtual, np.ones((4,1))))
    return points_virtual

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
    if points1.shape[1] == 2:
        points1 = np.hstack((points1, np.ones((points1.shape[0],1))))
    if points2.shape[1] == 2:
        points2 = np.hstack((points2, np.ones((points2.shape[0],1))))
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

def perspective_transformation_full(points1, points2):
    # Projective transform
    # [ x' ]   [ a b c d ]   [ x ]
    # [ y' ] = [ e f g h ] * [ y ]
    # [ 1  ]   [ i j k j ]   [ z ]
    #                        [ 1 ]
    if points1.shape[1] == 2:
        points1 = np.hstack((points1, np.ones((points1.shape[0],1))))
    if points2.shape[1] == 2:
        points2 = np.hstack((points2, np.ones((points2.shape[0],1))))
    assert((points1.shape[1] == 3 or points1.shape[1] == 4) and points2.shape[1] == 3)
    assert(all(points1[:,-1] == 1) and all(points2[:,2] == 1))

    # Find centroids of corner points
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)

    # Find translations to move centroids to origin
    T_1_to_center = np.eye(points1.shape[1])
    T_1_to_center[:-1,-1] = -centroid1[:-1]
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
    x = np.append(np.linalg.lstsq(A, b)[0], 1)

    # Collect terms into matrix
    T = np.reshape(x, (3,points1.shape[1]))

    # Find perspective transformation from image points to virtual points
    T_1_to_2 = T_center_to_2.dot(T).dot(T_1_to_center)

    return T_1_to_2

def perspective_transformation_3d(T_virtual_to_img, point_virtual, point_img):
    # [ a b u c ]   [ x ]   [ x' ]
    # [ d e v f ] * [ y ] = [ y' ]
    # [ g h w i ]   [ z ]   [ 1  ]
    #               [ 1 ]
    n = point_virtual.shape[0]

    # [ z 0 -zx' ]   [ u ]   [ (gx + hy + i) x' - (ax + by + c) ]
    # [ 0 z -zy' ] * [ v ] = [ (gx + hy + i) y' - (dx + ey + f) ]
    #                [ w ]
    point_virtual_flat = np.column_stack((point_virtual[:,:2], point_virtual[:,3]))
    point_img_flat = point_virtual_flat.dot(T_virtual_to_img.T)
    b = np.hstack((point_img_flat[:,2] * point_img[:,0] - point_img_flat[:,0],
                   point_img_flat[:,2] * point_img[:,1] - point_img_flat[:,1]))
    A = np.vstack((np.column_stack((point_virtual[:,2], np.zeros((n,)), -point_virtual[:,2] * point_img[:,0])),
                   np.column_stack((np.zeros((n,)), point_virtual[:,2], -point_virtual[:,2] * point_img[:,1]))))
    uvw, _, _, _ = np.linalg.lstsq(A.astype(np.float64), b.astype(np.float64))
    # point_img_flat = point_virtual.dot(T_virtual_to_img.dot(np.delete(point_virtual, 2))
    # uv = (point_img[:2] * (point_img_flat[-1] + point_virtual[-1]) - point_img_flat[:2]) / point_virtual[-1]
    # uvw = np.append(uv, 1)
    T_virtual3d_to_img = np.column_stack((T_virtual_to_img[:,:2], uvw, T_virtual_to_img[:,-1]))
    return T_virtual3d_to_img

def find_projection(points_img, points_virtual=None, dim=2):
    # Define virtual keyboard corner points
    if points_virtual is None:
        points_virtual = virtual_keyboard_corners(dim)

    # Find projection matrix
    T_virtual_to_img = perspective_transformation_full(points_virtual, points_img)
    if dim == 3:
        T_img_to_virtual = np.linalg.inv(np.column_stack((T_virtual_to_img[:,:2], T_virtual_to_img[:,-1])))
    else:
        T_img_to_virtual = np.linalg.inv(T_virtual_to_img)

    return T_img_to_virtual, T_virtual_to_img

def project_image(img, mask_keyboard, T_img_to_virtual):
    # Find keyboard pixels
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

    return img_virtual

def isInRightHalf(key):
    i = indexof(key)
    return (i > NUM_WHITE_KEYS/2 and i < NUM_WHITE_KEYS) or (i > NUM_WHITE_KEYS + NUM_BLACK_KEYS/2)

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
    # npz_calibration = np.load(os.path.join("..", "data", "calibration", "galaxy_s7.mp4.npz"))
    npz_calibration = np.load(os.path.join("..", "data", "calibration", "nexus5.mp4.npz"))
    camera_matrix = npz_calibration["camera_matrix"]
    dist_coefs = npz_calibration["dist_coefs"]

    # Load image
    # img = cv2.imread(os.path.join("..", "data", "image1.png"), cv2.IMREAD_COLOR)
    img = cv2.imread(os.path.join("..", "data", "image2.png"), cv2.IMREAD_COLOR)

    # Undistort image and save
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
    img = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
    x, y, w, h = roi
    img = img[y:y+h,x:x+w]
    cv2.imwrite(os.path.join("..", "data", "image1_calibrated.png"), img)

    # Define virtual keyboard corner points
    points_virtual = virtual_keyboard_corners()

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
