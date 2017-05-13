#!/usr/local/bin/python
import numpy as np
import cv2
import keripiav_helper_functions as helper

def imshow(img, scale_down=1):
    cv2.imshow("image", cv2.resize(img, (img.shape[1] / scale_down, img.shape[0] / scale_down)).astype(np.uint8))
    cv2.waitKey(0)

'''
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
    # Four test keyboard corner points in image: UL, UR, BR, BL
    points_img = np.array([[35,1581],[721,588],[856,597],[341,1689]])
    points_img = np.hstack((points_img, np.ones((4,1))))

    # Four virtual keyboard corner points
    # 23.5mm x 125mm white keys, 13.7mm x 80mm black keys
    # 52 white keys, 36 black keys
    # 1222mm x 125mm total
    SIZE_KEYBOARD = (1222, 125)
    points_virtual = np.array([[0,0],[0,SIZE_KEYBOARD[0]],[SIZE_KEYBOARD[1],SIZE_KEYBOARD[0]],[SIZE_KEYBOARD[1],0]])
    points_virtual = np.hstack((points_virtual, np.ones((4,1))))

    # Load image and plot corners
    img = cv2.imread("image1.png", cv2.IMREAD_COLOR)
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
    pixels_virtual = np.clip(pixels_virtual[:,:2], 0, np.array(SIZE_KEYBOARD)[1::-1] - 1)

    # Construct projected image
    img_virtual = np.zeros((SIZE_KEYBOARD[1], SIZE_KEYBOARD[0], 3))
    img_virtual[pixels_virtual[:,0],pixels_virtual[:,1],:] = img[pixels_img[:,1],pixels_img[:,0],:]
    imshow(img_virtual)

    cv2.destroyAllWindows()
