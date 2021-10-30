"""
Calibrate the camera with checkerboard pattern

We run calibration for two passes:
1. the first pass with raw images to get intrinsic + distortion parameters 
2. the second pass with undistorted images to get updated intrinsic
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

img_path = Path('/home/shzhou2/project/colmap/shenghao/secondSuccess/frames')
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
h, w = 7, 10
objp = np.zeros((h*w,3), np.float32)
objp[:,:2] = np.mgrid[0:h,0:w].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
i = 0
for fname in img_path.glob("00*.png"):
    img = cv2.imread(str(fname))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (h,w),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (h,w), corners2,ret)
        # plt.imshow(img)
        # cv2.waitKey(500)
    else:
        print(fname.name)
        i += 1
print('remove {} bad images'.format(i))

idx = np.random.choice(np.arange(len(objpoints)), 50)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(np.array(objpoints)[idx], np.array(imgpoints)[idx], gray.shape[::-1], None, None)
print('K: ', mtx)
print('dist: ', dist)
