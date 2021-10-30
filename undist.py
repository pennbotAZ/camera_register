"""
Undistort the image with estimated calibration distortion parameters
"""
import cv2

def undist(img, mtx, dist):
    # decide how much space I want to keep, based on orignal image size and the Free scaling parameter
    ## for the free scaling parameter, 0: all from undistorted image, 1: all from source image
    img_h, img_w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(img_w, img_h),1,(img_w, img_h)) 
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  
    # equivalent to initUndistortRectifyMap and remap

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    return dst