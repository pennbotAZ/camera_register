# steps
1. Calibrate the camera in two passes: estimate K and distortion parameters in the first pass, and estimate the final K after undistort the image (Skip this step in Tien's code). To do this, use checkerboard pattern data.  

2. Run [colmap](https://colmap.github.io/) or [Tien's code](https://github.com/tien-d/Ego4DLocalization/tree/fisheye)
to get camera pose estimation

3. Test with point cloud (either from Matterport or from Kinect RGBD data) projection. If the projected points match with the original data then the estimated pose is correct. 

4. Verify with relative pose test. We can use epipolar line to test the relative pose between two views. Ideally epipolar line will pass the feature points of the same color. We must start from a very accurate estimated view, otherwise the error will accumulate. So we use this as a verification step after step 3.

