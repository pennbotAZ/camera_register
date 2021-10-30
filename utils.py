import numpy as np
from pathlib import Path
import cv2 as cv
from scipy.spatial.transform import Rotation as R

class ImgLoader:
    def __init__(self, path):
        self.path = Path(path)

    def load(self, idx, fmt_str):
        img = cv.imread(str(self.path.joinpath(fmt_str.format(idx))))
        return img

class PoseLoader:
    def __init__(self, path) -> None:
        self.path = Path(path)

    def load_pose_pnp(self, idx):
        """
        Load computed poses from Tien's code
        Data should be "camera_poses_pnp.npy" & "good_pose_pnp.npy"
        """
        # import pdb; pdb.set_trace()
        camera_pose = np.load(str(self.path.joinpath("camera_poses_pnp.npy")))
        is_good = np.load(str(self.path.joinpath("good_pose_pnp.npy")))
        if not is_good[idx]:
            return None
        else:
            pose = camera_pose[idx] # 3 x 4 matrix
            return R.from_matrix(pose[:, :3]), pose[:, 3:].reshape(3, 1)


    def load_pose_colmap(self, idx, fmt_str):
        """
        Load computed poses from colmap
        Expect camera pose to be extracted beforehand with process_images_binary() from colmap_read_model.py
        data should be "colmap_poses.npy" 
        """
        colmap_pose = np.load(str(self.path.joinpath("colmap_poses.pkl")), allow_pickle=True)
        # import pdb; pdb.set_trace()
        pose = colmap_pose[fmt_str.format(idx)]
        quat_colmap = pose[0] # quat: w, x, y, z
        quat = np.r_[quat_colmap[1:4], quat_colmap[0]]
        return R.from_quat(quat), pose[1].reshape(3, 1)


def feat_match(img1, img2):
    """SIFT feature matching with ratio test

    Args:
        img1 
        img2

    Returns:
        pts1, pts2: matched keypoints
    """
    sift = cv.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2

