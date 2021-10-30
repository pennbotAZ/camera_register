"""
Test relative pose between two camera poses

steps:
1. compute relative pose
2. detect matched feature points
3. draw epipolar line
Ideally epipolar line should pass corresponding feature point in the second view
"""

import cv2 as cv
import numpy as np

from utils import PoseLoader, ImgLoader, feat_match

def relative_pose(r1, t1, r2, t2):
    """
    Given two poses (r1, t1), (r2, t2); 
    rotations are scipy.spatial.transform.Rotation
    
    Compute the relative transformation (R, t) 
    """
    r = r2 * r1.inv()
    t = r2.as_matrix() @ (-r1.inv().as_matrix() @ t1.reshape(3, 1)) + t2.reshape(3, 1)
    return r, t

def drawlines(img1, img2, lines, pts1, pts2):
    """img1 - image on witch we draw the epilines for the points in img2
       lines - corresponding epilines"""
    r, c = img1.shape[:2]
    if len(img1.shape) == 3:
        # img is color image to begin with 
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[0]/r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0]*c/r[1])])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
        
        # cv.imshow('test', img1)
        # cv.waitKey(1)
    return img1, img2

def skew_symm(a1, a2, a3):
    return np.array([[0, -a3, a2],
                    [a3, 0, -a1],
                    [-a2, a1, 0]])


def test_pose_epi(K, idx1, idx2, img_path, pose_path, mode, fmt_str=None):
    img_loader = ImgLoader(img_path)
    pose_loader = PoseLoader(pose_path)
    if mode == 'pnp':
        pose1, pose2 = pose_loader.load_pose_pnp(idx1), pose_loader.load_pose_pnp(idx2)
    elif mode == 'colmap':
        assert fmt_str is not None
        pose1, pose2 = pose_loader.load_pose_colmap(idx1, fmt_str), pose_loader.load_pose_colmap(idx2, fmt_str)
    
    img1, img2 = img_loader.load(idx1, fmt_str), img_loader.load(idx2, fmt_str)
    import pdb; pdb.set_trace()
    r1, t1 = pose1
    r2, t2 = pose2
    rel_pose = relative_pose(r1, t1, r2, t2)
    pts1, pts2 = feat_match(img1, img2)
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))[..., None]
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))[..., None]
    r = rel_pose[0].as_matrix()
    t = rel_pose[1]
    E = skew_symm(t[0].item(), t[1].item(), t[2].item()) @ r
    # recover the fundamental matrix
    F = np.linalg.inv(K.T) @ E @ np.linalg.inv(K)
    lines1 = F @ pts2_homo # epipolar line in image 1
    lines1 = lines1.reshape(-1, 3)
    res1, res2 = drawlines(img1, img2, lines1, pts1, pts2)
    return res1, res2


if __name__ == '__main__':
    K = np.array([[829.71, 0, 958.63],
             [0, 828.63, 538.28],
             [0, 0, 1]])
    # img_path = '/home/shzhou2/project/colmap/frames'
    # pose_path = '/home/shzhou2/project/colmap/sparse/0/'
    # res1, res2 = test_pose(K, 31, 34, img_path, pose_path, 'colmap', '{:05d}.png')

    img_path = '/home/shzhou2/project/colmap/frames'
    pose_path = '/home/shzhou2/project/colmap/Ego4DLocalization/data/egovideo/poses_reloc'
    res1, res2 = test_pose_epi(K, 18, 19, img_path, pose_path, 'pnp', '{:05d}.png')
    cv.imwrite('epi1.png', res1)
    cv.imwrite('epi2.png', res2)