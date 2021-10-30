"""
Test a single camera pose by projecting given point cloud
to the camera view
"""
import open3d as o3d
import cv2
import numpy as np

from utils import PoseLoader, ImgLoader


def gen_pcd_from_kinect(img, depth, K):
    # To get color point clould from Kinect RGBD
    color_raw = o3d.io.read_image(img)
    depth_raw = o3d.io.read_image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)
    h, w = img.shape[:2]
    f1, f2, x, y = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    K_ = o3d.camera.PinholeCameraIntrinsic()
    K_.set_intrinsics(w, h, f1, f2, x, y)


# To get color in the point clould file for matterpak, rename it
# from "xyz" file to "xyzrgb" file
def test_pose_pcd(K,
                  idx,
                  pcd_file,
                  img_path,
                  pose_path,
                  mode,
                  dist=np.zeros((4, 1)),
                  fmt_str=None):
    img_loader = ImgLoader(img_path)
    pose_loader = PoseLoader(pose_path)
    if mode == 'pnp':
        pose = pose_loader.load_pose_pnp(idx)
    elif mode == 'colmap':
        assert fmt_str is not None
        pose = pose_loader.load_pose_colmap(idx, fmt_str)
    img = img_loader.load(idx, fmt_str)
    pcd = o3d.io.read_point_cloud(pcd_file)
    pts_3d = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)[...,::-1]  # load in BGR
    r, t = pose
    r = r.as_rotvec()
    imgpts, _ = cv2.projectPoints(pts_3d, r, t, K, dist)
    imgpts = np.array(imgpts).squeeze().astype(int)
    mask = np.logical_and(
        np.logical_and(imgpts[:, 0] > 0, imgpts[:, 0] < img.shape[1]),
        np.logical_and(imgpts[:, 1] > 0, imgpts[:, 1] < img.shape[0]))
    # img in BGR order
    img[imgpts[mask, 1], imgpts[mask, 0], :] = colors[mask]
    return img


if __name__ == '__main__':
    coeff = 1
    matK = np.array([[
        coeff * 5.599287999999999101e+02,
        0.000000000000000000e+00,
        6.595000000000000000e+02,
    ],
                     [
                         0.000000000000000000e+00,
                         coeff * 5.599287999999999101e+02,
                         2.395000000000000000e+02
                     ],
                     [
                         0.000000000000000000e+00, 0.000000000000000000e+00,
                         1.000000000000000000e+00
                     ]])
    pcd_file = '/home/shzhou2/project/colmap/Ego4DLocalization/new_data/data/scan/matterpak/cloud.xyzrgb'
    res = test_pose_pcd(
        matK,
        43,
        pcd_file,
        img_path=
        '/home/shzhou2/project/colmap/Ego4DLocalization/new_data/data/egovideo/color',
        pose_path=
        '/home/shzhou2/project/colmap/Ego4DLocalization/new_data/data/egovideo/poses_reloc',
        mode='pnp',
        fmt_str='color_{:07d}.jpg')
    cv2.imwrite('projection_pcd.jpg', res)