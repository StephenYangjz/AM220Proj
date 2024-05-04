import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from params_proto import Proto, ParamsProto, Flag


class Args(ParamsProto):
    # for depth: mm to m
    scale_factor = 1.0  # 0.001
    # relative to $DATASETS
    dataset_root = Proto(env="DATASETS", dtype=Path)
    dataset_name = "curb_"

    pixel_downsample = 16
    image_skip = 8

    viz = Flag("visualize using open3D")
    save = True


def unormalize_to_mm(depth):
    depth = depth.astype(np.uint16)
    dnorm = depth / 255.0
    dnorm = 1 - dnorm

    near = 0.45
    far = 7.0

    # inv_D = dnorm * (1 / near - 1 / far) + 1 / far
    # return 1 / inv_D

    dtrans = dnorm * (far - near) + near
    return dtrans


def r2z(r, fov, h, w):
    """Convert range map to metric depth.

    :param r: range map, in Size(H, W)
    :param fov: verticle field of view
    :param h: height of the image
    :param w: width of the image
    """
    f = h / 2 / np.tan(fov / 2)
    x = (np.arange(w) - w / 2) / f
    y = (np.arange(h) - h / 2) / f
    xs, ys = np.meshgrid(x, y, indexing="xy")

    r_scale = np.sqrt(1 + xs**2 + ys**2)
    z = r / r_scale
    return z, xs, ys


def main(**deps):
    Args._update(deps)

    print(Args.dataset_root)

    prefix = Args.dataset_root / Args.dataset_name

    tf_fname = prefix / "transforms.json"
    output_path = str(prefix / "pointcloud.ply")

    with open(tf_fname) as f:
        dataset = json.load(f)

    # load tfs, depths
    all_tfs = []
    all_depths = []

    for frame in dataset["frames"]:
        depth_file = str(prefix / frame["depth"]) + ".jpg"
        matrix = np.array(frame["transform_matrix"])

        all_tfs.append(matrix)

        data = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        data = unormalize_to_mm(data)

        data *= Args.scale_factor

        # data, *_ = r2z(data, dataset["camera_angle_x"], *data.shape)

        all_depths.append(data)

    w, h = data.shape
    fovx = dataset["camera_angle_x"]
    f = w / (2 * np.tan(fovx / 2))
    cx = w / 2
    cy = h / 2

    points = []

    us = np.arange(w) + 0.5
    vs = np.arange(h) + 0.5
    us, vs = np.meshgrid(us, vs, indexing="xy")

    us.max(), vs.max()
    us_c = (us - cx) / f
    vs_c = (vs - cy) / f

    for depth, tf in zip(all_depths[:: Args.image_skip], all_tfs[:: Args.image_skip]):
        xs = (us_c * depth).flatten()[:: Args.pixel_downsample]
        ys = (vs_c * depth).flatten()[:: Args.pixel_downsample]
        zs = (depth).flatten()[:: Args.pixel_downsample]

        _h = np.ones_like(xs)

        # coordinate swap
        points_to_cam = np.stack([xs, -ys, -zs, _h], axis=-1).copy()

        points_to_world = points_to_cam @ tf.T
        points.append(points_to_world[..., :3])

    points = np.concatenate(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if Args.save:
        o3d.io.write_point_cloud(output_path, pcd)

    if Args.viz:
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    from pathlib import Path

    main(dataset_root=Path("/Users/alanyu/Downloads"), dataset_name="test/gen_dataset_old", viz=True)
