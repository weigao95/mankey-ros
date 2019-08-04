import open3d
import numpy as np
# from typing import List


# Get open3d point cloud from depth
def create_pointcloud_from_depth(
        depth_path,  # type: str,
        focal_x,  # type: float,
        focal_y,  # type: float,
        principal_x,  # type: float,
        principal_y,  # type: float
):
    depth_image = open3d.read_image(depth_path)
    intrinsic = open3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(640, 480, focal_x, focal_y, principal_x, principal_y)
    return open3d.geometry.create_point_cloud_from_depth_image(depth_image, intrinsic)


# Draw keypoint as small sphere
def get_keypoint_sphere(
        keypoint,  # type: List[float],
        radius=0.01):
    """
    Create a sphere mesh used for keypoint visualization
    :param keypoint: length 3 list (x, y, z) for the center of the mesh
    :param radius: The radius of the mesh
    :return:
    """
    assert len(keypoint) is 3
    mesh_sphere = open3d.create_mesh_sphere(radius=radius)
    translation = np.identity(4)
    translation[0, 3] = keypoint[0]
    translation[1, 3] = keypoint[1]
    translation[2, 3] = keypoint[2]
    mesh_sphere.transform(translation)
    return mesh_sphere


def draw_all_keypoints(
        point_cloud,
        keypoint,  # type: np.ndarray,
        keypoint_size=0.01):
    """
    Draw all the keypoint associated with the point_cloud (mesh)
    :param point_cloud: open3d point cloud
    :param keypoint: np.ndarray in the shape of (3, n_keypoint)
    :param keypoint_size: the radius of the keypoint
    :return: None
    """
    n_keypoint = keypoint.shape[1]
    geometries = []
    geometries.append(point_cloud)
    for i in range(n_keypoint):
        # Type conversion
        keypoint_i = []
        for k in range(3):
            keypoint_i.append(keypoint[k, i])

        # Insert into geometries
        geometries.append(get_keypoint_sphere(keypoint_i, keypoint_size))

    # The drawing command
    open3d.draw_geometries(geometries)


def draw_single_keypoint(
        point_cloud,
        keypoint,  # type: List[float],
        keypoint_size=0.01):
    """
    Draw a keypoint with point cloud that this
    keypoint belongs to.
    :param point_cloud: open3d point cloud
    :param keypoint: length 3 list (x, y, z) for the keypoint
    :param keypoint_size: the radius of the keypoint
    :return: None
    """
    assert len(keypoint) is 3
    mesh_sphere = get_keypoint_sphere(keypoint, radius=keypoint_size)
    open3d.draw_geometries([mesh_sphere, point_cloud])