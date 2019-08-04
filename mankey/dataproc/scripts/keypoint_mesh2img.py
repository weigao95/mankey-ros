import argparse
import os
import yaml
import numpy as np
from mankey.utils.transformations import quaternion_matrix, inverse_matrix
from typing import List


"""
The script used to transfer the keypoint annotation from
mesh to keypoint on images, given the pose and intrinsic of the camera
"""
parser = argparse.ArgumentParser()
parser.add_argument('--keypoint_yaml_path',
                    type=str, default='/home/wei/data/pdc/logs_proto/2018-05-14-22-10-53/processed/keypoint.yaml',
                    help='The full path for keypoint for a mesh model')
parser.add_argument('--scene_log_path',
                    type=str,
                    default='/home/wei/data/pdc/logs_proto/2018-05-14-22-10-53/',
                    help='The full log (scene) whose mesh corresponded to keypoint_yaml_path')
parser.add_argument('--output_yaml_path',
                    type=str, default='/home/wei/data/pdc/logs_proto/2018-05-14-22-10-53/processed/scene_keypoint.yaml',
                    help='The full output path for image level keypoint information. ')
args = parser.parse_args()

# The root for many level of directories
scene_data_root = args.scene_log_path
scene_processed_root = os.path.join(scene_data_root, 'processed')
raw_image_root = os.path.join(scene_processed_root, 'images')


# The path for camera config
camera_config_path = os.path.join(raw_image_root, 'camera_info.yaml')
assert os.path.exists(camera_config_path)
camera_config_map = yaml.load(open(camera_config_path, 'r'))
focal_x: float = camera_config_map['camera_matrix']['data'][0]
focal_y: float = camera_config_map['camera_matrix']['data'][4]
principal_x: float = camera_config_map['camera_matrix']['data'][2]
principal_y: float = camera_config_map['camera_matrix']['data'][5]


def point2pixel(keypoint_in_camera: np.ndarray) -> np.ndarray:
    """
    Given keypoint in camera frame, project them into image
    space and compute the depth value expressed in [mm]
    :param keypoint_in_camera: (4, n_keypoint) keypoint expressed in camera frame in meter
    :return: (3, n_keypoint) where (xy, :) is pixel location and (z, :) is depth in mm
    """
    assert len(keypoint_in_camera.shape) is 2
    n_keypoint: int = keypoint_in_camera.shape[1]
    xy_depth = np.zeros((3, n_keypoint), dtype=np.int)
    xy_depth[0, :] = (np.divide(keypoint_in_camera[0, :], keypoint_in_camera[2, :]) * focal_x + principal_x).astype(np.int)
    xy_depth[1, :] = (np.divide(keypoint_in_camera[1, :], keypoint_in_camera[2, :]) * focal_y + principal_y).astype(np.int)
    xy_depth[2, :] = (1000.0 * keypoint_in_camera[2, :]).astype(np.int)
    return xy_depth


def camera2world_from_map(camera2world_map) -> np.ndarray:
    """
    Get the transformation matrix from the storage map
    See ${processed_log_path}/processed/images/pose_data.yaml for an example
    :param camera2world_map:
    :return: The (4, 4) transform matrix from camera 2 world
    """
    # The rotation part
    camera2world_quat = [1, 0, 0, 0]
    camera2world_quat[0] = camera2world_map['quaternion']['w']
    camera2world_quat[1] = camera2world_map['quaternion']['x']
    camera2world_quat[2] = camera2world_map['quaternion']['y']
    camera2world_quat[3] = camera2world_map['quaternion']['z']
    camera2world_quat = np.asarray(camera2world_quat)
    camera2world_matrix = quaternion_matrix(camera2world_quat)

    # The linear part
    camera2world_matrix[0, 3] = camera2world_map['translation']['x']
    camera2world_matrix[1, 3] = camera2world_map['translation']['y']
    camera2world_matrix[2, 3] = camera2world_map['translation']['z']
    return camera2world_matrix


def world2camera_from_map(camera2world_map) -> np.ndarray:
    camera2world = camera2world_from_map(camera2world_map)
    return inverse_matrix(camera2world)


def get_keypoint_in_world(mesh_keypoint_map) -> np.ndarray:
    """
    Read the keypoint from the config and save them into an np.ndarray
    The keypoint is expressed in homogeneous coordinate (padding with 1)
    :param mesh_keypoint_map:
    :return:np.ndarray at (4, num_keypoints)
    """
    keypoint_in_world: List[List[float]] = mesh_keypoint_map['keypoint_world_position']
    keypoint_np = np.ones((4, len(keypoint_in_world)))
    offset: int = 0
    for keypoint in keypoint_in_world:
        for i in range(3):
            keypoint_np[i, offset] = keypoint[i]
        offset = offset + 1

    return keypoint_np


def process_scene_image(image_data_map, keypoint_in_world: np.ndarray):
    """
    The processor for an single image. Given the camera pose and intrinsic,
    compute the keypoint in camera frame and its pixel location, and store
    them into the image_data_map. Note that the keypoint might be out of the image.
    :param image_data_map:
    :param keypoint_in_world: (4, n_keypoint) numpy ndarray for keypoint in world frame
    :return: None
    """
    camera2world_map = image_data_map['camera_to_world']
    world2camera = world2camera_from_map(camera2world_map)

    # Do transform and save the keypoint in data map
    n_keypoints: int = keypoint_in_world.shape[1]
    keypoint_in_camera = world2camera.dot(keypoint_in_world)
    keypoint_in_camera_list = []
    for i in range(n_keypoints):
        keypoint_in_camera_list.append([
            float(keypoint_in_camera[0, i]),
            float(keypoint_in_camera[1, i]),
            float(keypoint_in_camera[2, i])]
        )
    image_data_map['3d_keypoint_camera_frame'] = keypoint_in_camera_list

    # Project the keypoint into pixel and depth
    pixel_xy_depth = point2pixel(keypoint_in_camera)
    pixel_xy_depth_list = []
    for i in range(n_keypoints):
        pixel_xy_depth_list.append([
            int(pixel_xy_depth[0, i]),
            int(pixel_xy_depth[1, i]),
            int(pixel_xy_depth[2, i])]
        )
    image_data_map['keypoint_pixel_xy_depth'] = pixel_xy_depth_list


def process_scene(pose_data_map, keypoint_in_world: np.ndarray, output_path: str):
    """
    The processor for scene which contains many images.
    Dispatch to image level processor and collect the result.
    :param pose_data_map: The data map loaded from pose yaml file
    :param keypoint_in_world: The data map loaded from keypoint yaml file
    :param output_path:
    :return: None
    """
    for image_idx in pose_data_map:
        image_data_map = pose_data_map[image_idx]
        process_scene_image(image_data_map, keypoint_in_world)

    # Save the output as path
    with open(output_path, 'w') as out_file:
        yaml.dump(pose_data_map, out_file)


def main():
    # Load the pose data
    pose_data_path: str = os.path.join(raw_image_root, 'pose_data.yaml')
    assert os.path.exists(pose_data_path)
    in_pose_file = open(pose_data_path, 'r')
    pose_data_map = yaml.load(in_pose_file)

    # Load the keypoint data
    keypoint_data_path = args.keypoint_yaml_path
    assert os.path.exists(keypoint_data_path)
    in_keypoint_file = open(keypoint_data_path, 'r')
    keypoint_data_map = yaml.load(in_keypoint_file)
    keypoint_in_world = get_keypoint_in_world(keypoint_data_map)

    # Do it
    output_path = args.output_yaml_path
    process_scene(pose_data_map, keypoint_in_world, output_path)


if __name__ == '__main__':
    main()
