import os
import argparse
import yaml
from typing import List, Set
import open3d
import numpy as np
from mankey.utils.cloudproc import draw_all_keypoints, draw_single_keypoint


"""
The script used to visualize the keypoint annotated by director
"""
parser = argparse.ArgumentParser()
parser.add_argument('--annotation_path', type=str,
                    default='/home/wei/data/pdc/annotations/keypoints/shoes/scene_2019-'
                            '02-12-16-39-00_date_2019-02-12-23-49-30.yaml',
                    help='The directory that contains all the annotation files')
parser.add_argument('--annotation_type', type=str, default='shoe_standard',
                    help='Which type of annotation need to be processed')
parser.add_argument('--log_root_path', type=str, default='/home/wei/data/pdc/logs_proto',
                    help='Full path/to/logs_proto')
parser.add_argument('--one_by_one', type=bool, default=True, help='Visualize the keypoint one-by-one')
args = parser.parse_args()


def get_keypoint_name_list(annotation_dict) -> List[str]:
    """
    Given the first annotation dict, return the list of keypoints
    that will be used for projection.
    Used to ensure that the order the keypoint
    :param annotation_dict: The dict from the annotation yaml
    :return: A list of keypoint in the dictionary
    """
    assert 'keypoints' in annotation_dict
    keypoint_name_list: List[str] = []
    for keypoint_name in annotation_dict['keypoints']:
        keypoint_name_list.append(keypoint_name)

    # OK
    return keypoint_name_list


def visualize_scene_keypoint(annotation_dict, keypoint_name_list: List[str]):
    # Get all the keypoint
    keypoint_3d_position: List[List[float]] = []
    annotation_keypoints_dict = annotation_dict['keypoints']
    for keypoint_name in keypoint_name_list:
        assert keypoint_name in annotation_keypoints_dict
        keypoint_pos = annotation_keypoints_dict[keypoint_name]['position']
        keypoint_3d_position.append([float(keypoint_pos[0]), float(keypoint_pos[1]), float(keypoint_pos[2])])

    # Get the scene root and ply file
    assert 'scene_name' in annotation_dict
    scene_name: str = annotation_dict['scene_name']
    scene_root_path = os.path.join(args.log_root_path, scene_name)
    scene_processed_path = os.path.join(scene_root_path, 'processed')
    fusion_mesh_path: str = os.path.join(scene_processed_path, 'fusion_mesh.ply')
    assert os.path.exists(fusion_mesh_path)

    # Get the open3d point cloud and draw it
    point_cloud = open3d.read_point_cloud(fusion_mesh_path)
    n_keypoint: int = len(keypoint_3d_position)
    if args.one_by_one:
        for i in range(n_keypoint):
            draw_single_keypoint(point_cloud, keypoint_3d_position[i])
    else:
        keypoint_np = np.zeros(shape=[3, n_keypoint])
        for i in range(n_keypoint):
            keypoint_np[0, i] = keypoint_3d_position[i][0]
            keypoint_np[1, i] = keypoint_3d_position[i][1]
            keypoint_np[2, i] = keypoint_3d_position[i][2]
        draw_all_keypoints(point_cloud, keypoint_np)


def main():
    # Check the directory
    annotation_path: str = args.annotation_path
    assert os.path.exists(annotation_path) and annotation_path.endswith('.yaml')

    # Some global meta info for given annotation_type
    keypoint_name_list: List[str] = []
    processed_scene: Set[str] = set()

    # Open the yaml file and get the map
    annotation_yaml_file = open(annotation_path, 'r')
    annotation_yaml_list = yaml.load(annotation_yaml_file)
    annotation_yaml_file.close()

    # Iterate over the list
    for annotation_dict in annotation_yaml_list:
        # Check the annotation type
        if 'annotation_type' not in annotation_dict or annotation_dict['annotation_type'] != args.annotation_type:
            continue

        # Init the global meta
        if len(keypoint_name_list) == 0:
            keypoint_name_list = get_keypoint_name_list(annotation_dict)

        # Update the processed scene
        assert 'scene_name' in annotation_dict
        scene_name: str = annotation_dict['scene_name']
        if scene_name in processed_scene:
            print('The scene %s is annotated twice for %s' % (scene_name, args.annotation_type))
            continue
        else:
            processed_scene.add(scene_name)

        # Do visualization
        visualize_scene_keypoint(annotation_dict, keypoint_name_list)


if __name__ == '__main__':
    main()
