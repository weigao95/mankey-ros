import os
import argparse
import yaml
from typing import List, Set


"""
Given keypoint annotation by director, this script transfer the annotation
to the required info used for training. The resultant yaml file is save to
the processed/ subdirectory for each scene.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--annotation_dir', type=str,
                    default='/home/wei/data/pdc/annotations/keypoints/heels',
                    help='The directory that contains all the annotation files')
parser.add_argument('--annotation_type', type=str, default='shoe_standard',
                    help='Which type of annotation need to be processed')
parser.add_argument('--save_relative_path', type=str, default='shoe_6_keypoint_image.yaml',
                    help='The path to save the image keypoint yaml file. '
                         'Relative to the log_root/2018-11-23-../processed')
parser.add_argument('--log_root_path', type=str, default='/home/wei/data/pdc/logs_proto',
                    help='Full path/to/logs_proto')
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


def process_annotation_dict(annotation_dict, keypoint_name_list: List[str]):
    """
    Given a standalone annotation, make the image annotation at given scene
    The scene is only annotated the first time.
    :param annotation_dict:
    :param keypoint_name_list:
    :return:
    """
    # Get the exe file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    bbox_exe_file = os.path.join(dir_path, 'bbox_from_mask_scene.py')
    dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
    keypoint_mesh2img_exe_file = os.path.join(dir_path, 'keypoint_mesh2img.py')

    # Get the scene root
    assert 'scene_name' in annotation_dict
    scene_name: str = annotation_dict['scene_name']
    scene_root_path = os.path.join(args.log_root_path, scene_name)
    scene_processed_path = os.path.join(scene_root_path, 'processed')
    assert os.path.exists(scene_root_path) and os.path.isdir(scene_root_path)
    assert os.path.exists(scene_processed_path) and os.path.isdir(scene_processed_path)

    # Check the existence of keypoints
    annotation_keypoints_dict = annotation_dict['keypoints']
    for keypoint_name in keypoint_name_list:
        assert keypoint_name in annotation_keypoints_dict

    # Create a tmp keypoint.yaml
    tmp_keypoint_yaml_name = 'director_keypoint_tmp.yaml'
    tmp_keypoint_yaml_path = os.path.join(scene_processed_path, tmp_keypoint_yaml_name)
    with open(tmp_keypoint_yaml_path, 'w') as tmp_keypoint_yaml_file:
        keypoint_3d_position = []
        for keypoint_name in keypoint_name_list:
            keypoint_pos = annotation_keypoints_dict[keypoint_name]['position']
            keypoint_3d_position.append([float(keypoint_pos[0]), float(keypoint_pos[1]), float(keypoint_pos[2])])

        # Save to yaml file
        tmp_keypoint_yaml_map = dict()
        tmp_keypoint_yaml_map['keypoint_world_position'] = keypoint_3d_position
        yaml.dump(tmp_keypoint_yaml_map, tmp_keypoint_yaml_file)

    # Do mesh3image keypoint projection
    tmp_imgkeypoint_yaml_name = 'director_img_keypoint_tmp.yaml'
    tmp_imgkeypoint_yaml_path = os.path.join(scene_processed_path, tmp_imgkeypoint_yaml_name)
    # Run the python command
    mesh2img_command = 'python %s --keypoint_yaml_path %s --scene_log_path %s --output_yaml_path %s' \
              % (keypoint_mesh2img_exe_file, tmp_keypoint_yaml_path, scene_root_path, tmp_imgkeypoint_yaml_path)
    print(mesh2img_command)
    os.system(mesh2img_command)

    # Compute bounding box
    img_keypoint_output_path = os.path.join(scene_processed_path, args.save_relative_path)
    bbox_command = 'python %s --scene_root_path %s --pose_yaml_path %s --output_yaml_path %s' \
                   % (bbox_exe_file, scene_root_path, tmp_imgkeypoint_yaml_path, img_keypoint_output_path)
    print(bbox_command)
    os.system(bbox_command)

    # Clean up tmp file
    if os.path.exists(tmp_keypoint_yaml_path):
        os.remove(tmp_keypoint_yaml_path)
    if os.path.exists(tmp_imgkeypoint_yaml_path):
        os.remove(tmp_imgkeypoint_yaml_path)


def main():
    # Check the directory
    annotation_dir = args.annotation_dir
    assert os.path.exists(annotation_dir) and os.path.isdir(annotation_dir)

    # Some global meta info for given annotation_type
    keypoint_name_list: List[str] = []
    processed_scene: Set[str] = set()

    # Iterate over the yaml file inside
    for annotation_yaml_file in os.listdir(annotation_dir):
        # Get the yaml path
        annotation_yaml_path = os.path.join(annotation_dir, annotation_yaml_file)
        if not annotation_yaml_path.endswith('.yaml'):
            continue

        # Open the yaml file and get the map
        annotation_yaml_file = open(annotation_yaml_path, 'r')
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

            assert 'scene_name' in annotation_dict
            scene_name: str = annotation_dict['scene_name']
            if scene_name in processed_scene:
                print('The scene %s is annotated twice for %s' % (scene_name, args.annotation_type))
                continue
            else:
                processed_scene.add(scene_name)

            # The annotation type is ok, need processing
            process_annotation_dict(annotation_dict, keypoint_name_list)


if __name__ == '__main__':
    main()
