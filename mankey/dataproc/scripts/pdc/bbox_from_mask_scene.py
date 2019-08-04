import os
import argparse
import yaml
import cv2
from mankey.utils.imgproc import mask2bbox


"""
Given the binary foreground, this script attempt to compute
the TIGHT bounding box of the object in the image. Note that this
script will take input from pose_data.yaml, add element to it and
save the map as another yaml file contains information in pose_data
"""
parser = argparse.ArgumentParser()
parser.add_argument('--scene_root_path', type=str,
                    help='Full path/to/2018-05-14.../')
parser.add_argument('--pose_yaml_path', type=str,
                    help='Full path to input yaml file, must contains camera pose. ')
parser.add_argument('--output_yaml_path', type=str,
                    help='Full path to output yaml file')
args = parser.parse_args()


def process_scene(scene_keypoint_map, scene_log_root: str, yaml_out_path: str):
    # Iterate over all image in the scene
    image_keys = scene_keypoint_map.keys()
    for img_key in image_keys:
        image_map = scene_keypoint_map[img_key]
        filename_str = image_map['depth_image_filename'][0:6]
        mask_name = filename_str + '_mask.png'
        mask_root = os.path.join(scene_log_root, 'processed/image_masks')
        mask_path = os.path.join(mask_root, mask_name)

        # Load the mask
        mask_img = cv2.imread(mask_path)
        assert mask_img is not None
        top_left, bottom_right = mask2bbox(mask_img)

        # Save to map
        image_map['bbox_top_left_xy'] = [top_left.x, top_left.y]
        image_map['bbox_bottom_right_xy'] = [bottom_right.x, bottom_right.y]

    # Save the output as path
    with open(yaml_out_path, 'w') as out_file:
        yaml.dump(scene_keypoint_map, out_file)


def main():
    # Check the existence of path
    assert os.path.exists(args.scene_root_path) and os.path.isdir(args.scene_root_path)
    assert os.path.exists(args.pose_yaml_path)

    # Load the map and close the file
    pose_data_yaml_file = open(args.pose_yaml_path, 'r')
    pose_data_yaml = yaml.load(pose_data_yaml_file)
    pose_data_yaml_file.close()

    # Process it
    process_scene(pose_data_yaml, args.scene_root_path, args.output_yaml_path)


if __name__ == '__main__':
    main()
