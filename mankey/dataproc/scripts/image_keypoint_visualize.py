import argparse
import os
import yaml
import random
import cv2
from typing import List


"""
The script is used to visualize the keypoint produced by
image_keypoint.py in the image frame.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--processed_log_path',
                    type=str, default='/home/wei/data/pdc/logs_proto/2018-05-14-22-10-53/',
                    help='The log (scene) that the scene_keypoint.yaml should lives in its processed/image folder')
parser.add_argument('--frame_idx',
                    type=int, default=2182,
                    help='The frame number to be visualize, If negative, select random one.')
args = parser.parse_args()


# A list of directory
scene_data_root = args.processed_log_path
raw_image_root = os.path.join(scene_data_root, 'processed/images')


def draw_image_keypoint(image, keypoint_xy_depth: List[int]):
    img_clone = image.copy()
    cv2.circle(img_clone, center=(keypoint_xy_depth[0], keypoint_xy_depth[1]), radius=10, color=(255, 255, 0))
    cv2.imshow('image', img_clone)
    cv2.waitKey(0)


def image_visualize(image_map):
    # Get the rgb map
    rgb_name = image_map['rgb_image_filename']
    rgb_path = os.path.join(raw_image_root, rgb_name)
    rgb_img = cv2.imread(rgb_path)

    # Load keypoint
    keypoint_pixel_xy_depth = image_map['keypoint_pixel_xy_depth']
    for i in range(len(keypoint_pixel_xy_depth)):
        pixel_xy_depth = keypoint_pixel_xy_depth[i]
        draw_image_keypoint(rgb_img, keypoint_xy_depth=pixel_xy_depth)


def scene_visualize_random(keypoint_config_map):
    """
    The visualize function for the scene
    Process each image one by one
    :param keypoint_config_map: The config map provide by scene_keypoint.yaml
    :return: None
    """
    image_map = random.choice(list(keypoint_config_map.values()))
    image_visualize(image_map)


def main():
    keypoint_config_path = os.path.join(raw_image_root, 'scene_keypoint.yaml')
    assert os.path.exists(keypoint_config_path)
    keypoint_config_map = yaml.load(open(keypoint_config_path, 'r'))

    # Get the frame number and do visualize
    frame_idx = args.frame_idx
    if frame_idx < 0:
        scene_visualize_random(keypoint_config_map)
    else:
        assert frame_idx in keypoint_config_map
        image_visualize(keypoint_config_map[frame_idx])


if __name__ == '__main__':
    main()
