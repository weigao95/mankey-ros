import os
import argparse


"""
Given keypoint annotation by open3d, this script transfer the annotation
to the required info used for training. The resultant yaml file is save to
the processed/ subdirectory for each scene.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--log_root_path',
                    type=str, default='/home/wei/data/pdc/logs_proto',
                    help='The root path that contains lots of scene recorded at different time')
parser.add_argument('--scene_mesh_keypoint_path', type=str, default='keypoint.yaml',
                    help='The keypoint annotation input for each scene. '
                         'Note that this file will be relative to ${log_root_path}/2018-05-14.../processed/ folder')
parser.add_argument('--scene_image_keypoint_output_path', type=str, default='shoe_6_keypoint_image.yaml',
                    help='The image keypoint for each scene. '
                         'Note that this file will be relative to ${log_root_path}/2018-05-14.../processed/ folder')
args = parser.parse_args()


def main():
    # Get the exe file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    bbox_exe_file = os.path.join(dir_path, 'bbox_from_mask_all.py')
    dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
    keypoint_exe_file = os.path.join(dir_path, 'keypoint_mesh2img.py')

    # the tmp file for keypoint image projection
    image_keypoint_tempfile = 'scene_bbox_keypoint_tmp.yaml'

    # Get the root keypoint
    log_root: str = args.log_root_path
    processed_counter = 0
    skip_counter = 0
    for folder in os.listdir(log_root):
        scene_log_path = os.path.join(log_root, folder)
        scene_processed_path = os.path.join(scene_log_path, 'processed')
        if os.path.isdir(scene_log_path):
            keypoint_yaml_path = os.path.join(scene_processed_path, args.scene_mesh_keypoint_path)
            output_yaml_path = os.path.join(scene_processed_path, image_keypoint_tempfile)
            if not os.path.exists(keypoint_yaml_path):
                print('Keypoint in %s is not founded!' % scene_log_path)
                skip_counter += 1
                continue

            # Run the python command
            command = 'python %s --keypoint_yaml_path %s --scene_log_path %s --output_yaml_path %s' \
                      % (keypoint_exe_file, keypoint_yaml_path, scene_log_path, output_yaml_path)
            print(command)
            os.system(command)
            processed_counter += 1

    # Also annotate the bounding box
    bbox_command = 'python %s --log_root_path %s --pose_yaml_filename %s --output_yaml_name %s' \
                   % (bbox_exe_file, log_root, image_keypoint_tempfile, args.scene_image_keypoint_output_path)
    print(bbox_command)
    os.system(bbox_command)

    # The cleanup
    for folder in os.listdir(log_root):
        scene_log_path = os.path.join(log_root, folder)
        scene_processed_path = os.path.join(scene_log_path, 'processed')
        if os.path.isdir(scene_log_path):
            tmp_yaml_path = os.path.join(scene_processed_path, image_keypoint_tempfile)
            if os.path.exists(tmp_yaml_path):
                os.remove(tmp_yaml_path)

    # Some info
    print('Processed logs (scenes) %d, while %d scene does not have keypoint.' % (processed_counter, skip_counter))


if __name__ == '__main__':
    main()
