import os
import argparse


"""
The script that calls bbox_from_mask_scene.py
for all the scene in the log_root
"""
parser = argparse.ArgumentParser()
parser.add_argument('--log_root_path',
                    type=str, default='/home/wei/data/pdc/logs_proto',
                    help='The root path that contains lots of scene recorded at different time')
parser.add_argument('--pose_yaml_filename', type=str, default='pose_data.yaml',
                    help='The input yaml which shound contain all info in pose_data.yaml. '
                         'Note that this fill will be relative to ${log_root_path}/2018-05-14.../processed/ folder')
parser.add_argument('--output_yaml_name', type=str, default='pose_bbox_data.yaml',
                    help='The saved directory with bounding box data. '
                         'Note that this fill will be relative to ${log_root_path}/2018-05-14.../processed/ folder')
args = parser.parse_args()


def main():
    # Get the exe file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    scene_bbox_exe_file = os.path.join(dir_path, 'bbox_from_mask_scene.py')

    # Get the root keypoint
    counter = 0
    log_root = args.log_root_path
    for folder in os.listdir(log_root):
        scene_log_path = os.path.join(log_root, folder)
        print('Processing: ', scene_log_path, ' the %d th item.' % (counter))
        counter += 1
        if not os.path.isdir(scene_log_path):
            continue

        # The pose data
        scene_processed_root = os.path.join(scene_log_path, 'processed')
        pose_data_yaml_path = os.path.join(scene_processed_root, args.pose_yaml_filename)
        if not os.path.exists(pose_data_yaml_path):
            print('Keypoint in %s is not founded!' % scene_log_path)
            continue

        # Get the command
        pose_bbox_data_yaml_path = os.path.join(scene_processed_root, args.output_yaml_name)
        bbox_command = 'python %s --scene_root_path %s --pose_yaml_path %s --output_yaml_path %s' \
                       % (scene_bbox_exe_file, scene_log_path, pose_data_yaml_path, pose_bbox_data_yaml_path)
        os.system(bbox_command)


if __name__ == '__main__':
    main()
