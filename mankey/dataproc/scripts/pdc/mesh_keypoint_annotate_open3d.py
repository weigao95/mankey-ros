import os
import argparse

"""
Sugar script to process the annotation of pdc data
"""
parser = argparse.ArgumentParser()
parser.add_argument('--processed_log_path',
                    type=str,
                    default='/home/wei/data/pdc/logs_proto/2018-05-14-22-10-53/',
                    help='The log (scene) whose mesh corresponded to keypoint_yaml')
args = parser.parse_args()

# The root directory
scene_data_root = args.processed_log_path


def main():
    mesh_path = os.path.join(scene_data_root, 'processed/fusion_mesh.ply')
    output_path = os.path.join(scene_data_root, 'processed/keypoint.yaml')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
    exe_file = os.path.join(dir_path, 'mesh_keypoint_annotate.py')
    command = 'python %s --mesh_path %s --output_yaml %s' % (exe_file, mesh_path, output_path)
    print(command)
    os.system(command)


if __name__ == '__main__':
    main()
