import os
import argparse
import yaml
from typing import List


parser = argparse.ArgumentParser()
parser.add_argument('--log_list_path', type=str,
                    default='/home/wei/Coding/mankey/config/mugs_20190221.txt',
                    help='The full path to specific_logs.txt, '
                         'which contains a list of logs that annotation is required')
parser.add_argument('--num_keypoints', type=int, default=3,
                    help='The number of keypoints. The name can be modified later')
parser.add_argument('--annotation_save_dir', type=str,
                    default='annotations/keypoints/mugs',
                    help='The field requested by director')
parser.add_argument('--object_type', type=str, default='mug',
                    help='Which type of object is being annotated')
parser.add_argument('--annotation_type', type=str, default='mug_3_keypoint',
                    help='The name of annotation')
parser.add_argument('--request_save_path', type=str, default='annotation_request.yaml',
                    help='Where to save the request yaml file')
args = parser.parse_args()


def main():
    # Check the existence of logs
    log_list_path: str = args.log_list_path
    assert os.path.exists(log_list_path)

    # Get a list of logs
    # Only logs_proto is supported
    scene_list: List[str] = []
    with open(log_list_path, 'r') as log_list_file:
        lines = log_list_file.read().split('\n')
        for line in lines:
            if len(line) == 0:
                continue
            # Check the first elements to make sure logs_proto
            if line[0:11] != 'logs_proto/':
                raise RuntimeError('Only logs_proto is supported for now')
            scene_list.append(line[11:])

    # Constrcut the yaml map
    request_map = dict()
    request_map['save_dir'] = args.annotation_save_dir
    request_map['logs_dir'] = 'logs_proto'
    request_map['object_type'] = args.object_type
    request_map['annotation_type'] = args.annotation_type
    request_map['scene_list'] = scene_list

    # The keypoint name is just placeholder
    keypoint_name_list: List[str] = []
    for i in range(args.num_keypoints):
        keypoint_name_list.append('keypoint-%d' % i)
    request_map['keypoint_names'] = keypoint_name_list

    # Save the map to yaml file
    with open(args.request_save_path, 'w') as yaml_file:
        yaml.dump(request_map, yaml_file)


if __name__ == '__main__':
    main()
