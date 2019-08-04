import argparse
import yaml
import open3d
import numpy as np
from typing import List
from mankey.utils.cloudproc import draw_all_keypoints, draw_single_keypoint


"""
The script used to visualize keypoint specification from yaml file, typically
produced from scripts/mesh_keypoint_annotate.py
In the future, we should switch to director annotator
"""
parser = argparse.ArgumentParser()
parser.add_argument('--keypoint_yaml_path',
                    type=str, default='/home/wei/data/pdc/logs_proto/2018-05-15-00-08-46/processed/keypoint.yaml',
                    help='The annotated keypoint using mesh_keypoint_annotate.py')
parser.add_argument('--one_by_one', type=bool, default=False,
                    help='Visualize keypoint one-by-one')

# The parsed argument
args = parser.parse_args()


def main():
    keypoint_path: str = args.keypoint_yaml_path
    keypoint_datamap = None
    with open(keypoint_path, 'r') as file:
        try:
            keypoint_datamap = yaml.load(file)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)

    # Process the map
    assert keypoint_datamap is not None
    mesh_path: str = keypoint_datamap['mesh_path']
    keypoint_index: List[int] = keypoint_datamap['keypoint_index']
    keypoint_3d_position: List[List[float]] = keypoint_datamap['keypoint_world_position']
    pcd = open3d.read_point_cloud(mesh_path)

    # Depends on visualization type
    if args.one_by_one:
        for i in range(len(keypoint_index)):
            draw_single_keypoint(pcd, keypoint_3d_position[i])
    else:
        n_keypoint = len(keypoint_3d_position)
        keypoint_np = np.zeros(shape=[3, n_keypoint])
        for i in range(n_keypoint):
            keypoint_np[0, i] = keypoint_3d_position[i][0]
            keypoint_np[1, i] = keypoint_3d_position[i][1]
            keypoint_np[2, i] = keypoint_3d_position[i][2]
        draw_all_keypoints(pcd, keypoint_np)


if __name__ == '__main__':
    main()
