import os
import open3d
import argparse
import numpy as np
import yaml


"""
Given a mesh in the file format of .ply/.obj, this script starts an interactive
window used for keypoint annotation, and save the annotated keypoint to a yaml file.
Please see the associating keypoint.yaml for an example output.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--mesh_path',
                    type=str, default='/home/wei/data/pdc/logs_proto/2018-05-14-22-10-53/processed/fusion_mesh.ply',
                    help='The full path to .ply mesh')
parser.add_argument('--output_yaml', type=str, default='keypoint.yaml',
                    help='Full path to the YAML file of selected keypoint')
args = parser.parse_args()


def pick_points(pcd):
    """
    http://www.open3d.org/docs/tutorial/Advanced/interactive_visualization.html
    :param pcd: open3d point cloud data
    :return: None
    """
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = open3d.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def load_point_cloud(mesh_path: str):
    """
    Currently open3d doesn't support obj file, thus we need some
    hacky solution here
    :param mesh_path: Full path to an ply/obj file
    :return:
    """
    if mesh_path.endswith('.ply'):
        return open3d.read_point_cloud(mesh_path)
    elif mesh_path.endswith('.obj'):
        raise NotImplementedError()
    else:
        raise RuntimeError('Unknown data format')


def main():
    mesh_path: str = args.mesh_path
    assert os.path.exists(mesh_path)
    pcd = load_point_cloud(mesh_path)
    picked_ids = pick_points(pcd)

    # Extract the keypoint in world
    point_cloud = np.asarray(pcd.points)
    keypoint_3d_position = []
    for point_id in picked_ids:
        point_in_world = point_cloud[point_id, :]
        keypoint_3d_position.append([float(point_in_world[0]), float(point_in_world[1]), float(point_in_world[2])])

    # Save them to a map
    data_map = dict()
    data_map['mesh_path'] = mesh_path
    data_map['keypoint_index'] = picked_ids
    data_map['keypoint_world_position'] = keypoint_3d_position
    with open(args.output_yaml, 'w') as outfile:
        yaml.dump(data_map, outfile)


if __name__ == '__main__':
    main()
