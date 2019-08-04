import attr
import os
from typing import List
import yaml
import numpy as np
from mankey.utils.imgproc import PixelCoord, pixel_in_bbox
from mankey.utils.transformations import quaternion_matrix
from mankey.dataproc.supervised_keypoint_db import SupervisedImageKeypointDatabase, SupervisedKeypointDBEntry, sanity_check_spartan


@attr.s
class SpartanSupvervisedKeypointDBConfig:
    # ${pdc_data_root}/logs_proto/2018-10....
    pdc_data_root = ''

    # A list of file indicates which logs will be used for dataset.
    # If None, all logs with keypoint annotation in data_root will be used
    # Usually specific_logs.txt
    config_file_path = None

    # The name of the yaml file with keypoint and bound-box annotation
    # Relative to the "${pdc_data_root}/logs_proto/2018-10..../processed" folder
    # Should be in ${pdc_data_root}/logs_proto/2018-10..../processed/${keypoint_yaml_name}
    keypoint_yaml_name = 'scene_bbox_keypoint.yaml'
    # Output the loading process
    verbose = True


def camera2world_from_map(camera2world_map) -> np.ndarray:
    """
    Get the transformation matrix from the storage map
    See ${processed_log_path}/processed/images/pose_data.yaml for an example
    :param camera2world_map:
    :return: The (4, 4) transform matrix from camera 2 world
    """
    # The rotation part
    camera2world_quat = [1, 0, 0, 0]
    camera2world_quat[0] = camera2world_map['quaternion']['w']
    camera2world_quat[1] = camera2world_map['quaternion']['x']
    camera2world_quat[2] = camera2world_map['quaternion']['y']
    camera2world_quat[3] = camera2world_map['quaternion']['z']
    camera2world_quat = np.asarray(camera2world_quat)
    camera2world_matrix = quaternion_matrix(camera2world_quat)

    # The linear part
    camera2world_matrix[0, 3] = camera2world_map['translation']['x']
    camera2world_matrix[1, 3] = camera2world_map['translation']['y']
    camera2world_matrix[2, 3] = camera2world_map['translation']['z']
    return camera2world_matrix


class SpartanSupervisedKeypointDatabase(SupervisedImageKeypointDatabase):
    """
    The spartan multi-view RGBD dataset with keypoint-annotation. This one
    is specified to one object and back-ground subtracted bounding box (instance mask).
    Compared with the tree-like config used in pytorch-dense-correspondence, I would
    favor simple, flatten dataset like this one ...
    """
    def __init__(self, config: SpartanSupvervisedKeypointDBConfig):
        super(SpartanSupervisedKeypointDatabase, self).__init__()
        self._config = config  # Not actually use it, but might be useful

        # Build a list of scene that will be used as the dataset
        self._scene_path_list = []
        if config.config_file_path is not None:
            self._scene_path_list = self._get_scene_from_config(config)
        else:
            # Use everything in pdc root with keypoint annotation
            raise NotImplementedError

        # For each scene
        self._keypoint_entry_list = []
        self._num_keypoint = -1
        for scene_root in self._scene_path_list:
            # The info code
            if config.verbose:
                print('Processing: ', scene_root)

            # The processing code
            scene_entry = self._build_scene_entry(scene_root, config.keypoint_yaml_name)
            for item in scene_entry:
                self._keypoint_entry_list.append(item)

        # Simple info
        print('The number of images is %d' % len(self._keypoint_entry_list))

    def get_entry_list(self) -> List[SupervisedKeypointDBEntry]:
        return self._keypoint_entry_list

    @property
    def num_keypoints(self):
        assert self._num_keypoint > 0
        return self._num_keypoint

    def _get_scene_from_config(self, config: SpartanSupvervisedKeypointDBConfig) -> List[str]:
        assert os.path.exists(config.pdc_data_root)
        assert os.path.exists(config.config_file_path)

        # Read the config file
        scene_root_list = []
        with open(config.config_file_path, 'r') as config_file:
            lines = config_file.read().split('\n')
            for line in lines:
                if len(line) == 0:
                    continue
                scene_root = os.path.join(config.pdc_data_root, line)
                if self._is_scene_valid(scene_root, config.keypoint_yaml_name):
                    scene_root_list.append(scene_root)

        # OK
        return scene_root_list

    def _is_scene_valid(self, scene_root: str, keypoint_yaml_name: str) -> bool:
        # The path must be valid
        if not os.path.exists(scene_root):
            return False

        # Must contains keypoint annotation
        scene_processed_root = os.path.join(scene_root, 'processed')
        keypoint_yaml_path = os.path.join(scene_processed_root, keypoint_yaml_name)
        if not os.path.exists(keypoint_yaml_path):
            return False

        # OK
        return True

    def _build_scene_entry(self, scene_root: str, keypoint_yaml_name: str) -> List[SupervisedKeypointDBEntry]:
        # Get the yaml file
        scene_processed_root = os.path.join(scene_root, 'processed')
        keypoint_yaml_path = os.path.join(scene_processed_root, keypoint_yaml_name)
        assert os.path.exists(keypoint_yaml_path)

        # Read the yaml map
        keypoint_yaml_file = open(keypoint_yaml_path, 'r')
        keypoint_yaml_map = yaml.load(keypoint_yaml_file)
        keypoint_yaml_file.close()

        # Iterate over image
        entry_list = []
        for image_key in keypoint_yaml_map.keys():
            image_map = keypoint_yaml_map[image_key]
            image_entry = self._get_image_entry(image_map, scene_root)
            if image_entry is not None and self._check_image_entry(image_entry):
                entry_list.append(image_entry)

        # Ok
        return entry_list

    def _get_image_entry(self, image_map, scene_root: str) -> SupervisedKeypointDBEntry:
        entry = SupervisedKeypointDBEntry()
        # The path for rgb image
        rgb_name = image_map['rgb_image_filename']
        rgb_path = os.path.join(scene_root, 'processed/images/' + rgb_name)
        assert os.path.exists(rgb_path)
        entry.rgb_image_path = rgb_path

        # The path for depth image
        depth_name = image_map['depth_image_filename']
        depth_path = os.path.join(scene_root, 'processed/images/' + depth_name)
        assert os.path.exists(depth_path) # Spartan must have depth image
        entry.depth_image_path = depth_path

        # The path for mask image
        mask_name = depth_name[0:6] + '_mask.png'
        mask_path = os.path.join(scene_root, 'processed/image_masks/' + mask_name)
        assert os.path.exists(mask_path)
        entry.binary_mask_path = mask_path

        # The camera pose in world
        camera2world_map = image_map['camera_to_world']
        entry.camera_in_world = camera2world_from_map(camera2world_map)

        # The bounding box
        top_left = PixelCoord()
        bottom_right = PixelCoord()
        top_left.x, top_left.y = image_map['bbox_top_left_xy'][0], image_map['bbox_top_left_xy'][1]
        bottom_right.x, bottom_right.y = image_map['bbox_bottom_right_xy'][0], image_map['bbox_bottom_right_xy'][1]
        entry.bbox_top_left = top_left
        entry.bbox_bottom_right = bottom_right

        # The size of keypoint
        keypoint_camera_frame_list = image_map['3d_keypoint_camera_frame']
        n_keypoint = len(keypoint_camera_frame_list)
        if self._num_keypoint < 0:
            self._num_keypoint = n_keypoint
        else:
            assert self._num_keypoint == n_keypoint

        # The keypoint in camera frame
        entry.keypoint_camera = np.zeros((3, n_keypoint))
        for i in range(n_keypoint):
            for j in range(3):
                entry.keypoint_camera[j, i] = keypoint_camera_frame_list[i][j]

        # The pixel coordinate and depth of keypoint
        keypoint_pixelxy_depth_list = image_map['keypoint_pixel_xy_depth']
        assert n_keypoint == len(keypoint_pixelxy_depth_list)
        entry.keypoint_pixelxy_depth = np.zeros((3, n_keypoint), dtype=np.int)
        for i in range(n_keypoint):
            for j in range(3):
                entry.keypoint_pixelxy_depth[j, i] = keypoint_pixelxy_depth_list[i][j]

        # Check the validity
        entry.keypoint_validity_weight = np.ones((3, n_keypoint))
        for i in range(n_keypoint):
            pixel = PixelCoord()
            pixel.x = entry.keypoint_pixelxy_depth[0, i]
            pixel.y = entry.keypoint_pixelxy_depth[1, i]
            depth_mm = entry.keypoint_pixelxy_depth[2, i]
            valid = True
            if depth_mm < 0:  # The depth cannot be negative
                valid = False

            # The pixel must be in bounding box
            if not pixel_in_bbox(pixel, entry.bbox_top_left, entry.bbox_bottom_right):
                valid = False

            # Invalid all the dimension
            if not valid:
                entry.keypoint_validity_weight[0, i] = 0
                entry.keypoint_validity_weight[1, i] = 0
                entry.keypoint_validity_weight[2, i] = 0
                entry.on_boundary = True

        # OK
        return entry

    def _check_image_entry(self, entry: SupervisedKeypointDBEntry) -> bool:
        # Check the bounding box
        if entry.bbox_top_left.x is None or entry.bbox_top_left.y is None:
            return False

        if entry.bbox_bottom_right.x is None or entry.bbox_bottom_right.y is None:
            return False

        # OK
        return True


# Simple code to test the db
def spartan_db_test():
    config = SpartanSupvervisedKeypointDBConfig()
    config.keypoint_yaml_name = 'shoe_6_keypoint_image.yaml'
    config.pdc_data_root = '/home/wei/data/pdc'
    config.config_file_path = '/home/wei/Coding/mankey/config/boot_logs.txt'
    database = SpartanSupervisedKeypointDatabase(config)
    entry_list = database.get_entry_list()
    for entry in entry_list:
        assert sanity_check_spartan(entry)


if __name__ == '__main__':
    spartan_db_test()
