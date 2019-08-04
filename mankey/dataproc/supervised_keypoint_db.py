import attr
import os
import numpy as np
from mankey.utils.imgproc import PixelCoord
from typing import List


@attr.s
class SupervisedKeypointDBEntry:
    # The path to rgb is must
    rgb_image_path = ''

    # The path to depth image
    depth_image_path = ''

    # The path to mask image
    binary_mask_path = ''

    # If length zero, indicates no depth
    @property
    def has_depth(self):
        return len(self.depth_image_path) > 0

    @property
    def has_mask(self):
        return len(self.binary_mask_path) > 0

    # The bounding box is tight
    bbox_top_left = PixelCoord()
    bbox_bottom_right = PixelCoord()

    # The information related to keypoint
    # All of these element should be in size of (3, n_keypoint)
    # The first element iterate over x, y, or z, the second element iterate over keypoints
    keypoint_camera = None  # The position of keypoint expressed in camera frame using meter as unit

    # (pixel_x, pixel_y, mm_depth) for each keypoint
    # Note that the pixel might be outside the image space
    keypoint_pixelxy_depth = None

    # Each element indicate the validity of the corresponded keypoint coordinate
    # 1 means valid, 0 means not valid
    keypoint_validity_weight = None
    on_boundary = False

    # The pose of the camera
    # Homogeneous transformation matrix
    camera_in_world = np.ndarray(shape=[4, 4])


def sanity_check_spartan(entry: SupervisedKeypointDBEntry) -> bool:
    if len(entry.rgb_image_path) < 1 or (not os.path.exists(entry.rgb_image_path)):
        return False

    if len(entry.rgb_image_path) >= 1 and (not os.path.exists(entry.depth_image_path)):
        return False

    if len(entry.rgb_image_path) >= 1 and (not os.path.exists(entry.binary_mask_path)):
        return False

    if entry.keypoint_validity_weight is None or entry.keypoint_pixelxy_depth is None or entry.keypoint_camera is None:
        return False

    if entry.bbox_bottom_right.x is None or entry.bbox_bottom_right.y is None:
        return False

    if entry.bbox_top_left.x is None or entry.bbox_top_left.y is None:
        return False

    if entry.camera_in_world.shape[0] != 4 or entry.camera_in_world.shape[1] != 4:
        return False

    # OK
    return True


class SupervisedImageKeypointDatabase(object):
    """
    The class serves as an thin interface for the REAL torch.Dataset.
    The purpose of this abstraction is that we might use different DB
    implementation, such as the spartan and the synthetic, which might
    contains different meta info, file structure, etc.
    """
    def __init__(self):
        pass

    def get_entry_list(self) -> List[SupervisedKeypointDBEntry]:
        raise NotImplementedError

    @property
    def num_keypoints(self):
        raise NotImplementedError
