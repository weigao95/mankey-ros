import numpy as np
import cv2
import attr
# from typing import List


@attr.s
class PixelCoord(object):
    """
    A small struct used to index in image.
    Note that x is the column index, y is row index
    """
    x = 0
    y = 0

    @property
    def row_location(self): # type: (PixelCoord) -> int
        return self.y

    @property
    def col_location(self): # type: (PixelCoord) -> int
        return self.x


# The group of method used for bounding box processing.
# Mainly used to rectify the bounding box
def first_nonzero_idx(
        binary_array,
        reversed):  # type: (np.ndarray, bool) -> int
    """
    Get the index of the first element in an array that is not zero
    reversed means whether the binary_array should be reversed
    :param binary_array: A 1-D numpy array
    :param reversed:
    :return: The index to the first non-zero element
    """
    start = 0
    end = binary_array.size
    step = 1
    if reversed:
        start = binary_array.size - 1
        end = -1
        step = -1

    # The iteration
    for i in range(start, end, step):
        if binary_array[i] > 0:
            return i

    # Everything is zero
    return None


def mask2bbox(mask_img):  # type: (np.ndarray) -> (PixelCoord, PixelCoord)
    """
    Given an object binary mask, get the tight object bounding box
    as a tuple contains top_left and bottom_right pixel coord
    :param mask_img: (height, width, 3) mask image
    :return: A tuple contains top_left and bottom_right pixel coord
    """
    binary_mask = mask_img.max(axis=2)
    n_rows, n_cols = binary_mask.shape
    # Compute sum over the row and compute the left and right
    mask_rowsum = np.sum(binary_mask, axis=0, keepdims=False)
    assert mask_rowsum.size == n_cols
    left = first_nonzero_idx(mask_rowsum, False)
    right = first_nonzero_idx(mask_rowsum, True)

    # Compute sum over the col and compute the top and bottom
    mask_colsum = np.sum(binary_mask, axis=1)
    assert mask_colsum.size == n_rows
    top = first_nonzero_idx(mask_colsum, False)
    bottom = first_nonzero_idx(mask_colsum, True)

    # Ok
    top_left = PixelCoord()
    top_left.x = left
    top_left.y = top
    bottom_right = PixelCoord()
    bottom_right.x = right
    bottom_right.y = bottom
    return top_left, bottom_right


def pixel_in_bbox(
        pixel,  # type: PixelCoord
        top_left,  # type: PixelCoord
        bottom_right,  # type: PixelCoord
):  # type: (PixelCoord, PixelCoord, PixelCoord) -> bool
    """
    Given an pixel, check if that pixel in bounding box specificed by top_left and bottom_right
    The bounding box must be valid (in the image).
    :param pixel:
    :param top_left:
    :param bottom_right:
    :return:
    """
    if pixel.row_location < 0 or pixel.col_location < 0:
        return False

    if pixel.row_location < top_left.row_location or pixel.row_location > bottom_right.row_location:
        return False

    if pixel.col_location < top_left.col_location or pixel.col_location > bottom_right.col_location:
        return False

    # Seems ok
    return True


# The group of method related to image transformation.
# Mainly used for data augmentation.
def rotate_2d(pt_2d, rot_rad):  # type: (np.ndarray, float) -> np.ndarray
    """
    Rotate an given 2d direction
    :param pt_2d: A 2d direction expressed in np.ndarray
    :param rot_rad: The angle of rotation
    :return: The rotated direction. Note that the norm doesn't change
    """
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def transform_2d(point_2d, transform): # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    Homogeneous transformation of 2D point.
    :param point_2d: 2d point expressed in np.ndarray
    :param transform: 3x3 homogeneous transform matrix
    :return: the transformed point
    """
    src_pt = np.array([point_2d[0], point_2d[1], 1.]).T
    dst_pt = np.dot(transform, src_pt)
    return dst_pt[0:2]


def get_transform_to_patch(
        center_x,  # type: int
        center_y,  # type: int
        bbox_width,  # type: int
        bbox_height,  # type: int
        dst_width,  # type: int
        dst_height,  # type: int
        scale=1.0,  # type: float
        rot_rad=0.0,  # type: float
):
    """
    Given a bounding box expressed as center and size, first augment
    it with scale and rotation, then compute an image transformation
    that will map the augmented bounding box to an image at the size of (dst_height, dst_width).
    To avoid distortion, the bbox_width and bbox_height should have
    similar aspect ration with target width and height.
    :param center_x:
    :param center_y:
    :param bbox_width:
    :param bbox_height:
    :param dst_width:
    :param dst_height:
    :param scale:
    :param rot_rad:
    :return: The opencv transformation
    """
    # Augment with scale
    src_width = bbox_width * scale
    src_height = bbox_height * scale
    src_center = np.array([center_x, center_y], dtype=np.float32)

    # Augment with rotation
    src_downdir = rotate_2d(np.array([0, src_height * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_width * 0.5, 0], dtype=np.float32), rot_rad)

    # The target info
    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    # Construct the matrix
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    # The opencv transformation
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def get_bbox2patch(
    bbox_topleft,  # type: PixelCoord
    bbox_bottomright,  # type: PixelCoord
    image_width,  # type: int
    image_height,  # type: int
    patch_width,  # type: int
    patch_height,  # type: int
    on_boundary=False,  # type: bool
    bbox_scale=1.0,  # type: float
    # These are agumentation parameter
    scale=1.0,  # type: float
    rot_rad=0.0,  # type: float
):
    """
    Given a tight bounding box, compute the rectified bounding box,
    do scale and rotation augmentation and compute the transform that map the
    bounding box into the target patch.
    :param bbox_topleft: The top-left pixel of the bounding box
    :param bbox_bottomright: The bottom-right pixel of the bounding box
    :param image_width: The width of image where bbox lives
    :param image_height:
    :param patch_width: The target size of the transform, only support width==height now
    :param patch_height:
    :param on_boundary: Is the detected object on the boundary of the image. Only available in training
    :param bbox_scale: As the bounding box is tight, use this parameter to make it losser
    :param scale: Domain randomization parameter
    :param rot_rad:
    :return:
    """
    #from utils.imgproc import rectify_bbox_center_align, rectify_bbox_in_image, get_transform_to_patch
    # Get the bounding box
    assert patch_height == patch_width
    if on_boundary:
        rectifified_bbox_topleft, rectifified_bbox_bottomright = rectify_bbox_in_image(
            bbox_topleft, bbox_bottomright,
            image_width, image_height)
    else:
        rectifified_bbox_topleft, rectifified_bbox_bottomright = rectify_bbox_center_align(
            bbox_topleft, bbox_bottomright)

    # Another representation of bounding box
    center_x = int(0.5 * (rectifified_bbox_topleft.x + rectifified_bbox_bottomright.x))
    center_y = int(0.5 * (rectifified_bbox_topleft.y + rectifified_bbox_bottomright.y))
    bbox_width = rectifified_bbox_bottomright.x - rectifified_bbox_topleft.x
    bbox_height = rectifified_bbox_bottomright.y - rectifified_bbox_topleft.y
    bbox_width *= bbox_scale
    bbox_height *= bbox_scale

    # Invoke the method
    return get_transform_to_patch(
        center_x, center_y,
        bbox_width, bbox_height,
        patch_width, patch_height,
        scale, rot_rad)


def get_bbox_cropped_image_raw(
        cv_img,  # type: np.ndarray
        is_rgb,  # type: bool
        bbox_topleft,  # type: PixelCoord,
        bbox_bottomright,  # type: PixelCoord
        patch_width,  # type: int
        patch_height,  # type: int
        bbox_scale=1.0,  # type: float
        on_boundary=False,  # type: bool
        # The augmentation parameter
        scale=1.0,  # type: float
        rot_rad=0.0,  # type: float
):
    # Get the size of image
    assert cv_img is not None
    if is_rgb:
        img_height, img_width, _ = cv_img.shape
    else:
        img_height, img_width = cv_img.shape
    assert img_height > 0 and img_width > 0

    # Get the transformation to bounding box
    bbox2patch = get_bbox2patch(
        bbox_topleft=bbox_topleft, bbox_bottomright=bbox_bottomright,
        image_width=img_width, image_height=img_height,
        patch_width=patch_width, patch_height=patch_height,
        on_boundary=on_boundary,
        bbox_scale=bbox_scale,
        scale=scale, rot_rad=rot_rad)

    # Do transformation
    warped_img = cv2.warpAffine(
        cv_img, bbox2patch,
        (int(patch_width), int(patch_height)),
        flags=cv2.INTER_LINEAR)
    return warped_img, bbox2patch


def get_bbox_cropped_image_path(
    imgpath,  # type: str
    is_rgb,  # type: bool
    bbox_topleft,  # type: PixelCoord
    bbox_bottomright,  # type: PixelCoord
    patch_width,  # type: int
    patch_height,  # type: int
    bbox_scale=1.0,  # type: float
    on_boundary=False,  # type: bool
    # The augmentation parameter
    scale=1.0,  # type: float
    rot_rad=0.0,  # type: float
):  # type: (str, bool, PixelCoord, PixelCoord, int, int, float, bool, float, float) -> (np.ndarray, np.ndarray)
    # Load the image
    if is_rgb:
        cv_img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    else:
        cv_img = cv2.imread(imgpath, cv2.IMREAD_ANYDEPTH)

    # OK
    return get_bbox_cropped_image_raw(
        cv_img, is_rgb,
        bbox_topleft=bbox_topleft, bbox_bottomright=bbox_bottomright,
        patch_width=patch_width, patch_height=patch_height,
        on_boundary=on_boundary,
        bbox_scale=bbox_scale,
        scale=scale, rot_rad=rot_rad)


def rectify_bbox_center_align(
        top_left_in,  # type: PixelCoord
        bottom_right_in,  # type: PixelCoord
):  # type: (PixelCoord, PixelCoord) -> (PixelCoord, PixelCoord)
    """
    Given an input bounding box, change its width or height to make
    it has a aspect ratio of 1, which will be used for most case.
    The center before and after rectification will be the same.
    Note that the rectified bounding box may not in the image.
    :param top_left_in:
    :param bottom_right_in:
    :return: A tuple of new top_left and bottom_right
    """
    center_x = int(0.5 * (top_left_in.x + bottom_right_in.x))
    center_y = int(0.5 * (top_left_in.y + bottom_right_in.y))
    width = bottom_right_in.col_location - top_left_in.col_location
    height = bottom_right_in.row_location - top_left_in.row_location
    length = max(width, height)
    half_length = int(length * 0.5)

    # Construct the output
    top_left = PixelCoord()
    top_left.x = center_x - half_length
    top_left.y = center_y - half_length
    bottom_right = PixelCoord()
    bottom_right.x = center_x + half_length
    bottom_right.y = center_y + half_length
    return top_left, bottom_right


def rectify_bbox_in_image(
        top_left_in,  # type: PixelCoord
        bottom_right_in,  # type: PixelCoord
        image_width,  # type: int
        image_height,  # type: int
):  # type: (PixelCoord, PixelCoord, int, int) -> (PixelCoord, PixelCoord)
    """
    Rectify the bounding box to have unit aspect ratio, but keep it inside
    the image. Note that the center of bounding box might not aligned with
    the bounding box before rectification.
    :param top_left_in:
    :param bottom_right_in:
    :param image_width:
    :param image_height:
    :return: A tuple of new top_left and bottom_right
    """
    # Do rectification as normal
    aspect_fixed_topleft, aspect_fixed_bottomright = rectify_bbox_center_align(top_left_in, bottom_right_in)

    # Check each coord
    if aspect_fixed_topleft.x < 0:
        move_x = aspect_fixed_topleft.x
        aspect_fixed_topleft.x = 0
        aspect_fixed_bottomright.x -= move_x  # move_x is negative

    if aspect_fixed_topleft.y < 0:
        move_y = aspect_fixed_topleft.y
        aspect_fixed_topleft.y = 0
        aspect_fixed_bottomright.y -= move_y

    if aspect_fixed_bottomright.x >= image_width:
        move_x = image_width - aspect_fixed_bottomright.x - 1
        aspect_fixed_bottomright.x = image_width - 1
        aspect_fixed_topleft.x += move_x

    if aspect_fixed_bottomright.y >= image_height:
        move_y = image_height - aspect_fixed_bottomright.y - 1
        aspect_fixed_bottomright.y = image_height - 1
        aspect_fixed_topleft.y += move_y

    # Check it
    # This might not be true, as the aspect-fixed bounding box
    # might be larger than the shorter edge of the image
    # assert aspect_fixed_topleft.x >= 0 and aspect_fixed_topleft.y >= 0
    # assert aspect_fixed_bottomright.x < image_width and aspect_fixed_bottomright.y < image_height

    # OK
    return aspect_fixed_topleft, aspect_fixed_bottomright


# The group of method to scale the rgb image
def rgb_image_normalize(
        cv_img,  # type: np.ndarray
        rgb_mean,  # type: List[float]
        rgb_scale,  # type: List[float]
):  # type: (np.ndarray, List[float], List[float]) -> np.ndarray
    """
    (height, width, channels) -> (channels, height, width), BGR->RGB and normalize
    :param cv_img: The raw opencv image as np.ndarray in the shape of (height, width, 3)
    :param rgb_mean: The mean value for RGB, all of them are in [0, 1]
    :param rgb_scale: The scale value of RGB, should be close to 1.0
    :return: The normalized, randomized RGB tensor
    """
    tensor = cv_img.copy()
    tensor = np.transpose(tensor, (2, 0, 1))  # (c, h, w)
    tensor = tensor[::-1, :, :]  # (R, G, B)
    tensor = tensor.astype(np.float32)

    # Scale and normalize
    normalizer = [1.0/255.0, 1.0/255.0, 1.0/255.0]
    for i in range(3):
        normalizer[i] = normalizer[i] * rgb_scale[i]

    # Apply to image
    for channel in range(len(rgb_mean)):
        tensor[channel, :, :] = (normalizer[channel] * tensor[channel, :, :]) - rgb_mean[channel]

    # OK
    return tensor


def depth_image_normalize(
        cv_depth,  # type: np.ndarray
        depth_clip,  # type: int
        depth_mean,  # type: int
        depth_scale,  # type: int
):  # type: (np.ndarray, int, int, int) -> np.ndarray
    """
    :param cv_depth: image in the size of (img_height, img_width)
    :param depth_clip:
    :param depth_mean:
    :param depth_scale:
    :return: out = (clip(depth_in, 0, depth_clip) - depth_mean) / depth_scale
    """
    tensor = cv_depth.copy()
    tensor[tensor >= depth_clip] = 0

    # Do normalize
    tensor = tensor.astype(np.float32)
    tensor -= float(depth_mean)
    normalizer = 1.0 / float(depth_scale)
    tensor = tensor * normalizer
    return tensor


def get_guassian_heatmap(
        point_float,  # type: np.ndarray
        heatmap_size,  # type: int
        sigma=1
):  # type: (np.ndarray, int, float) -> np.ndarray
    """
    Given the 2d keypoint, return the target heatmap with Guassian distribution
    :param point: np.ndarray in shape (2,)
    :param heatmap_size: Current limited to heatmap with the same height and width
    :param sigma: The sigma value for Guassian distribution, by default 1
    :return: The heatmap with Gaussian center at point and sigma be variance
    """
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    point = point_float.astype(np.int)
    ul = [int(point[0] - tmpSize), int(point[1] - tmpSize)]
    br = [int(point[0] + tmpSize + 1), int(point[1] + tmpSize + 1)]

    # Return an empty heatmap if not in bound
    if (ul[0] >= heatmap_size or ul[1] >= heatmap_size or
            br[0] < 0 or br[1] < 0):
        return np.zeros(shape=(heatmap_size, heatmap_size))

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = size / 4.0
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], heatmap_size) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], heatmap_size) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], heatmap_size)
    img_y = max(0, ul[1]), min(br[1], heatmap_size)

    # Construct the image and return
    heatmap = np.zeros((heatmap_size, heatmap_size))
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return heatmap


# Misc functions for visualization
# All these method SHOULD NOT modify the input
def get_visible_depth(depth_in):  # type: (np.ndarray) -> np.ndarray
    max_value = np.max(depth_in)
    return cv2.convertScaleAbs(src=depth_in, alpha=(255.0 / max_value))


def get_visible_mask(binary_mask):  # type: (np.ndarray) -> np.ndarray
    return get_visible_depth(binary_mask)


def draw_image_keypoint(
        image,  # type: np.ndarray,
        keypoint_pixelxy_depth,  # type: np.ndarray,
        keypoint_validity,  # type: np.ndarray
    ): # type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
    """
    Draw give keypoints on the image
    :param image: opencv image
    :param keypoint_pixelxy_depth: (3, n_keypoint) np.ndarray where (0:2, :) are pixel coordinates
    :param keypoint_validity: (3, n_keypoint) validity of the corresponded coordinate
    :return: The image with keypoint annotated
    """
    img_clone = image.copy()
    n_keypoint = keypoint_pixelxy_depth.shape[1]
    for i in range(n_keypoint):
        # Not valid
        if not keypoint_validity[0, i] > 0:
            continue

        # Draw it
        cv2.circle(img_clone,
                   center=(int(keypoint_pixelxy_depth[0, i]), int(keypoint_pixelxy_depth[1, i])),
                   radius=8, color=(255, 255, 0))

    # OK
    return img_clone


def draw_visible_heatmap(heatmap_np, verbose=True): # type: (np.ndarray, bool) -> np.ndarray
    """
    Visualize the given heatmap
    :param heatmap_np: (height, width) image
    :param verbose:
    :return: An image in the same size as input but visible by cv2.imwrite()
    """
    max_heatmap_np = heatmap_np.max()
    if verbose:
        print('The max value in heatmap is %f' % max_heatmap_np)

    # The actual drawing method
    heatmap_vis_raw = cv2.convertScaleAbs(heatmap_np, alpha=(255 / max_heatmap_np))
    kernel = np.ones((3, 3), np.float32)
    heatmap_vis = cv2.filter2D(heatmap_vis_raw, -1, kernel)
    return heatmap_vis
