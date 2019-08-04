import attr
import numpy as np
import torch
from mankey.utils.imgproc import PixelCoord
import mankey.utils.imgproc as imgproc
import mankey.config.parameter as parameter
import mankey.network.resnet_nostage as resnet_nostage
import mankey.network.hourglass_staged as hourglass_staged
import mankey.network.predict as predict


# The construction of network
def construct_resnet_nostage(chkpt_path):
    # type: (str) -> (resnet_nostage.ResnetNoStage, resnet_nostage.ResnetNoStageConfig)
    """
    Construct the resnet nostage, load weight
    and upload the network to cuda.
    Currently, the parameter are hardcoded.
    :param chkpt_path: The path to checkpoint
    :return: (ResnetNoStage, ResnetNoStageConfig)
    """
    # The state dict of the network
    state_dict = torch.load(chkpt_path)
    n_keypoint = state_dict['head_net.features.9.weight'].shape[0] // 2
    assert n_keypoint * 2 == state_dict['head_net.features.9.weight'].shape[0]

    # Construct the network
    net_config = resnet_nostage.ResnetNoStageConfig()
    net_config.num_keypoints = n_keypoint
    net_config.image_channels = 4
    net_config.depth_per_keypoint = 2
    net_config.num_layers = 34
    network = resnet_nostage.ResnetNoStage(net_config)

    # Load the network
    network.load_state_dict(state_dict)
    network.cuda()
    network.eval()
    return network, net_config


# The network with stage
def construct_hourglass_staged(chkpt_path):
    # type: (str) -> (hourglass_staged.HourglassNet, hourglass_staged.HourglassNetConfig):
    # The state dict of the network
    state_dict = torch.load(chkpt_path)
    if 'score.0.weight' in state_dict:
        num_keypoints = state_dict['score.0.weight'].shape[0]
        data_parallel = False
    else:
        num_keypoints = state_dict['module.score.0.weight'].shape[0]
        data_parallel = True

    # The actual network
    net_config = hourglass_staged.HourglassNetConfig()
    net_config.num_keypoints = num_keypoints
    net_config.image_channels = 4
    net_config.depth_per_keypoint = 2
    net_config.num_layers = 34
    net_config.num_stages = 2
    network = hourglass_staged.HourglassNet(net_config)

    # Load the weight
    if data_parallel:
        network = torch.nn.DataParallel(network).cuda()
        network.load_state_dict(state_dict)
    else:
        network = network.cuda()
        network.load_state_dict(state_dict)
    return network, net_config


# The function used for image processing
@attr.s
class ImageProcOut(object):
    """
    Thin struct to hold the result of image processing
    """
    stacked_rgbd = np.ndarray(shape=[])
    bbox2patch = np.ndarray(shape=[])

    # Visualization only
    warped_rgb = np.ndarray(shape=[])
    warped_depth = np.ndarray(shape=[])


def proc_input_img_internal(
        rgb, depth,
        is_path_input,  # type: bool
        bbox_topleft,  # type: PixelCoord
        bbox_bottomright,  # type: PixelCoord
    ):
    """
    The worker for image processing.
    :param rgb: numpy ndarray if is_path_input==False, str if is_path_input==True
    :param depth: The same as rgb
    :param is_path_input: binary flag for whether the input is path or actual image
    :param bbox_topleft: Tight bounding box by maskrcnn
    :param bbox_bottomright: Tight bounding box by maskrcnn
    :return:
    """
    # Get the image and crop them using tight bounding box
    if is_path_input:
        warped_rgb, bbox2patch = imgproc.get_bbox_cropped_image_path(
            rgb, True,
            bbox_topleft, bbox_bottomright,
            patch_width=parameter.default_patch_size_input, patch_height=parameter.default_patch_size_input,
            bbox_scale=parameter.bbox_scale)
        warped_depth, _ = imgproc.get_bbox_cropped_image_path(
            depth, False,
            bbox_topleft, bbox_bottomright,
            patch_width=parameter.default_patch_size_input, patch_height=parameter.default_patch_size_input,
            bbox_scale=parameter.bbox_scale)
    else:  # Image input
        warped_rgb, bbox2patch = imgproc.get_bbox_cropped_image_raw(
            rgb, True,
            bbox_topleft, bbox_bottomright,
            patch_width=parameter.default_patch_size_input, patch_height=parameter.default_patch_size_input,
            bbox_scale=parameter.bbox_scale)
        warped_depth, _ = imgproc.get_bbox_cropped_image_raw(
            depth, False,
            bbox_topleft, bbox_bottomright,
            patch_width=parameter.default_patch_size_input, patch_height=parameter.default_patch_size_input,
            bbox_scale=parameter.bbox_scale)

    # Perform normalization
    normalized_rgb = imgproc.rgb_image_normalize(warped_rgb, parameter.rgb_mean, [1.0, 1.0, 1.0])
    normalized_depth = imgproc.depth_image_normalize(
        warped_depth,
        parameter.depth_image_clip,
        parameter.depth_image_mean,
        parameter.depth_image_scale)

    # Construct the tensor
    channels, height, width = normalized_rgb.shape
    stacked_rgbd = np.zeros(shape=(channels + 1, height, width), dtype=np.float32)
    stacked_rgbd[0:3, :, :] = normalized_rgb
    stacked_rgbd[3, :, :] = normalized_depth

    # OK
    imgproc_out = ImageProcOut()
    imgproc_out.stacked_rgbd = stacked_rgbd
    imgproc_out.bbox2patch = bbox2patch
    imgproc_out.warped_rgb = warped_rgb
    imgproc_out.warped_depth = warped_depth
    return imgproc_out


def proc_input_img_path(
        rgb_path,  # type: str
        depth_path,  # type: str
        bbox_topleft,  # type: PixelCoord
        bbox_bottomright,  # type: PixelCoord
):  # type: (str, str, PixelCoord, PixelCoord) -> ImageProcOut
    """
    Process the image for network input
    :param rgb_path:
    :param depth_path:
    :param bbox_topleft: Tight bounding box by maskrcnn
    :param bbox_bottomright: Tight bounding box by maskrcnn
    :return:
    """
    return proc_input_img_internal(
        rgb_path, depth_path, True,
        bbox_topleft, bbox_bottomright)


def proc_input_img_raw(
        rgb_img,  # type: np.ndarray
        depth_img,  # type: np.ndarray
        bbox_topleft,  # type: PixelCoord
        bbox_bottomright,  # type: PixelCoord
):  # type: (np.ndarray, np.ndarray, PixelCoord, PixelCoord) -> ImageProcOut
    """
    Process the image for network input
    :param rgb_img:
    :param depth_img:
    :param bbox_topleft: Tight bounding box by maskrcnn
    :param bbox_bottomright: Tight bounding box by maskrcnn
    :return:
    """
    return proc_input_img_internal(
        rgb_img, depth_img, False,
        bbox_topleft, bbox_bottomright)


def inference_resnet_nostage(
        network,  # type: resnet_nostage.ResnetNoStage
        imgproc_out,  # type: ImageProcOut
):  # type: (resnet_nostage.ResnetNoStage, ImageProcOut) -> np.ndarray
    """
    :param network: The network must be on GPU
    :param imgproc_out:
    :return: (3, n_keypoint) np array. (0:2, :) are x and y coords of keypoints in [-0.5, 0.5]
                                       (3, :) are the scaled depth
    """
    # Upload the image
    stacked_rgbd = torch.from_numpy(imgproc_out.stacked_rgbd)
    stacked_rgbd = torch.unsqueeze(stacked_rgbd, dim=0)
    stacked_rgbd = stacked_rgbd.cuda()

    # Do forward
    raw_pred = network(stacked_rgbd)
    num_keypoints = raw_pred.shape[1] // 2
    assert raw_pred.shape[1] == 2 * num_keypoints
    prob_pred = raw_pred[:, 0:num_keypoints, :, :]
    depthmap_pred = raw_pred[:, num_keypoints:, :, :]
    heatmap = predict.heatmap_from_predict(prob_pred, num_keypoints)
    coord_x, coord_y = predict.heatmap2d_to_normalized_imgcoord_gpu(heatmap, num_keypoints)
    depth_pred = predict.depth_integration(heatmap, depthmap_pred)

    # The scaled image coord and depth
    coord_x = coord_x.cpu().detach().numpy()
    coord_y = coord_y.cpu().detach().numpy()
    depth_pred = depth_pred.cpu().detach().numpy()

    # Combine them
    keypointxy_depth_pred = np.zeros((3, num_keypoints))
    keypointxy_depth_pred[0, :] = coord_x[0, :, 0]
    keypointxy_depth_pred[1, :] = coord_y[0, :, 0]
    keypointxy_depth_pred[2, :] = depth_pred[0, :, 0]
    return keypointxy_depth_pred


def inference_hourglass_staged(
        network,  # type: hourglass_staged.HourglassNet
        imgproc_out,  # type: ImageProcOut
):  # type: (hourglass_staged.HourglassNet, ImageProcOut) -> np.ndarray
    # Upload the image
    stacked_rgbd = torch.from_numpy(imgproc_out.stacked_rgbd)
    stacked_rgbd = torch.unsqueeze(stacked_rgbd, dim=0)
    stacked_rgbd = stacked_rgbd.cuda()

    # Do forward
    raw_pred_all = network(stacked_rgbd)
    raw_pred = raw_pred_all[-1]
    num_keypoints = raw_pred.shape[1] // 2
    assert raw_pred.shape[1] == 2 * num_keypoints
    prob_pred = raw_pred[:, 0:num_keypoints, :, :]
    depthmap_pred = raw_pred[:, num_keypoints:, :, :]
    heatmap = predict.heatmap_from_predict(prob_pred, num_keypoints)
    coord_x, coord_y = predict.heatmap2d_to_normalized_imgcoord_gpu(heatmap, num_keypoints)
    depth_pred = predict.depth_integration(heatmap, depthmap_pred)

    # The scaled image coord and depth
    coord_x = coord_x.cpu().detach().numpy()
    coord_y = coord_y.cpu().detach().numpy()
    depth_pred = depth_pred.cpu().detach().numpy()

    # Combine them
    keypointxy_depth_pred = np.zeros((3, num_keypoints))
    keypointxy_depth_pred[0, :] = coord_x[0, :, 0]
    keypointxy_depth_pred[1, :] = coord_y[0, :, 0]
    keypointxy_depth_pred[2, :] = depth_pred[0, :, 0]
    return keypointxy_depth_pred


# The post processing code to get the prediction in 3d
def get_keypoint_xy_depth_real_unit(
        keypointxy_depth_scaled):  # type: (np.ndarray) -> np.ndarray
    """
    From the scaled prediction by nn, get the keypoint
    pixel coordinate and depth in real unit.
    The pixel is in (patch_size, patch_size) image, while
    the depth is in mm
    :param keypointxy_depth_scaled:
    :return:
    """
    keypointxy_depth_realunit = np.zeros_like(keypointxy_depth_scaled)
    keypointxy_depth_realunit[0, :] = (keypointxy_depth_scaled[0, :] + 0.5) * parameter.default_patch_size_input
    keypointxy_depth_realunit[1, :] = (keypointxy_depth_scaled[1, :] + 0.5) * parameter.default_patch_size_input
    keypointxy_depth_realunit[2, :] = (keypointxy_depth_scaled[2, :] * parameter.depth_image_scale) + parameter.depth_image_mean
    return keypointxy_depth_realunit


def get_3d_prediction(
        keypoint_xy_depth_patch,
        bbox2patch):
    # type: (np.ndarray, np.ndarray) -> (np.ndarray, np.ndarray)
    """
    :param keypoint_xy_depth_patch: (3, n_keypoint) np array.
                                    keypoint_xy_depth[0:2, :] are the x and y coords of keypoints in the patch (0-256)
                                    keypoint_xy_depth[3, :] are the depth of keypoints in mm
    :param bbox2patch: 2x3 image transformation that map boundbox to patch
    :return: (the keypoint_xy_depth for original image, keypoint in camera frame in meters)
    """
    keypoint_xy = keypoint_xy_depth_patch[0:2, :]
    transform_homo = np.zeros((3, 3))
    transform_homo[0:2, :] = bbox2patch
    transform_homo[2, 2] = 1
    patch2bbox = np.linalg.inv(transform_homo)

    # To original image
    keypoint_xy_homo = np.ones_like(keypoint_xy_depth_patch)
    keypoint_xy_homo[0:2, :] = keypoint_xy
    keypoint_xy_depth_img = patch2bbox.dot(keypoint_xy_homo)
    keypoint_xy_depth_img[2, :] = keypoint_xy_depth_patch[2, :]

    # The point in camera frame
    camera_vertex = np.zeros_like(keypoint_xy_depth_img)
    camera_vertex[2, :] = keypoint_xy_depth_img[2, :].astype(np.float) / 1000.0
    camera_vertex[0, :] = (keypoint_xy_depth_img[0, :] - parameter.principal_x) * camera_vertex[2, :] / parameter.focal_x
    camera_vertex[1, :] = (keypoint_xy_depth_img[1, :] - parameter.principal_y) * camera_vertex[2, :] / parameter.focal_y
    return keypoint_xy_depth_img, camera_vertex
