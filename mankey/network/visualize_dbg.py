import os
import numpy as np
import torch
import cv2
import mankey.network.predict as predict
import mankey.config.parameter as parameter
from mankey.dataproc.supervised_keypoint_loader import SupervisedKeypointDatasetConfig, SupervisedKeypointDataset


def visualize_entry_nostage(
        entry_idx: int,
        network: torch.nn.Module,
        dataset: SupervisedKeypointDataset,
        config: SupervisedKeypointDatasetConfig,
        save_dir: str):
    # The raw input
    processed_entry = dataset.get_processed_entry(dataset.entry_list[entry_idx])

    # The processed input
    stacked_rgbd = dataset[entry_idx][parameter.rgbd_image_key]
    normalized_xy_depth = dataset[entry_idx][parameter.keypoint_xyd_key]

    stacked_rgbd = torch.from_numpy(stacked_rgbd)
    stacked_rgbd = torch.unsqueeze(stacked_rgbd, dim=0)
    stacked_rgbd = stacked_rgbd.cuda()

    # Do forward
    raw_pred = network(stacked_rgbd)
    prob_pred = raw_pred[:, 0:dataset.num_keypoints, :, :]
    depthmap_pred = raw_pred[:, dataset.num_keypoints:, :, :]
    heatmap = predict.heatmap_from_predict(prob_pred, dataset.num_keypoints)
    coord_x, coord_y = predict.heatmap2d_to_normalized_imgcoord_gpu(heatmap, dataset.num_keypoints)
    depth_pred = predict.depth_integration(heatmap, depthmap_pred)

    # To actual image coord
    coord_x = coord_x.cpu().detach().numpy()
    coord_y = coord_y.cpu().detach().numpy()
    coord_x = (coord_x + 0.5) * config.network_in_patch_width
    coord_y = (coord_y + 0.5) * config.network_in_patch_height

    # To actual depth value
    depth_pred = depth_pred.cpu().detach().numpy()
    depth_pred = (depth_pred * config.depth_image_scale) + config.depth_image_mean

    # Combine them
    keypointxy_depth_pred = np.zeros((3, dataset.num_keypoints), dtype=np.int)
    keypointxy_depth_pred[0, :] = coord_x[0, :, 0].astype(np.int)
    keypointxy_depth_pred[1, :] = coord_y[0, :, 0].astype(np.int)
    keypointxy_depth_pred[2, :] = depth_pred[0, :, 0].astype(np.int)

    # Get the image
    from mankey.utils.imgproc import draw_image_keypoint, draw_visible_heatmap
    keypoint_rgb_cv = draw_image_keypoint(processed_entry.cropped_rgb, keypointxy_depth_pred, processed_entry.keypoint_validity)
    rgb_save_path = os.path.join(save_dir, 'image_%d_rgb.png' % entry_idx)
    cv2.imwrite(rgb_save_path, keypoint_rgb_cv)

    # The depth error
    depth_error_mm = np.abs(processed_entry.keypoint_xy_depth[2, :] - keypointxy_depth_pred[2, :])
    max_depth_error = np.max(depth_error_mm)
    print('Entry %d' % entry_idx)
    print('The max depth error (mm) is ', max_depth_error)

    # The pixel error
    pixel_error = np.sum(np.sqrt((processed_entry.keypoint_xy_depth[0:2, :] - keypointxy_depth_pred[0:2, :])**2), axis=0)
    max_pixel_error = np.max(pixel_error)
    print('The max pixel error (pixel in 256x256 image) is ', max_pixel_error)

    # Save the depth map
    # from utils.imgproc import get_visible_depth
    # assert processed_entry.has_depth
    # depth_map = processed_entry.cropped_depth
    # visible_depth = get_visible_depth(depth_map)
    # depth_save_path = os.path.join(save_dir, 'image_%d_depth.png' % entry_idx)
    # cv2.imwrite(depth_save_path, visible_depth)


def visualize_entry_staged(
        entry_idx: int,
        network: torch.nn.Module,
        dataset: SupervisedKeypointDataset,
        config: SupervisedKeypointDatasetConfig,
        save_dir: str):
    # The raw input
    processed_entry = dataset.get_processed_entry(dataset.entry_list[entry_idx])

    # The processed input
    stacked_rgbd = dataset[entry_idx][parameter.rgbd_image_key]
    normalized_xy_depth = dataset[entry_idx][parameter.keypoint_xyd_key]

    # Preprocessing
    stacked_rgbd = torch.from_numpy(stacked_rgbd)
    stacked_rgbd = torch.unsqueeze(stacked_rgbd, dim=0)
    stacked_rgbd = stacked_rgbd.cuda()

    # Do forward
    raw_pred_all = network(stacked_rgbd)
    raw_pred = raw_pred_all[-1]
    prob_pred = raw_pred[:, 0:dataset.num_keypoints, :, :]
    depthmap_pred = raw_pred[:, dataset.num_keypoints:, :, :]
    heatmap = predict.heatmap_from_predict(prob_pred, dataset.num_keypoints)
    coord_x, coord_y = predict.heatmap2d_to_normalized_imgcoord_gpu(heatmap, dataset.num_keypoints)
    depth_pred = predict.depth_integration(heatmap, depthmap_pred)

    # To actual image coord
    coord_x = coord_x.cpu().detach().numpy()
    coord_y = coord_y.cpu().detach().numpy()
    coord_x = (coord_x + 0.5) * config.network_in_patch_width
    coord_y = (coord_y + 0.5) * config.network_in_patch_height

    # To actual depth value
    depth_pred = depth_pred.cpu().detach().numpy()
    depth_pred = (depth_pred * config.depth_image_scale) + config.depth_image_mean

    # Combine them
    keypointxy_depth_pred = np.zeros((3, dataset.num_keypoints), dtype=np.int)
    keypointxy_depth_pred[0, :] = coord_x[0, :, 0].astype(np.int)
    keypointxy_depth_pred[1, :] = coord_y[0, :, 0].astype(np.int)
    keypointxy_depth_pred[2, :] = depth_pred[0, :, 0].astype(np.int)

    # Get the image
    from mankey.utils.imgproc import draw_image_keypoint, draw_visible_heatmap
    keypoint_rgb_cv = draw_image_keypoint(processed_entry.cropped_rgb, keypointxy_depth_pred, processed_entry.keypoint_validity)
    rgb_save_path = os.path.join(save_dir, 'image_%d_rgb.png' % entry_idx)
    cv2.imwrite(rgb_save_path, keypoint_rgb_cv)

    # The depth error
    depth_error_mm = np.abs(processed_entry.keypoint_xy_depth[2, :] - keypointxy_depth_pred[2, :])
    max_depth_error = np.max(depth_error_mm)
    print('Entry %d' % entry_idx)
    print('The max depth error (mm) is ', max_depth_error)

    # The pixel error
    pixel_error = np.sum(np.sqrt((processed_entry.keypoint_xy_depth[0:2, :] - keypointxy_depth_pred[0:2, :])**2), axis=0)
    max_pixel_error = np.max(pixel_error)
    print('The max pixel error (pixel in 256x256 image) is ', max_pixel_error)
