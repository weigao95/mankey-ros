import torch
import os
import cv2
import numpy as np
import random
from torch.utils.data import DataLoader
from mankey.network.resnet_nostage import ResnetNoStageConfig, ResnetNoStage, init_from_modelzoo
from mankey.network.weighted_loss import weighted_mse_loss, weighted_l1_loss
import mankey.network.predict as predict
import mankey.config.parameter as parameter
from mankey.dataproc.spartan_supervised_db import SpartanSupvervisedKeypointDBConfig, SpartanSupervisedKeypointDatabase
from mankey.dataproc.supervised_keypoint_loader import SupervisedKeypointDatasetConfig, SupervisedKeypointDataset


# Some global parameter
learning_rate = 2e-4
n_epoch = 120
heatmap_loss_weight = 0.1


def construct_dataset(is_train: bool) -> (SupervisedKeypointDataset, SupervisedKeypointDatasetConfig):
    # Construct the db info
    db_config = SpartanSupvervisedKeypointDBConfig()
    db_config.keypoint_yaml_name = 'mug_3_keypoint_image.yaml'
    db_config.pdc_data_root = '/home/wei/data/pdc'
    if is_train:
        db_config.config_file_path = '/home/wei/Coding/mankey/config/mugs_up_with_flat_logs.txt'
    else:
        db_config.config_file_path = '/home/wei/Coding/mankey/config/mugs_up_with_flat_test_logs.txt'

    # Construct the database
    database = SpartanSupervisedKeypointDatabase(db_config)

    # Construct torch dataset
    config = SupervisedKeypointDatasetConfig()
    config.network_in_patch_width = 256
    config.network_in_patch_height = 256
    config.network_out_map_width = 64
    config.network_out_map_height = 64
    config.image_database_list.append(database)
    config.is_train = is_train
    dataset = SupervisedKeypointDataset(config)
    return dataset, config


def construct_network():
    net_config = ResnetNoStageConfig()
    net_config.num_keypoints = 3
    net_config.image_channels = 4
    net_config.depth_per_keypoint = 3  # For integral heatmap, depthmap and regress heatmap
    net_config.num_layers = 34
    network = ResnetNoStage(net_config)
    return network, net_config


def visualize_entry(
        entry_idx: int,
        network: torch.nn.Module,
        dataset: SupervisedKeypointDataset,
        config: SupervisedKeypointDatasetConfig,
        save_dir: str):
    # The raw input
    processed_entry = dataset.get_processed_entry(dataset.entry_list[entry_idx])

    # The processed input
    stacked_rgbd, normalized_xy_depth, _, _ = dataset[entry_idx]
    stacked_rgbd = torch.from_numpy(stacked_rgbd)
    stacked_rgbd = torch.unsqueeze(stacked_rgbd, dim=0)
    stacked_rgbd = stacked_rgbd.cuda()

    # Do forward
    raw_pred = network(stacked_rgbd)
    prob_pred = raw_pred[:, 0:dataset.num_keypoints, :, :]
    depthmap_pred = raw_pred[:, dataset.num_keypoints:2*dataset.num_keypoints, :, :]
    regress_heatmap = raw_pred[:, 2*dataset.num_keypoints:, :, :]
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
    from utils.imgproc import draw_image_keypoint, draw_visible_heatmap
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

    # Save the heatmap for the one with largest pixel error
    max_error_keypoint = np.argmax(pixel_error)
    raw_heatmap_np = regress_heatmap.cpu().detach().numpy()[0, max_error_keypoint, :, :]
    heatmap_vis = draw_visible_heatmap(raw_heatmap_np)
    heatmap_save_path = os.path.join(save_dir, 'image_%d_heatmap.png' % entry_idx)
    cv2.imwrite(heatmap_save_path, heatmap_vis)


def visualize(network_path: str, save_dir: str):
    # Get the network
    network, _ = construct_network()

    # Load the network
    network.load_state_dict(torch.load(network_path))
    network.cuda()
    network.eval()

    # Construct the dataset
    dataset, config = construct_dataset(is_train=False)

    # try the entry
    num_entry = 50
    entry_idx = []
    for i in range(num_entry):
        entry_idx.append(random.randint(0, len(dataset) - 1))

    # A good example and a bad one
    for i in range(len(entry_idx)):
        visualize_entry(entry_idx[i], network, dataset, config, save_dir)


def train(checkpoint_dir: str, start_from_ckpnt: str = '', save_epoch_offset: int = 0):
    # Construct the dataset
    dataset_train, train_config = construct_dataset(is_train=True)

    # And the dataloader
    loader_train = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True, num_workers=4)

    # Construct the regressor
    network, net_config = construct_network()
    if len(start_from_ckpnt) > 0:
        network.load_state_dict(torch.load(start_from_ckpnt))
    else:
        init_from_modelzoo(network, net_config)
    network.cuda()

    # The checkpoint
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # The optimizer and scheduler
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 90], gamma=0.1)

    # The loss for heatmap
    heatmap_criterion = torch.nn.MSELoss().cuda()

    # The training loop
    for epoch in range(n_epoch):
        # Save the network
        if epoch % 4 == 0 and epoch > 0:
            file_name = 'checkpoint-%d.pth' % (epoch + save_epoch_offset)
            checkpoint_path = os.path.join(checkpoint_dir, file_name)
            print('Save the network at %s' % checkpoint_path)
            torch.save(network.state_dict(), checkpoint_path)

        # Prepare info for training
        network.train()
        train_error_xy = 0
        train_error_depth = 0
        train_error_xy_heatmap = 0

        # The learning rate step
        scheduler.step()
        for param_group in optimizer.param_groups:
            print('The learning rate is ', param_group['lr'])

        # The training iteration over the dataset
        for idx, data in enumerate(loader_train):
            # Get the data
            image = data[parameter.rgbd_image_key]
            keypoint_xy_depth = data[parameter.keypoint_xyd_key]
            keypoint_weight = data[parameter.keypoint_validity_key]
            target_heatmap = data[parameter.target_heatmap_key]

            # Upload to cuda
            image = image.cuda()
            keypoint_xy_depth = keypoint_xy_depth.cuda()
            keypoint_weight = keypoint_weight.cuda()
            target_heatmap = target_heatmap.cuda()

            # To predict
            optimizer.zero_grad()
            raw_pred = network(image)
            prob_pred = raw_pred[:, 0:net_config.num_keypoints, :, :]
            depthmap_pred = raw_pred[:, net_config.num_keypoints:2*net_config.num_keypoints, :, :]
            regress_heatmap = raw_pred[:, 2*net_config.num_keypoints:, :, :]
            integral_heatmap = predict.heatmap_from_predict(prob_pred, net_config.num_keypoints)
            _, _, heatmap_height, heatmap_width = integral_heatmap.shape

            # Compute the coordinate
            coord_x, coord_y = predict.heatmap2d_to_normalized_imgcoord_gpu(integral_heatmap, net_config.num_keypoints)
            depth_pred = predict.depth_integration(integral_heatmap, depthmap_pred)

            # Concantate them
            xy_depth_pred = torch.cat((coord_x, coord_y, depth_pred), dim=2)

            # Compute loss
            loss = weighted_mse_loss(xy_depth_pred, keypoint_xy_depth, keypoint_weight)
            loss = loss + heatmap_loss_weight * heatmap_criterion(regress_heatmap, target_heatmap)

            # Do update
            loss.backward()
            optimizer.step()

            # Log info
            xy_error = float(weighted_l1_loss(xy_depth_pred[:, :, 0:2], keypoint_xy_depth[:, :, 0:2],keypoint_weight[:, :, 0:2]).item())
            depth_error = float(weighted_l1_loss(xy_depth_pred[:, :, 2], keypoint_xy_depth[:, :, 2], keypoint_weight[:, :, 2]).item())
            keypoint_xy_pred_heatmap, _ = predict.heatmap2d_to_normalized_imgcoord_argmax(regress_heatmap)
            xy_error_heatmap = float(weighted_l1_loss(keypoint_xy_pred_heatmap[:, :, 0:2], keypoint_xy_depth[:, :, 0:2], keypoint_weight[:, :, 0:2]).item())
            if idx % 100 == 0:
                print('Iteration %d in epoch %d' % (idx, epoch))
                print('The averaged pixel error is (pixel in 256x256 image): ', 256 * xy_error / len(xy_depth_pred))
                print('The averaged depth error is (mm): ', train_config.depth_image_scale * depth_error / len(xy_depth_pred))
                print('The averaged heatmap argmax pixel error is (pixel in 256x256 image): ', 256 * xy_error_heatmap / len(xy_depth_pred))

            # Update info
            train_error_xy += float(xy_error)
            train_error_depth += float(depth_error)
            train_error_xy_heatmap += float(xy_error_heatmap)

        # The info at epoch level
        print('Epoch %d' % epoch)
        print('The training averaged pixel error is (pixel in 256x256 image): ', 256 * train_error_xy / len(dataset_train))
        print('The training averaged depth error is (mm): ',
              train_config.depth_image_scale * train_error_depth / len(dataset_train))
        print('The training averaged heatmap pixel error is (pixel in 256x256 image): ',
              256 * train_error_xy_heatmap / len(dataset_train))


def main():
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'ckpnt')
    train(checkpoint_dir)


if __name__ == '__main__':
    main()
