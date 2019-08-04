import torch
import os
import random
from torch.utils.data import DataLoader
from mankey.network.hourglass_staged import HourglassNetConfig, HourglassNet
from mankey.network.weighted_loss import weighted_l1_loss
import mankey.network.predict as predict
import mankey.config.parameter as parameter
import mankey.network.visualize_dbg as visualize_dbg
from mankey.dataproc.spartan_supervised_db import SpartanSupvervisedKeypointDBConfig, SpartanSupervisedKeypointDatabase
from mankey.dataproc.supervised_keypoint_loader import SupervisedKeypointDatasetConfig, SupervisedKeypointDataset


# The global parameter
learning_rate = 2e-5
n_epoch = 100


def construct_dataset(is_train: bool) -> (torch.utils.data.Dataset, SupervisedKeypointDatasetConfig):
    # Construct the db
    db_config = SpartanSupvervisedKeypointDBConfig()
    db_config.keypoint_yaml_name = 'mug_3_keypoint_image.yaml'
    db_config.pdc_data_root = '/home/wei/data/pdc'
    if is_train:
        db_config.config_file_path = '/home/wei/Coding/mankey/config/mugs_up_with_flat_logs.txt'
    else:
        db_config.config_file_path = '/home/wei/Coding/mankey/config/mugs_up_with_flat_test_logs.txt'
    database = SpartanSupervisedKeypointDatabase(db_config)

    # Construct torch dataset
    config = SupervisedKeypointDatasetConfig()
    config.network_in_patch_height = 256
    config.network_in_patch_width = 256
    config.network_out_map_width = 64
    config.network_out_map_height = 64
    config.image_database_list.append(database)
    config.is_train = is_train
    dataset = SupervisedKeypointDataset(config)
    return dataset, config


def construct_network():
    net_config = HourglassNetConfig()
    net_config.num_keypoints = 3
    net_config.image_channels = 4
    net_config.depth_per_keypoint = 2
    net_config.num_layers = 34
    net_config.num_stages = 2
    network = HourglassNet(net_config)
    return network, net_config


def visualize(network_path: str, save_dir: str):
    # Get the network
    network, net_config = construct_network()

    # Load the network
    state_dict = torch.load(network_path)
    network = torch.nn.DataParallel(network).cuda()
    network.load_state_dict(state_dict)
    network.eval()

    # Construct the dataset
    dataset, config = construct_dataset(is_train=False)

    # try the entry
    num_entry = 10
    entry_idx = []
    for i in range(num_entry):
        entry_idx.append(random.randint(0, len(dataset) - 1))

    # A good example and a bad one
    for i in range(len(entry_idx)):
        visualize_dbg.visualize_entry_staged(entry_idx[i], network, dataset, config, save_dir)


def train(checkpoint_dir: str, start_from_ckpnt: str = '', save_epoch_offset: int = 0):
    # Construct the dataset
    dataset_train, train_config = construct_dataset(is_train=True)
    # dataset_val, val_config = construct_dataset(is_train=False)

    # And the dataloader
    loader_train = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True, num_workers=4)
    # loader_val = DataLoader(dataset=dataset_val, batch_size=16, shuffle=False, num_workers=4)

    # Construct the regressor
    network, net_config = construct_network()

    # To cuda
    network = torch.nn.DataParallel(network).cuda()
    if len(start_from_ckpnt) > 0:
        network.load_state_dict(torch.load(start_from_ckpnt))

    # The checkpoint
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # The optimizer and scheduler
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 80], gamma=0.1)

    # The training loop
    for epoch in range(n_epoch):
        # Save the network
        if epoch % 2 == 0 and epoch > 0:
            file_name = 'checkpoint-%d.pth' % (epoch + save_epoch_offset)
            checkpoint_path = os.path.join(checkpoint_dir, file_name)
            print('Save the network at %s' % checkpoint_path)
            torch.save(network.state_dict(), checkpoint_path)

        # Prepare info for training
        network.train()
        train_error_xy = 0
        train_error_depth = 0
        train_error_xy_internal = 0

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

            # Move to gpu
            image = image.cuda()
            keypoint_xy_depth = keypoint_xy_depth.cuda()
            keypoint_weight = keypoint_weight.cuda()

            # To predict
            optimizer.zero_grad()
            raw_pred = network(image)

            # The last stage
            raw_pred_last = raw_pred[-1]
            prob_pred_last = raw_pred_last[:, 0:net_config.num_keypoints, :, :]
            depthmap_pred_last = raw_pred_last[:, net_config.num_keypoints:, :, :]
            heatmap_last = predict.heatmap_from_predict(prob_pred_last, net_config.num_keypoints)
            _, _, heatmap_height, heatmap_width = heatmap_last.shape

            # Compute the coordinate
            coord_x, coord_y = predict.heatmap2d_to_normalized_imgcoord_gpu(heatmap_last, net_config.num_keypoints)
            depth_pred = predict.depth_integration(heatmap_last, depthmap_pred_last)

            # Concantate them
            xy_depth_pred = torch.cat([coord_x, coord_y, depth_pred], dim=2)

            # Compute loss
            loss = weighted_l1_loss(xy_depth_pred, keypoint_xy_depth, keypoint_weight)

            # For all other layers
            for stage_i in range(len(raw_pred) - 1):
                prob_pred_i = raw_pred[stage_i]   # Only 2d prediction on previous layers
                assert prob_pred_i.shape == prob_pred_last.shape
                heatmap_i = predict.heatmap_from_predict(prob_pred_i, net_config.num_keypoints)
                coord_x_i, coord_y_i = predict.heatmap2d_to_normalized_imgcoord_gpu(heatmap_i, net_config.num_keypoints)
                xy_pred_i = torch.cat([coord_x_i, coord_y_i], dim=2)
                loss = loss + weighted_l1_loss(xy_pred_i, keypoint_xy_depth[:, :, 0:2], keypoint_weight[:, :, 0:2])

            # The SGD step
            loss.backward()
            optimizer.step()
            del loss

            # Log info
            xy_error = float(weighted_l1_loss(
                xy_depth_pred[:, :, 0:2],
                keypoint_xy_depth[:, :, 0:2],
                keypoint_weight[:, :, 0:2]).item())
            depth_error = float(weighted_l1_loss(
                xy_depth_pred[:, :, 2],
                keypoint_xy_depth[:, :, 2],
                keypoint_weight[:, :, 2]).item())
            # The error of internal stage
            keypoint_xy_pred_internal, _ = predict.heatmap2d_to_normalized_imgcoord_argmax(raw_pred[0])
            xy_error_internal = float(
                weighted_l1_loss(keypoint_xy_pred_internal[:, :, 0:2], keypoint_xy_depth[:, :, 0:2],
                                 keypoint_weight[:, :, 0:2]).item())
            if idx % 100 == 0:
                print('Iteration %d in epoch %d' % (idx, epoch))
                print('The averaged pixel error is (pixel in 256x256 image): ', 256 * xy_error / len(xy_depth_pred))
                print('The averaged depth error is (mm): ', 256 * depth_error / len(xy_depth_pred))
                print('The averaged internal pixel error is (pixel in 256x256 image): ',
                      256 * xy_error_internal / image.shape[0])

            # Update info
            train_error_xy += float(xy_error)
            train_error_depth += float(depth_error)
            train_error_xy_internal += float(xy_error_internal)

        # The info at epoch level
        print('Epoch %d' % epoch)
        print('The training averaged pixel error is (pixel in 256x256 image): ', 256 * train_error_xy / len(dataset_train))
        print('The training averaged depth error is (mm): ',
              train_config.depth_image_scale * train_error_depth / len(dataset_train))
        print('The training internal averaged pixel error is (pixel in 256x256 image): ',
              256 * train_error_xy_internal / len(dataset_train))


if __name__ == '__main__':
    chkpt_dir = 'ckpnt'
    train(checkpoint_dir=chkpt_dir)

    # The visualization code
    # tmp_dir = 'tmp'
    # if not os.path.exists(tmp_dir):
    #     os.mkdir(tmp_dir)
    # visualize(net_path, tmp_dir)
