import torch
import os
import random
import mankey.config.parameter as parameter
from torch.utils.data import DataLoader
from mankey.network.resnet_nostage import ResnetNoStageConfig, ResnetNoStage, init_from_modelzoo
from mankey.network.weighted_loss import weighted_l1_loss
import mankey.network.predict as predict
import mankey.network.visualize_dbg as visualize_dbg
from mankey.dataproc.spartan_supervised_db import SpartanSupvervisedKeypointDBConfig, SpartanSupervisedKeypointDatabase
from mankey.dataproc.supervised_keypoint_loader import SupervisedKeypointDatasetConfig, SupervisedKeypointDataset

# This script is a 2d experiment to test the direct heatmap regression
# Some global parameter
learning_rate = 2e-4
heatmap_loss_weight = 1.0
n_epoch = 60


def construct_dataset(is_train: bool) -> (torch.utils.data.Dataset, SupervisedKeypointDatasetConfig):
    # Construct the db
    db_config = SpartanSupvervisedKeypointDBConfig()
    db_config.keypoint_yaml_name = 'shoe_6_keypoint_image.yaml'
    db_config.pdc_data_root = '/home/wei/data/pdc'
    db_config.config_file_path = '/home/wei/Coding/mankey/config/boot_logs.txt'
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
    net_config.num_keypoints = 6
    net_config.image_channels = 4
    net_config.depth_per_keypoint = 1
    net_config.num_layers = 34
    network = ResnetNoStage(net_config)
    return network, net_config


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
    num_entry = 0
    entry_idx = []
    for i in range(num_entry):
        entry_idx.append(random.randint(0, len(dataset) - 1))

    # A good example and a bad one
    entry_idx.append(35)
    entry_idx.append(1126)
    for i in range(len(entry_idx)):
        visualize_dbg.visualize_entry_nostage(entry_idx[i], network, dataset, config, save_dir)


def train(checkpoint_dir: str, start_from_ckpnt: str = '', save_epoch_offset: int = 0):
    # Construct the dataset
    dataset_train, train_config = construct_dataset(is_train=True)
    dataset_val, val_config = construct_dataset(is_train=False)

    # And the dataloader
    loader_train = DataLoader(dataset=dataset_train, batch_size=16, shuffle=True, num_workers=4)
    loader_val = DataLoader(dataset=dataset_val, batch_size=16, shuffle=False, num_workers=4)

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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 40], gamma=0.1)

    # The loss for heatmap
    heatmap_criterion = torch.nn.MSELoss().cuda()

    # The training loop
    for epoch in range(n_epoch):
        # Save the network
        if epoch % 100 == 0 and epoch > 0:
            file_name = 'checkpoint-%d.pth' % (epoch + save_epoch_offset)
            checkpoint_path = os.path.join(checkpoint_dir, file_name)
            print('Save the network at %s' % checkpoint_path)
            torch.save(network.state_dict(), checkpoint_path)

        # Prepare info for training
        network.train()
        train_error_xy = 0

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

            # Upload to GPU
            image = image.cuda()
            keypoint_xy_depth = keypoint_xy_depth.cuda()
            keypoint_weight = keypoint_weight.cuda()
            target_heatmap = target_heatmap.cuda()

            # To predict
            optimizer.zero_grad()
            raw_pred = network(image)
            prob_pred = raw_pred[:, 0:net_config.num_keypoints, :, :]
            _, _, heatmap_height, heatmap_width = prob_pred.shape

            # Compute loss
            loss = heatmap_criterion(prob_pred, target_heatmap)

            # Do update
            loss.backward()
            optimizer.step()

            # cleanup
            del loss

            # Do some pred and log
            keypoint_xy_pred, _ = predict.heatmap2d_to_normalized_imgcoord_argmax(prob_pred)
            xy_error = float(weighted_l1_loss(keypoint_xy_pred[:, :, 0:2], keypoint_xy_depth[:, :, 0:2], keypoint_weight[:, :, 0:2]).item())
            if idx % 100 == 0:
                print('Iteration %d in epoch %d' % (idx, epoch))
                print('The averaged pixel error is (pixel in 256x256 image): ', 256 * xy_error / image.shape[0])

            # Update info
            train_error_xy += float(xy_error)

        # The info at epoch level
        print('Epoch %d' % epoch)
        print('The training averaged pixel error is (pixel in 256x256 image): ', 256 * train_error_xy / len(dataset_train))

        # Prepare info at epoch level
        network.eval()
        val_error_xy = 0

        # The validation iteration of the data
        for idx, data in enumerate(loader_val):
            # Get the data
            image = data[parameter.rgbd_image_key]
            keypoint_xy_depth = data[parameter.keypoint_xyd_key]
            keypoint_weight = data[parameter.keypoint_validity_key]

            image = image.cuda()
            keypoint_xy_depth = keypoint_xy_depth.cuda()
            keypoint_weight = keypoint_weight.cuda()

            # To predict
            pred = network(image)
            prob_pred = pred[:, 0:net_config.num_keypoints, :, :]
            _, _, heatmap_height, heatmap_width = prob_pred.shape

            # Compute the coordinate
            keypoint_xy_pred, _ = predict.heatmap2d_to_normalized_imgcoord_argmax(prob_pred)
            xy_error = float(weighted_l1_loss(keypoint_xy_pred[:, :, 0:2], keypoint_xy_depth[:, :, 0:2],
                                              keypoint_weight[:, :, 0:2]).item())

            # Update info
            val_error_xy += float(xy_error)

        # The info at epoch level
        print('The validation averaged pixel error is (pixel in 256x256 image): ', 256 * val_error_xy / len(dataset_val))


if __name__ == '__main__':
    #ckpnt_dir = os.path.join(os.path.dirname(__file__), 'checkpoint_hm_l2')
    #net_path = 'trained_model/checkpoint_hm_l2/checkpoint-98.pth'
    train(checkpoint_dir='ckpnt')

    # The visualization code
    #tmp_dir = 'tmp'
    #if not os.path.exists(tmp_dir):
    #    os.mkdir(tmp_dir)
    #visualize(net_path, tmp_dir)
