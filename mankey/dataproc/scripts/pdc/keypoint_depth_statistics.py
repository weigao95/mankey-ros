from mankey.dataproc.spartan_supervised_db import SpartanSupvervisedKeypointDBConfig, SpartanSupervisedKeypointDatabase
from mankey.dataproc.supervised_keypoint_loader import SupervisedKeypointDatasetConfig, SupervisedKeypointDataset

"""
Given a specific list of logs, this script
computes the mean, minimum and maximum for the depth
"""


def main():
    # Construct the db
    db_config = SpartanSupvervisedKeypointDBConfig()
    db_config.keypoint_yaml_name = 'shoe_6_keypoint_image.yaml'
    db_config.pdc_data_root = '/home/wei/data/pdc'
    db_config.config_file_path = '/home/wei/Coding/mankey/config/shoe_logs.txt'
    database = SpartanSupervisedKeypointDatabase(db_config)

    # Construct torch dataset
    config = SupervisedKeypointDatasetConfig()
    config.network_in_patch_height = 256
    config.network_in_patch_width = 256
    config.network_out_map_width = 64
    config.network_out_map_height = 64
    config.image_database_list.append(database)
    config.is_train = False
    dataset = SupervisedKeypointDataset(config)

    # The counter
    min_keypoint_depth: float = 10000
    max_keypoint_depth: float = 0
    avg_keypoint_depth: float = 0
    counter = 0

    # Iterate
    for i in range(len(dataset)):
        processed_entry = dataset.get_processed_entry(dataset.entry_list[i])
        n_keypoint = processed_entry.keypoint_xy_depth.shape[1]
        for j in range(n_keypoint):
            min_keypoint_depth = min(min_keypoint_depth, processed_entry.keypoint_xy_depth[2, j])
            max_keypoint_depth = max(max_keypoint_depth, processed_entry.keypoint_xy_depth[2, j])
            avg_keypoint_depth += processed_entry.keypoint_xy_depth[2, j]
            counter += 1

    # Some output
    print('The min value is ', min_keypoint_depth)
    print('The max value is ', max_keypoint_depth)
    print('The average value is ', avg_keypoint_depth / float(counter))


if __name__ == '__main__':
    main()
