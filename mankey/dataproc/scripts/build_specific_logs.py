import os
import argparse
import yaml
from typing import List

"""
The script would be use in companion with pdc-scripts/scripts/specific to build
the required specific_logs.txt from a given composite data in pytorch-dense-correspondence.
You need to modifiy the pdc root to the root path of pytorch dense correspondence 
"""
parser = argparse.ArgumentParser()
parser.add_argument('--pdc_root',
                    type=str, default='/home/wei/Coding/correspondence',
                    help='The root path of pytorch-dense-correspondence')
parser.add_argument('--composite_dataset',
                    type=str, default='shoe_train_all_shoes.yaml',
                    help='The composite dataset the specific_logs.txt file is required')
parser.add_argument('--output_path', type=str, default='specific_logs.txt')


# The root path of all level of folders
args = parser.parse_args()
pdc_root = args.pdc_root
dataset_config_root = os.path.join(pdc_root, 'config/dense_correspondence/dataset')
composite_dataset_config_root = os.path.join(dataset_config_root, 'composite')
singleobj_dataset_config_root = os.path.join(dataset_config_root, 'single_object')
multiobj_dataset_config_root = os.path.join(dataset_config_root, 'multi_object')


# The method to process a single_object dataset
def process_singleobj_dataset(config_full_path: str, logs_root_path: str) -> (List[str], List[str]):
    """
    Given a full path to the yaml config of a single object dataset,
    return all logs (scenes) associated with it
    :param config_full_path: The full path to the config file
    :param logs_root_path: The root_path read from the composite dataset, just for check
    :return: Two list of strings be the training and testing logs (scenes)
    """
    with open(config_full_path, 'r') as stream:
        try:
            single_config = yaml.load(stream)
            # Check the logs_root_path
            single_logs_root = single_config['logs_root_path']
            assert single_logs_root.__eq__(logs_root_path)
            # Just use the data
            return single_config['train'], single_config['test']
        except yaml.YAMLError as exc:
            print(exc)


# The method to process a composite dataset
def process_composite_dataset(config_fullpath: str, output_txt_path: str):
    """
    The main processing function, given an fullpath of a yaml config of a
    composite dataset, write all logs (scenes) as a file to output_txt_path
    :param config_fullpath:
    :param output_txt_path:
    :return: None
    """
    # the final list of train and test logs
    train_logs: List[str] = []
    test_logs: List[str] = []

    # The processing loop
    with open(config_fullpath, 'r') as stream:
        try:
            composite_config = yaml.load(stream)
            logs_root_path = composite_config['logs_root_path']
            # Process the single object dataset
            singleobj_config_path = composite_config['single_object_scenes_config_files']
            for config in singleobj_config_path:
                singleobj_config_fullpath = os.path.join(singleobj_dataset_config_root, config)
                train_list, test_list = process_singleobj_dataset(singleobj_config_fullpath, logs_root_path)
                for item in train_list:
                    item_with_root_path = os.path.join(logs_root_path, item)
                    train_logs.append(item_with_root_path)
                for item in test_list:
                    item_with_root_path = os.path.join(logs_root_path, item)
                    test_logs.append(item_with_root_path)

            # Process the multi object dataset
            multiobj_config_path = composite_config['multi_object_scenes_config_files']
            if len(multiobj_config_path) > 0:
                raise NotImplementedError()
        except yaml.YAMLError as exc:
            print(exc)

    # The output
    outfile = open(output_txt_path, 'w')
    for item in train_logs:
        outfile.write(item)
        outfile.write('\n')


def main():
    # Get the dataset and check its existence
    composite_dataset_fullpath = os.path.join(composite_dataset_config_root, args.composite_dataset)
    assert os.path.exists(composite_dataset_fullpath)

    # Hand in to processor
    output_path = args.output_path
    process_composite_dataset(composite_dataset_fullpath, output_path)


if __name__ == '__main__':
    main()

