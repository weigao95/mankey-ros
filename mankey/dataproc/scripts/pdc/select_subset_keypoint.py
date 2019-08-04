import os
import yaml
import copy


# The config
keypoint_src_dir = '/home/wei/data/pdc/annotations/keypoints/mug_3_keypoint'
keypoint_target_dir = '/home/wei/data/pdc/annotations/keypoints/mug_2_keypoint'
keypoint_target_name = ['bottom_center', 'top_center']
target_annotation_type = 'mug_2_keypoint'


def process_annotation_file(keypoint_yaml_list, target_save_path: str):
    processed_list = []
    for annotation_dict in keypoint_yaml_list:
        processed_dict = copy.deepcopy(annotation_dict)
        # Update the value
        processed_dict['annotation_type'] = target_annotation_type

        # Update the keypoint
        keypoint_dict_from = annotation_dict['keypoints']
        keypoint_dict_to = dict()
        for keypoint_name in keypoint_target_name:
            assert keypoint_name in keypoint_dict_from
            keypoint_dict_to[keypoint_name] = copy.deepcopy(keypoint_dict_from[keypoint_name])
        processed_dict['keypoints'] = keypoint_dict_to

        # OK
        processed_list.append(processed_dict)

    # Save the path
    with open(target_save_path, 'w') as save_file:
        yaml.dump(processed_list, save_file)


def main():
    # Check the directory
    assert os.path.exists(keypoint_src_dir)
    if not os.path.exists(keypoint_target_dir):
        os.mkdir(keypoint_target_dir)

    # Iterate over file in the src
    for annotation_yaml_filename in os.listdir(keypoint_src_dir):
        # Get the yaml path
        annotation_yaml_path = os.path.join(keypoint_src_dir, annotation_yaml_filename)
        if not annotation_yaml_path.endswith('.yaml'):
            continue

        # The target path
        target_yaml_path = os.path.join(keypoint_target_dir, annotation_yaml_filename)

        # Open the yaml file and get the map
        annotation_yaml_file = open(annotation_yaml_path, 'r')
        annotation_yaml_list = yaml.load(annotation_yaml_file)
        annotation_yaml_file.close()

        # Do processing
        process_annotation_file(annotation_yaml_list, target_yaml_path)


if __name__ == '__main__':
    main()
