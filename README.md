# mankey-ros

This repo is part of [kPAM](https://github.com/weigao95/kPAM) that provides ros service for 3D keypoint detection. The name ManKey stands for manipulation based on keypoints.


### Install Instruction

- Clone this repo into your `catkin workspace` by `git clone https://github.com/weigao95/mankey-ros mankey_ros` (Note the underscore)
- Run `catkin_make` to build the message types
- To run the code in `nodes/` and `scripts/`, you need to add `${project_path}` to `PYTHONPATH` [1]. You might run `export PYTHONPATH="${project_path}:${PYTHONPATH}"`

### Run Instruction

- Download the pre-trained network [here](https://drive.google.com/open?id=1ak3REzfSP3rqLOe27non8fbSGDSMDDls), the test data is available at `${project_root}/test_data`
- Start the keypoint detection server with `python nodes/mankey_keypoint_server.py --net_path path/to/pretrained/net`
- Run the test by `python scripts/simple_mankey_client_test.py`

### Training Instruction

Please follow the instruction [here](https://github.com/RobotLocomotion/manip_dataset) to download the dataset and annotation. The dataset directory structure is described [here](https://github.com/RobotLocomotion/pytorch-dense-correspondence/blob/master/doc/data_organization.md). We will use the mug data as an example.

You need to run `mankey/dataproc/scripts/pdc/keypoint_mesh2img_all_director.py` to generate the annotation files. For this script, the `annotation_dir` parameter should be the path to `mug_3_keypoint` directory of the annotation you just download. The `log_root_path` is the full path to `logs_proto` in [this document](https://github.com/RobotLocomotion/pytorch-dense-correspondence/blob/master/doc/data_organization.md). Other two parameters are less important and can be set to some intutive strings.

After data generation, you can run training using scripts in `mankey/experiment/xxxx.py`. You need to modify the `db_config` in `construct_dataset` function. The `keypoint_yaml_name` is the same as the `save_relative_path` in the `keypoint_mesh2img_all_director.py` script. The `config_file_path` is the scene list, for mug you might use `mankey/config/mug_up_with_flat_logs.txt`.


### MISC

[1] The official way for python package installation is the `setup.py` file in `ros`. However, I always forgot to re-run `catkin_make` after modifying the python code, which causes confusion. Thus, I prefer to directly add `${project_path}` to `PYTHONPATH`.
