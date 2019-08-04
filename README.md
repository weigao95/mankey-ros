# mankey-ros

This repo is part of [kPAM](https://github.com/weigao95/kPAM) that provides ros service for 3D keypoint detection. The name ManKey stands for manipulation based on keypoints.


### Install Instruction

- Clone this repo into your `catkin workspace` by `git clone https://github.com/weigao95/mankey_ros`
- Run `catkin_make` to build the message types
- To run the code in `nodes/` and `scripts/`, you need to add `${project_path}` to `PYTHONPATH` [1]. You might run `export PYTHONPATH="${project_path}:${PYTHONPATH}"`

### Run Instruction

- Download the pre-trained network [here](https://drive.google.com/open?id=1ak3REzfSP3rqLOe27non8fbSGDSMDDls), the test data is available at `${project_root}/test_data`
- Start the keypoint detection server with `python nodes/mankey_keypoint_server.py --net_path path/to/pretrained/net`
- Run the test by `python scripts/simple_mankey_client_test.py`


### MISC

[1] The official way for python package installation is the `setup.py` file in `ros`. However, I always forgot to re-run `catkin_make` after modifying the python code, which causes confusion. Thus, I prefer to directly add `${project_path}` to `PYTHONPATH`.