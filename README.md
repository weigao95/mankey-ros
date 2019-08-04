# mankey_ros

This repo is part of [kPAM](https://github.com/weigao95/kPAM) that provides ros service for 3D keypoint detection.

### Run Instruction

- Clone this repo into your `catkin workspace` by `git clone https://github.com/weigao95/mankey_ros`, Run `git submodule update --init --recursive` to get the [mankey]() submodule
- Download the pre-trained network [here](https://drive.google.com/open?id=1ak3REzfSP3rqLOe27non8fbSGDSMDDls), the test data is available [here]()
- Run `catkin_make` to build the message types
- Add `${project_root}` to your `PYTHONPATH` by running `export PYTHONPATH=${project_root}:${PYTHONPATH}`
- Start the keypoint detection server with `python nodes/mankey_keypoint_server.py`
