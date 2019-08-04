# mankey
ManKey stands for manipulation based on keypoints. Different from most existing pose-aware manipulation algorithm that must contains an explicit pose estimation, we aim to directly define manipulation goal on top of semantic keypoints. 

This repo aims at keypoint detection for the manipulation pipeline. The typical input should be RGBD images, and the output is a list of semantic keypoint in 3D space.

##### Refactor TODOs:

- [x] Change to latest pytorch version
- [x] Change to dict based Dataset (and remove the tuple based Dataset)
- [x] Move the inference function into main repo from `inference_pdc.py`
- [x] Switch to python2 (at least the inference code), but make the code runnable for both version
- [x] Write ROS msg and action (in another repo), and test it in local and docker ros
- [ ] Write ROS msg and topic (after the maskrcnn is done), and test it in local and docker ros

