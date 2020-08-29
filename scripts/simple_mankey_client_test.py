#! /usr/bin/env python

import argparse
import os

import cv2
import rospy
from sensor_msgs.msg import RegionOfInterest
from cv_bridge import CvBridge, CvBridgeError

from mankey_ros.srv import *


def main(visualize):
    rospy.wait_for_service('detect_keypoints')
    detect_keypoint = rospy.ServiceProxy(
        'detect_keypoints', MankeyKeypointDetection)

    # Get the test data path
    project_path = os.path.join(os.path.dirname(__file__), os.path.pardir)
    project_path = os.path.abspath(project_path)
    test_data_path = os.path.join(project_path, 'test_data')
    cv_rbg_path = os.path.join(test_data_path, '000000_rgb.png')
    cv_depth_path = os.path.join(test_data_path, '000000_depth.png')

    # Read the image
    cv_rgb = cv2.imread(cv_rbg_path, cv2.IMREAD_COLOR)
    cv_depth = cv2.imread(cv_depth_path, cv2.IMREAD_ANYDEPTH)

    # The bounding box
    roi = RegionOfInterest()
    roi.x_offset = 261
    roi.y_offset = 194
    roi.width = 327 - 261
    roi.height = 260 - 194

    # Build the request
    request = MankeyKeypointDetectionRequest()
    bridge = CvBridge()
    request.rgb_image = bridge.cv2_to_imgmsg(cv_rgb, 'bgr8')
    request.depth_image = bridge.cv2_to_imgmsg(cv_depth)
    request.bounding_box = roi
    response = detect_keypoint(request)
    print response

    if visualize:
        import open3d as o3d

        vis_list = []

        color = o3d.geometry.Image(cv_rgb)
        depth = o3d.geometry.Image(cv_depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        vis_list.append(pcd)

        for keypoint in response.keypoints_camera_frame:
            keypoints_coords \
                = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.1, origin=[keypoint.x, keypoint.y, keypoint.z])
            vis_list.append(keypoints_coords)
        o3d.visualization.draw_geometries(vis_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--visualize', '-v', type=int,
        default=0)
    args = parser.parse_args()
    visualize = args.visualize
    main(visualize)
