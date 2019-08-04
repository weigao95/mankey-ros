#! /usr/bin/env python
import mankey.network.inference as inference
from mankey.utils.imgproc import PixelCoord
import argparse
import os
import numpy as np

# The ros staff
from mankey_ros.srv import *
import rospy
from sensor_msgs.msg import RegionOfInterest
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError

# The argument is only the network path
parser = argparse.ArgumentParser()
parser.add_argument('--net_path', type=str,
                    default='/home/wei/data/trained_model/keypoint/mug/checkpoint-135.pth',
                    help='The absolute path to network checkpoint')


class MankeyKeypointDetectionServer(object):

    def __init__(self, network_chkpt_path):
        # The converter of opencv image
        self._bridge = CvBridge()

        # The network
        assert os.path.exists(network_chkpt_path)
        self._network, self._net_config = inference.construct_resnet_nostage(network_chkpt_path)

    def handle_keypoint_request(self, request):
        # type: (MankeyKeypointDetectionRequest) -> MankeyKeypointDetectionResponse
        # Decode the image
        try:
            cv_color = self._bridge.imgmsg_to_cv2(request.rgb_image, 'bgr8')
            cv_depth = self._bridge.imgmsg_to_cv2(request.depth_image)
        except CvBridgeError as err:
            print('Image conversion error. Please check the image encoding.')
            print(err.message)
            return self.get_invalid_response()

        # The image is correct, perform inference
        try:
            bbox = request.bounding_box
            camera_keypoint = self.process_request_raw(cv_color, cv_depth, bbox)
        except (RuntimeError, TypeError, ValueError):
            print('The inference is not correct.')
            return self.get_invalid_response()

        # The response
        response = MankeyKeypointDetectionResponse()
        response.num_keypoints = camera_keypoint.shape[1]
        for i in range(camera_keypoint.shape[1]):
            point = Point()
            point.x = camera_keypoint[0, i]
            point.y = camera_keypoint[1, i]
            point.z = camera_keypoint[2, i]
            response.keypoints_camera_frame.append(point)
        return response

    def process_request_raw(
            self,
            cv_color,  # type: np.ndarray
            cv_depth,  # type: np.ndarray
            bbox,  # type: RegionOfInterest
    ):  # type: (np.ndarray, np.ndarray, RegionOfInterest) -> np.ndarray
        # Parse the bounding box
        top_left, bottom_right = PixelCoord(), PixelCoord()
        top_left.x = bbox.x_offset
        top_left.y = bbox.y_offset
        bottom_right.x = bbox.x_offset + bbox.width
        bottom_right.y = bbox.y_offset + bbox.height

        # Perform the inference
        imgproc_out = inference.proc_input_img_raw(
            cv_color, cv_depth,
            top_left, bottom_right)
        keypointxy_depth_scaled = inference.inference_resnet_nostage(self._network, imgproc_out)
        keypointxy_depth_realunit = inference.get_keypoint_xy_depth_real_unit(keypointxy_depth_scaled)
        _, camera_keypoint = inference.get_3d_prediction(
            keypointxy_depth_realunit,
            imgproc_out.bbox2patch)
        return camera_keypoint

    @staticmethod
    def get_invalid_response():
        response = MankeyKeypointDetectionResponse()
        response.num_keypoints = -1
        return response

    def run(self):
        rospy.init_node('mankey_keypoint_server')
        srv = rospy.Service('detect_keypoints', MankeyKeypointDetection, self.handle_keypoint_request)
        print('The server for mankey keypoint detection initialization OK!')
        rospy.spin()


def main(netpath):
    server = MankeyKeypointDetectionServer(netpath)
    server.run()


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    print('Unknown arguments')
    print(unknown)
    net_path = args.net_path
    main(netpath=net_path)
