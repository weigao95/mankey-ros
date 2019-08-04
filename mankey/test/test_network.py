import unittest
import torch


class HourglassNetTest(unittest.TestCase):

    def test_output_size(self):
        from mankey.network.hourglass_staged import HourglassNet, HourglassNetConfig
        config = HourglassNetConfig()
        config.num_keypoints = 10
        config.image_channels = 4
        config.depth_per_keypoint = 2
        net = HourglassNet(config)

        # Test on some dymmy image
        batch_size = 10
        img = torch.zeros((batch_size, config.image_channels, 256, 256))
        net_out = net(img)
        self.assertTrue(len(net_out) == config.num_stages)
        out = net_out[1]
        self.assertEqual(out.shape[0], batch_size)
        self.assertEqual(out.shape[1], config.num_keypoints * config.depth_per_keypoint)
        self.assertEqual(out.shape[2], 256 / 4)
        self.assertEqual(out.shape[3], 256 / 4)


class ResnetNoStageTest(unittest.TestCase):

    def test_output_size(self):
        from mankey.network.resnet_nostage import ResnetNoStageConfig, ResnetNoStage, init_from_modelzoo
        config = ResnetNoStageConfig()
        config.num_layers = 50
        config.num_keypoints = 10
        config.depth_per_keypoint = 1
        config.image_channels = 4
        net = ResnetNoStage(config)

        # Load from model zoo
        init_from_modelzoo(net, config)

        # Test on some dymmy image
        batch_size = 10
        img = torch.zeros((batch_size, config.image_channels, 256, 256))
        out = net(img)

        # Check it
        self.assertEqual(out.shape[0], batch_size)
        self.assertEqual(out.shape[1], config.num_keypoints * config.depth_per_keypoint)
        self.assertEqual(out.shape[2], 256 / 4)
        self.assertEqual(out.shape[3], 256 / 4)


if __name__ == '__main__':
    unittest.main()
