import unittest
import torch


class IntegralPredict3DTest(unittest.TestCase):

    def test_cpu_pred(self):
        from mankey.network.predict import get_integral_preds_3d_cpu
        num_keypoint = 2
        resolution = 32
        volume = torch.ones((1, num_keypoint * resolution, resolution, resolution))
        volume *= 1.0 / float(resolution * resolution * resolution)
        pred_x, pred_y, pred_z = get_integral_preds_3d_cpu(volume, num_keypoint, resolution, resolution, resolution)
        self.assertAlmostEqual(float(pred_x[0, 0, 0].item()), float(resolution) * 0.5 - 0.5)
        self.assertAlmostEqual(float(pred_x[0, 1, 0].item()), float(resolution) * 0.5 - 0.5)
        self.assertAlmostEqual(float(pred_y[0, 0, 0].item()), float(resolution) * 0.5 - 0.5)
        self.assertAlmostEqual(float(pred_y[0, 1, 0].item()), float(resolution) * 0.5 - 0.5)
        self.assertAlmostEqual(float(pred_z[0, 0, 0].item()), float(resolution) * 0.5 - 0.5)
        self.assertAlmostEqual(float(pred_z[0, 1, 0].item()), float(resolution) * 0.5 - 0.5)

    def test_gpu_pred(self):
        from mankey.network.predict import get_integral_preds_3d_gpu
        num_keypoint = 2
        resolution = 32
        volume = torch.ones((1, num_keypoint * resolution, resolution, resolution)).cuda()
        volume *= 1.0 / float(resolution * resolution * resolution)
        pred_x, pred_y, pred_z = get_integral_preds_3d_gpu(volume, num_keypoint, resolution, resolution, resolution)
        self.assertAlmostEqual(float(pred_x[0, 0, 0].item()), float(resolution) * 0.5 - 0.5)
        self.assertAlmostEqual(float(pred_x[0, 1, 0].item()), float(resolution) * 0.5 - 0.5)
        self.assertAlmostEqual(float(pred_y[0, 0, 0].item()), float(resolution) * 0.5 - 0.5)
        self.assertAlmostEqual(float(pred_y[0, 1, 0].item()), float(resolution) * 0.5 - 0.5)
        self.assertAlmostEqual(float(pred_z[0, 0, 0].item()), float(resolution) * 0.5 - 0.5)
        self.assertAlmostEqual(float(pred_z[0, 1, 0].item()), float(resolution) * 0.5 - 0.5)


class HeatMapTest(unittest.TestCase):

    def test_2d_heatmap_cpu(self):
        n_batch = 16
        n_keypoint = 6
        image_size = 256

        # Some random pred
        rand_pred = torch.rand(size=(n_batch, n_keypoint, image_size, image_size))
        from mankey.network.predict import heatmap_from_predict
        heatmap = heatmap_from_predict(rand_pred, n_keypoint)

        # Check it
        for batch_idx in range(n_batch):
            for keypoint_idx in range(n_keypoint):
                prob_value = heatmap[batch_idx, keypoint_idx, :, :].sum()
                self.assertTrue(abs(float(prob_value.item()) - 1.0) < 1e-4)

    def test_2d_heatmap_gpu(self):
        n_batch = 16
        n_keypoint = 6
        image_size = 256

        # Some random pred
        rand_pred = torch.rand(size=(n_batch, n_keypoint, image_size, image_size))
        rand_pred = rand_pred.cuda()
        from mankey.network.predict import heatmap_from_predict
        heatmap = heatmap_from_predict(rand_pred, n_keypoint)

        # Check it
        for batch_idx in range(n_batch):
            for keypoint_idx in range(n_keypoint):
                prob_value = heatmap[batch_idx, keypoint_idx, :, :].sum()
                self.assertTrue(abs(float(prob_value.item()) - 1.0) < 1e-4)


class CoordinateIntegrationTest(unittest.TestCase):

    def test_imgcoord_cpu(self):
        n_batch = 16
        n_keypoint = 6
        image_size = 256

        # Some random pred
        rand_pred = torch.rand(size=(n_batch, n_keypoint, image_size, image_size))
        from mankey.network.predict import heatmap_from_predict, heatmap2d_to_imgcoord_cpu
        heatmap = heatmap_from_predict(rand_pred, n_keypoint)
        coord_x, coord_y = heatmap2d_to_imgcoord_cpu(heatmap, num_keypoints=n_keypoint)

        # Check the size
        self.assertEqual(coord_x.shape, (n_batch, n_keypoint, 1))
        self.assertEqual(coord_y.shape, (n_batch, n_keypoint, 1))

        # Check the value, method can be slow
        check_batch_idx = 0
        check_keypoint_idx = 0
        specific_heatmap = heatmap[check_batch_idx, check_keypoint_idx, :, :].numpy()
        x_value = 0
        y_value = 0
        for y_idx in range(image_size):
            for x_idx in range(image_size):
                x_value += specific_heatmap[y_idx, x_idx] * x_idx
                y_value += specific_heatmap[y_idx, x_idx] * y_idx

        # Compare with original value
        x_pred = float(coord_x[check_batch_idx, check_keypoint_idx, 0].item())
        y_pred = float(coord_y[check_batch_idx, check_keypoint_idx, 0].item())
        self.assertTrue(abs(x_value - x_pred) < 1e-4)
        self.assertTrue(abs(y_value - y_pred) < 1e-4)

    def test_imgcoord_gpu(self):
        n_batch = 16
        n_keypoint = 6
        image_size = 256

        # Some random pred
        rand_pred = torch.rand(size=(n_batch, n_keypoint, image_size, image_size))
        rand_pred = rand_pred.cuda()
        from mankey.network.predict import heatmap_from_predict, heatmap2d_to_imgcoord_gpu
        heatmap = heatmap_from_predict(rand_pred, n_keypoint)
        coord_x, coord_y = heatmap2d_to_imgcoord_gpu(heatmap, num_keypoints=n_keypoint)

        # Check the size
        self.assertEqual(coord_x.shape, (n_batch, n_keypoint, 1))
        self.assertEqual(coord_y.shape, (n_batch, n_keypoint, 1))

        # Check the value, method can be slow
        check_batch_idx = 0
        check_keypoint_idx = 0
        specific_heatmap = heatmap[check_batch_idx, check_keypoint_idx, :, :].cpu().numpy()
        x_value = 0
        y_value = 0
        for y_idx in range(image_size):
            for x_idx in range(image_size):
                x_value += specific_heatmap[y_idx, x_idx] * x_idx
                y_value += specific_heatmap[y_idx, x_idx] * y_idx

        # Compare with original value
        x_pred = float(coord_x[check_batch_idx, check_keypoint_idx, 0].item())
        y_pred = float(coord_y[check_batch_idx, check_keypoint_idx, 0].item())
        self.assertTrue(abs(x_value - x_pred) < 1e-4)
        self.assertTrue(abs(y_value - y_pred) < 1e-4)

    def test_depth_integration(self):
        n_batch = 16
        n_keypoint = 6
        image_size = 256

        # Some random pred
        rand_pred = torch.rand(size=(n_batch, n_keypoint, image_size, image_size))
        from mankey.network.predict import heatmap_from_predict, depth_integration
        heatmap = heatmap_from_predict(rand_pred, n_keypoint)
        depth_pred = depth_integration(heatmap, rand_pred)

        # Check the value, method can be slow
        check_batch_idx = 0
        check_keypoint_idx = 0
        specific_heatmap = heatmap[check_batch_idx, check_keypoint_idx, :, :].numpy()
        specific_depthmap = rand_pred[check_batch_idx, check_keypoint_idx, :, :].numpy()
        depth_value = 0
        for y_idx in range(image_size):
            for x_idx in range(image_size):
                depth_value += specific_heatmap[y_idx, x_idx] * specific_depthmap[y_idx, x_idx]

        # Compare with the original value
        d_pred = float(depth_pred[check_batch_idx, check_keypoint_idx, 0].item())
        self.assertTrue(abs(depth_value - d_pred) < 1e-4)


if __name__ == '__main__':
    unittest.main()
