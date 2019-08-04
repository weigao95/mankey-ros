import unittest


class BBoxRectificationTest(unittest.TestCase):

    def test_center_aligned(self):
        from mankey.utils.imgproc import PixelCoord, rectify_bbox_center_align

        # Some test of rectification
        topleft, bottomright = PixelCoord(), PixelCoord()
        topleft.x = 0
        topleft.y = 10
        bottomright.x = 20
        bottomright.y = 50
        
        # Test of center-aligned
        rectified_topleft, rectified_bottomright = rectify_bbox_center_align(topleft, bottomright)
        self.assertEqual(rectified_bottomright.x - rectified_topleft.x, rectified_bottomright.y - rectified_topleft.y)
        self.assertEqual(rectified_bottomright.x + rectified_topleft.x, topleft.x + bottomright.x)
        self.assertEqual(rectified_bottomright.y + rectified_topleft.y, topleft.y + bottomright.y)

    def test_in_image(self):
        from mankey.utils.imgproc import PixelCoord, rectify_bbox_in_image

        # Some test of rectification
        topleft, bottomright = PixelCoord(), PixelCoord()
        topleft.x = 0
        topleft.y = 10
        bottomright.x = 20
        bottomright.y = 50
        
        # Test of in_image
        rectified_topleft, rectified_bottomright = rectify_bbox_in_image(topleft, bottomright, 640, 480)
        self.assertEqual(rectified_bottomright.x - rectified_topleft.x, rectified_bottomright.y - rectified_topleft.y)
        self.assertEqual(rectified_bottomright.x - rectified_topleft.x, 40)


if __name__ == '__main__':
    unittest.main()
