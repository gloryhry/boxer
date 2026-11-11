"""Regression tests for filling small holes in exported room footprints."""

import unittest
from importlib.util import find_spec


@unittest.skipUnless(
    find_spec("numpy") and find_spec("shapely"),
    "Install NumPy and Shapely to exercise room polygon exports",
)
class TestRoomPolygonHoles(unittest.TestCase):
    def export(self, labels, resolution, room_ids):
        from shapely.geometry import shape

        from utils.voxel_connectivity_runtime.room_polygons import room_polygons

        height, width = labels.shape
        transform = dict(height=height, width=width, row_order="y_up", resolution=resolution, origin=[0., 0.])
        result = room_polygons(labels, transform, room_ids)
        rooms = {}
        for room in result["rooms"]:
            footprint = shape(room["geometry"])
            self.assertTrue(footprint.is_valid)
            self.assertAlmostEqual(footprint.area, room["area_m2"])
            rooms[room["id"]] = footprint
        return rooms

    def test_fill_inclusive_metric_threshold_but_preserve_large_hole_and_notch(self):
        import numpy as np
        from shapely.geometry import Point

        for resolution, side in ((0.05, 10), (0.1, 5)):
            with self.subTest(resolution=resolution):
                labels = np.full((24, 32), 7, dtype=np.int32)
                labels[2:2 + side, 2:2 + side] = -1
                labels[2:2 + side, 17:17 + side] = -1
                labels[2 + side, 17] = -1  # One cell over 0.25 m².
                labels[15:17, :2] = -1  # Open to the exterior.
                original = labels.copy()

                footprint = self.export(labels, resolution, [7])[7]

                self.assertTrue(footprint.covers(Point(2.5 * resolution, 2.5 * resolution)))
                self.assertFalse(footprint.covers(Point(17.5 * resolution, 2.5 * resolution)))
                self.assertFalse(footprint.covers(Point(0.5 * resolution, 15.5 * resolution)))
                self.assertEqual(len(footprint.interiors), 1)
                self.assertAlmostEqual(footprint.area, (labels.size - side**2 - 1 - 4) * resolution**2)
                np.testing.assert_array_equal(labels, original)

    def test_hole_containing_another_room_is_protected_when_filtered(self):
        import numpy as np
        from shapely.geometry import Point

        labels = np.zeros((10, 10), dtype=np.int32)
        labels[3:6, 3:6] = -1
        labels[4, 4] = 9

        for room_ids in ([0], [0, 9]):
            with self.subTest(room_ids=room_ids):
                rooms = self.export(labels, 0.1, room_ids)
                self.assertEqual(set(rooms), set(room_ids))
                self.assertEqual(len(rooms[0].interiors), 1)
                self.assertFalse(rooms[0].covers(Point(0.45, 0.45)))
                if 9 in rooms:
                    self.assertTrue(rooms[9].covers(Point(0.45, 0.45)))
                    self.assertEqual(rooms[0].intersection(rooms[9]).area, 0.)

    def test_filled_hole_absorbs_a_disconnected_part_of_the_same_room(self):
        import numpy as np
        from shapely.geometry import box

        labels = np.zeros((10, 10), dtype=np.int32)
        labels[3:6, 3:6] = -1
        labels[4, 4] = 0

        footprint = self.export(labels, 0.1, [0])[0]

        self.assertTrue(footprint.equals(box(0, 0, 1, 1)))
