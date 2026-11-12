"""Regression tests for filling enclosed holes in exported room footprints."""

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

    def test_fill_small_and_large_holes_but_keep_exterior_notch(self):
        import numpy as np
        from shapely.geometry import Point

        for resolution in (0.05, 0.1):
            with self.subTest(resolution=resolution):
                labels = np.full((24, 36), 7, dtype=np.int32)
                labels[2, 2] = -1
                labels[2:18, 17:33] = -1
                labels[15:17, :2] = -1  # Open to the exterior.
                original = labels.copy()

                footprint = self.export(labels, resolution, [7])[7]

                self.assertTrue(footprint.covers(Point(2.5 * resolution, 2.5 * resolution)))
                self.assertTrue(footprint.covers(Point(17.5 * resolution, 2.5 * resolution)))
                self.assertFalse(footprint.covers(Point(0.5 * resolution, 15.5 * resolution)))
                self.assertEqual(len(footprint.interiors), 0)
                self.assertAlmostEqual(footprint.area, (labels.size - 4) * resolution**2)
                np.testing.assert_array_equal(labels, original)

    def test_hole_containing_another_room_is_protected_when_filtered(self):
        import numpy as np
        from shapely.geometry import Point

        for side in (10, 270):
            labels = np.zeros((side, side), dtype=np.int32)
            labels[2:-2, 2:-2] = -1
            # Place the nested room in the final row batch of the larger hole.
            labels[-3, -3] = 9
            point = Point((side - 2.5) * 0.1, (side - 2.5) * 0.1)
            for room_ids in ([0], [0, 9]):
                with self.subTest(side=side, room_ids=room_ids):
                    rooms = self.export(labels, 0.1, room_ids)
                    self.assertEqual(set(rooms), set(room_ids))
                    self.assertEqual(len(rooms[0].interiors), 1)
                    self.assertFalse(rooms[0].covers(point))
                    if 9 in rooms:
                        self.assertTrue(rooms[9].covers(point))
                        self.assertEqual(rooms[0].intersection(rooms[9]).area, 0.)

    def test_filled_hole_absorbs_a_disconnected_part_of_the_same_room(self):
        import numpy as np
        from shapely.geometry import box

        labels = np.zeros((10, 10), dtype=np.int32)
        labels[3:6, 3:6] = -1
        labels[4, 4] = 0

        footprint = self.export(labels, 0.1, [0])[0]

        self.assertTrue(footprint.equals(box(0, 0, 1, 1)))

    def test_separate_parts_of_same_room_keep_gap_to_exterior(self):
        import numpy as np
        from shapely.geometry import Point

        labels = np.zeros((8, 20), dtype=np.int32)
        labels[:, 8:12] = -1
        labels[2:5, 2:5] = -1
        labels[2:5, 14:17] = -1

        footprint = self.export(labels, 0.1, [0])[0]

        self.assertEqual(footprint.geom_type, "MultiPolygon")
        self.assertEqual(len(footprint.geoms), 2)
        self.assertTrue(all(not part.interiors for part in footprint.geoms))
        self.assertFalse(footprint.covers(Point(1.0, 0.4)))
