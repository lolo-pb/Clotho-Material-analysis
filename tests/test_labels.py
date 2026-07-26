import unittest

import numpy as np

from common.labels import (
    CLASS_COLORS_RGB,
    decode_mask,
    encode_mask,
    repair_invalid_colors,
)


class LabelTests(unittest.TestCase):
    def test_round_trip_all_classes(self):
        class_ids = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        encoded, invalid = encode_mask(decode_mask(class_ids))
        np.testing.assert_array_equal(encoded, class_ids)
        self.assertFalse(invalid.any())

    def test_invalid_color_becomes_unidentified(self):
        mask = CLASS_COLORS_RGB[[[0, 1], [2, 3]]].copy()
        mask[0, 0] = (12, 34, 56)
        encoded, invalid = encode_mask(mask)
        self.assertEqual(encoded[0, 0], 3)
        self.assertTrue(invalid[0, 0])

    def test_invalid_region_uses_most_common_colored_neighbors(self):
        class_ids = np.array(
            [
                [0, 0, 1],
                [0, 3, 1],
                [2, 3, 3],
            ],
            dtype=np.uint8,
        )
        invalid = np.zeros((3, 3), dtype=bool)
        invalid[1, 1] = True

        repaired = repair_invalid_colors(class_ids, invalid)

        self.assertEqual(repaired[1, 1], 0)

    def test_unidentified_does_not_vote(self):
        class_ids = np.full((3, 3), 3, dtype=np.uint8)
        invalid = np.zeros((3, 3), dtype=bool)
        invalid[1, 1] = True

        repaired = repair_invalid_colors(class_ids, invalid)

        self.assertEqual(repaired[1, 1], 3)


if __name__ == "__main__":
    unittest.main()
