import tempfile
import unittest
from pathlib import Path

import numpy as np

from common.labels import decode_mask, encode_mask, read_rgb, write_rgb
from common.validation import validate_dataset


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    blended = 0.55 * image.astype(np.float32) + 0.45 * mask.astype(np.float32)
    return np.rint(blended).clip(0, 255).astype(np.uint8)


class ValidationTests(unittest.TestCase):
    def test_valid_pair_is_cleaned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            masks = root / "masks"
            cleaned = root / "cleaned"
            images.mkdir()
            masks.mkdir()

            image = np.full((8, 8, 3), 120, dtype=np.uint8)
            class_ids = np.zeros((8, 8), dtype=np.uint8)
            class_ids[:, 4:] = 1
            mask = decode_mask(class_ids)
            mask[0, 0] = (1, 2, 3)
            write_rgb(images / "sample.jpg", image)
            write_rgb(masks / "sample.png", mask)

            result = validate_dataset(images, masks, cleaned)

            self.assertTrue(result.ok)
            self.assertEqual(len(result.warnings), 1)
            self.assertTrue((cleaned / "sample.png").exists())
            cleaned_ids, invalid = encode_mask(read_rgb(cleaned / "sample.png"))
            self.assertFalse(invalid.any())
            self.assertEqual(cleaned_ids[0, 0], 0)

    def test_missing_mask_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            write_rgb(images / "sample.jpg", np.zeros((8, 8, 3), dtype=np.uint8))

            result = validate_dataset(images, masks)

            self.assertFalse(result.ok)
            self.assertIn("missing PNG mask", result.errors[0])

    def test_with_overlays_converts_overlay_to_clean_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            masks = root / "masks"
            cleaned = root / "cleaned"
            images.mkdir()
            masks.mkdir()

            image = np.full((8, 8, 3), 120, dtype=np.uint8)
            class_ids = np.zeros((8, 8), dtype=np.uint8)
            class_ids[:, 4:] = 1
            mask = decode_mask(class_ids)
            overlay = make_overlay(image, mask)
            write_rgb(images / "sample.jpg", image)
            write_rgb(masks / "sample.png", overlay)

            result = validate_dataset(images, masks, cleaned, with_overlays=True)

            self.assertTrue(result.ok)
            self.assertIn("converted overlay to mask", result.warnings[0])
            cleaned_ids, invalid = encode_mask(read_rgb(cleaned / "sample.png"))
            self.assertFalse(invalid.any())
            np.testing.assert_array_equal(cleaned_ids, class_ids)

    def test_with_overlays_keeps_exact_mask_as_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            masks = root / "masks"
            cleaned = root / "cleaned"
            images.mkdir()
            masks.mkdir()

            image = np.full((8, 8, 3), 120, dtype=np.uint8)
            class_ids = np.zeros((8, 8), dtype=np.uint8)
            class_ids[:, 4:] = 2
            mask = decode_mask(class_ids)
            write_rgb(images / "sample.jpg", image)
            write_rgb(masks / "sample.png", mask)

            result = validate_dataset(images, masks, cleaned, with_overlays=True)

            self.assertTrue(result.ok)
            self.assertEqual(result.warnings, [])
            cleaned_ids, invalid = encode_mask(read_rgb(cleaned / "sample.png"))
            self.assertFalse(invalid.any())
            np.testing.assert_array_equal(cleaned_ids, class_ids)

    def test_with_overlays_rejects_original_image_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            masks = root / "masks"
            cleaned = root / "cleaned"
            images.mkdir()
            masks.mkdir()

            image = np.full((8, 8, 3), 120, dtype=np.uint8)
            write_rgb(images / "sample.png", image)
            write_rgb(masks / "sample.png", image)

            result = validate_dataset(images, masks, cleaned, with_overlays=True)

            self.assertFalse(result.ok)
            self.assertIn("copy of the original image", result.errors[0])


if __name__ == "__main__":
    unittest.main()
