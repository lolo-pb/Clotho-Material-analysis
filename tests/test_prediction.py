import unittest

import numpy as np
import torch
from torch import nn

from predicting.predict import predict_image, tile_starts


class ConstantModel(nn.Module):
    def forward(self, image):
        batch, _, height, width = image.shape
        output = torch.zeros(batch, 4, height, width, device=image.device)
        output[:, 2] = 1
        return output


class PredictionTests(unittest.TestCase):
    def test_tile_starts_covers_end(self):
        self.assertEqual(tile_starts(1000, 512, 128), [0, 384, 488])

    def test_tiled_prediction_preserves_size(self):
        image = np.zeros((173, 219, 3), dtype=np.uint8)
        prediction = predict_image(
            ConstantModel(), image, torch.device("cpu"), tile_size=128, overlap=32
        )
        self.assertEqual(prediction.shape, image.shape[:2])
        self.assertTrue(np.all(prediction == 2))


if __name__ == "__main__":
    unittest.main()
