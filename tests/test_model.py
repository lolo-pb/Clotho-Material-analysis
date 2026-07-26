import unittest

import torch

from common.model import FiberglassUNet


class ModelTests(unittest.TestCase):
    def test_output_matches_input_resolution(self):
        model = FiberglassUNet(pretrained=False).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 128, 160))
        self.assertEqual(output.shape, (1, 4, 128, 160))


if __name__ == "__main__":
    unittest.main()
