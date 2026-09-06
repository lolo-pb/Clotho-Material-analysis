import base64
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from fastapi import HTTPException, UploadFile
from torch import nn

import web


class ConstantModel(nn.Module):
    def forward(self, image):
        batch, _, height, width = image.shape
        output = torch.zeros(batch, 4, height, width, device=image.device)
        output[:, 2] = 1
        return output


def encode_test_image(height: int = 64, width: int = 80) -> bytes:
    image = np.full((height, width, 3), 120, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Could not create test image")
    return encoded.tobytes()


def request_with_model() -> SimpleNamespace:
    state = SimpleNamespace(model=ConstantModel(), device=torch.device("cpu"))
    return SimpleNamespace(app=SimpleNamespace(state=state))


class WebTests(unittest.TestCase):
    def test_home_contains_upload_and_result_controls(self):
        html = web.home()
        self.assertIn('id="file-input"', html)
        self.assertIn('id="analyze"', html)
        self.assertIn('id="result-mask"', html)
        self.assertIn('id="csv-download"', html)

    def test_prediction_returns_pngs_and_pore_statistics(self):
        upload = UploadFile(file=io.BytesIO(encode_test_image()), filename="sample.png")
        result = web.predict(request_with_model(), upload)

        self.assertEqual(result["total_pixels"], 64 * 80)
        self.assertEqual(result["statistics"]["pore"]["percent"], 100.0)
        self.assertEqual(result["statistics"]["fiber"]["pixels"], 0)
        self.assertTrue(result["original_data_url"].startswith("data:image/png;base64,"))

        prefix, payload = result["mask_data_url"].split(",", 1)
        self.assertEqual(prefix, "data:image/png;base64")
        mask = cv2.imdecode(
            np.frombuffer(base64.b64decode(payload), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(mask.shape[:2], (64, 80))
        np.testing.assert_array_equal(mask[0, 0], (255, 0, 0))

        csv_payload = result["statistics_csv_data_url"].split(",", 1)[1]
        csv_text = base64.b64decode(csv_payload).decode("utf-8")
        lines = csv_text.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("pore_pixels,pore_percent", lines[0])
        self.assertIn("5120,100.000000", lines[1])

    def test_empty_and_unreadable_uploads_are_rejected(self):
        for data in (b"", b"not an image"):
            with self.subTest(data=data), self.assertRaises(HTTPException) as raised:
                upload = UploadFile(file=io.BytesIO(data), filename="bad.png")
                web.predict(request_with_model(), upload)
            self.assertEqual(raised.exception.status_code, 400)

    def test_upload_size_limit_is_enforced(self):
        with patch.object(web, "MAX_UPLOAD_BYTES", 8):
            upload = UploadFile(file=io.BytesIO(b"123456789"), filename="large.png")
            with self.assertRaises(HTTPException) as raised:
                web.predict(request_with_model(), upload)
        self.assertEqual(raised.exception.status_code, 413)

    def test_decoded_pixel_limit_is_enforced(self):
        with patch.object(web, "MAX_IMAGE_PIXELS", 10):
            upload = UploadFile(file=io.BytesIO(encode_test_image()), filename="large.png")
            with self.assertRaises(HTTPException) as raised:
                web.predict(request_with_model(), upload)
        self.assertEqual(raised.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
