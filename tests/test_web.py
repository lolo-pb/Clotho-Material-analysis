import asyncio
import base64
import io
import unittest
import zipfile
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
    state = SimpleNamespace(
        model=ConstantModel(),
        device=torch.device("cpu"),
        auth_token=None,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


class RawRequest:
    def __init__(self, request: SimpleNamespace, data: bytes, filename: str):
        self.app = request.app
        self.headers = {
            "X-Clotho-Filename": filename,
            "X-Clotho-Session": getattr(request.app.state, "auth_token", None),
        }
        self._data = data

    async def stream(self):
        yield self._data


class WebTests(unittest.TestCase):
    def tearDown(self):
        web.app.state.auth_token = None
        web.BATCH_SESSIONS.clear()

    def test_health_requires_the_configured_local_session(self):
        web.app.state.auth_token = "desktop-token"
        request = SimpleNamespace(
            app=web.app,
            headers={"X-Clotho-Session": "desktop-token"},
        )
        self.assertEqual(web.health(request), {"status": "ready"})

        request.headers = {}
        with self.assertRaises(HTTPException) as raised:
            web.health(request)
        self.assertEqual(raised.exception.status_code, 403)

    def test_home_contains_upload_and_result_controls(self):
        html = web.home()
        self.assertIn('id="file-input"', html)
        self.assertIn('id="analyze"', html)
        self.assertIn('id="result-mask"', html)
        self.assertIn('id="csv-download"', html)
        self.assertIn('id="selection-list"', html)
        self.assertIn('id="batch-progress-bar"', html)
        self.assertIn('id="batch-detail"', html)

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

    def test_prediction_requires_the_configured_local_session(self):
        request = request_with_model()
        request.app.state.auth_token = "desktop-token"
        request.headers = {}
        upload = UploadFile(file=io.BytesIO(encode_test_image()), filename="sample.png")

        with self.assertRaises(HTTPException) as raised:
            web.predict(request, upload)
        self.assertEqual(raised.exception.status_code, 403)

    def test_batch_keeps_results_in_memory_and_downloads_a_zip(self):
        request = request_with_model()
        created = web.create_batch(request)
        batch_id = created["batch_id"]
        upload = RawRequest(request, encode_test_image(), "sample image.png")

        result = asyncio.run(web.add_batch_image(batch_id, upload))

        self.assertEqual(result["filename"], "sample image.png")
        self.assertEqual(result["summary"]["successful_images"], 1)
        self.assertEqual(result["result_index"], 0)
        self.assertEqual(len(web.BATCH_SESSIONS[batch_id].results), 1)
        self.assertTrue(result["thumbnail_data_url"].startswith("data:image/png;base64,"))

        overlay = web.batch_overlay(batch_id, result["result_index"], request)
        self.assertEqual(overlay.media_type, "image/png")
        self.assertTrue(overlay.body.startswith(b"\x89PNG"))

        archive = web.download_batch(batch_id, request)
        with zipfile.ZipFile(io.BytesIO(archive.body)) as downloaded:
            self.assertEqual(
                set(downloaded.namelist()),
                {
                    "masks/001-sample image-mask.png",
                    "overlays/001-sample image-overlay.png",
                    "statistics.csv",
                    "batch-summary.csv",
                    "image-variation.csv",
                },
            )
            self.assertIn("sample image.png", downloaded.read("statistics.csv").decode())
        self.assertIn(batch_id, web.BATCH_SESSIONS)

    def test_batch_records_invalid_image_and_keeps_processing(self):
        request = request_with_model()
        batch_id = web.create_batch(request)["batch_id"]
        bad_upload = RawRequest(request, b"not an image", "bad.png")

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(web.add_batch_image(batch_id, bad_upload))
        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(web.BATCH_SESSIONS[batch_id].errors[0]["filename"], "bad.png")

        good_upload = RawRequest(request, encode_test_image(), "good.png")
        result = asyncio.run(web.add_batch_image(batch_id, good_upload))
        self.assertEqual(result["summary"]["successful_images"], 1)
        self.assertEqual(result["summary"]["failed_images"], 1)

    def test_batch_upload_size_limit_is_recorded_as_a_failure(self):
        request = request_with_model()
        batch_id = web.create_batch(request)["batch_id"]
        with patch.object(web, "MAX_UPLOAD_BYTES", 8):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(web.add_batch_image(batch_id, RawRequest(request, b"123456789", "large.png")))
        self.assertEqual(raised.exception.status_code, 413)
        self.assertEqual(web.BATCH_SESSIONS[batch_id].errors[0]["filename"], "large.png")

    def test_cancelled_batch_rejects_the_next_image_and_can_be_discarded(self):
        request = request_with_model()
        batch_id = web.create_batch(request)["batch_id"]
        self.assertEqual(web.cancel_batch(batch_id, request), {"cancelled": True})

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(web.add_batch_image(batch_id, RawRequest(request, encode_test_image(), "sample.png")))
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(web.discard_batch(batch_id, request), {"discarded": True})
        self.assertNotIn(batch_id, web.BATCH_SESSIONS)

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
