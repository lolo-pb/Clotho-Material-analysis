import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from common.dataset import normalize_image
from common.labels import decode_mask, read_rgb, write_rgb
from common.model import FiberglassUNet


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


@torch.inference_mode()
def predict_image(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 128,
) -> np.ndarray:
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be at least 0 and smaller than tile size")

    original_height, original_width = image_rgb.shape[:2]
    bottom = max(0, tile_size - original_height)
    right = max(0, tile_size - original_width)
    padded = cv2.copyMakeBorder(
        image_rgb, 0, bottom, 0, right, cv2.BORDER_REFLECT_101
    )
    height, width = padded.shape[:2]
    logits_sum = np.zeros((4, height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    model.eval()
    for top in tile_starts(height, tile_size, overlap):
        for left in tile_starts(width, tile_size, overlap):
            tile = padded[top : top + tile_size, left : left + tile_size]
            tensor = normalize_image(tile).unsqueeze(0).to(device)
            logits = model(tensor)[0].cpu().numpy()
            logits_sum[:, top : top + tile_size, left : left + tile_size] += logits
            counts[top : top + tile_size, left : left + tile_size] += 1

    prediction = np.argmax(logits_sum / counts[None, :, :], axis=0)
    return prediction[:original_height, :original_width].astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment a fiberglass micrograph.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/final.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("predicting/outputs"))
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = FiberglassUNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])

    image = read_rgb(args.image)
    class_ids = predict_image(
        model, image, device, tile_size=args.tile_size, overlap=args.overlap
    )
    mask = decode_mask(class_ids)
    overlay = cv2.addWeighted(image, 0.55, mask, 0.45, 0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    write_rgb(args.output_dir / f"{stem}-mask.png", mask)
    write_rgb(args.output_dir / f"{stem}-overlay.png", overlay)
    print(f"Saved results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
