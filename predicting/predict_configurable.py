import argparse
from pathlib import Path

import cv2
import torch

from common.labels import decode_mask, read_rgb, write_rgb
from common.model import FiberglassUNet
from predicting.predict import predict_image


CHECKPOINT_PATH = Path("checkpoints/final.pt")
OUTPUT_DIR = Path("predicting/outputs")
TILE_SIZE = 512
OVERLAP = 128
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        return [
            path
            for path in sorted(input_path.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def load_model(device: torch.device) -> FiberglassUNet:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model = FiberglassUNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def save_prediction(model: FiberglassUNet, image_path: Path, device: torch.device) -> None:
    image = read_rgb(image_path)
    class_ids = predict_image(
        model,
        image,
        device,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
    )
    mask = decode_mask(class_ids)
    overlay = cv2.addWeighted(image, 0.55, mask, 0.45, 0)

    write_rgb(OUTPUT_DIR / f"{image_path.stem}-mask.png", mask)
    write_rgb(OUTPUT_DIR / f"{image_path.stem}-overlay.png", overlay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment one image or all images directly inside a folder."
    )
    parser.add_argument("input_path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_paths = find_input_images(args.input_path)
    if not image_paths:
        print(f"No supported images found in {args.input_path}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    print("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠋⠋⠉⠉⠉⠛⠛⠓⠈⠉⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿")
    print("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠋⣠⡶⢉⣀⣄⣀⠀⢀⣀⣀⡀⠀⠀⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿")
    print("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢉⣾⠟⣰⠏⠉⣽⣿⢊⣻⡿⠿⠛⢷⣶⡀⠀⢛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿")
    print("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠋⢠⡾⠋⢸⣧⣤⣾⡴⠀⠀⠉⠠⣤⣾⣧⡻⠃⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿")
    print("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢉⢀⣿⠁⠀⠈⠉⠈⠕⠀⠀⠀⠀⠀⠐⢭⡛⠋⠀⠀⠀⠈⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿")
    print("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢘⣼⠃⠀⠀⣀⡀⠄⠀⣠⠘⢧⡴⢢⡀⠀⢢⡀⠀⠀⠀⠀⢴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿")
    print("⣿⣿⣿⣿⣿⣯⣽⣿⣿⣿⣿⣿⣿⣿⡇⠺⣿⠃⠠⣄⢻⡇⠀⣴⣿⣿⣧⣿⣾⣇⠀⢨⣿⣤⡄⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿")
    print("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡗⠀⣿⣧⡆⠁⠙⠆⢠⢻⡿⠿⠟⡻⡿⠿⠆⠸⣟⣃⣣⣴⡦⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿")
    print("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⣾⢸⣿⡟⣿⠐⣩⣾⣧⣶⣤⣴⣬⣟⡀⠀⠀⠀⠈⠻⣿⣿⣷⠘⠘⠛⠛⠽⠋⠉⠟⢛⣩⣭⣉⣿")
    print("⠾⠿⡿⠛⠛⠛⣻⣿⣿⣿⣿⣿⠿⠁⠆⢸⣿⡇⠏⣰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⡄⢢⠀⠹⠌⠛⣿⣰⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿")
    print("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠠⡄⢸⣿⡇⡆⢻⣿⣿⣿⣿⣿⣿⡛⠝⠉⠀⠁⠀⠀⠐⠀⠀⠈⢯⣙⡿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿")
    print("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠁⣸⣿⣿⡇⠄⠿⣿⣿⣿⣿⣿⣿⡦⠀⠀⢀⡀⣀⠀⠀⠀⠀⠀⠉⠛⢾⣿⣿⣿⣿⣿⣿⣿⣿⣿")
    print("⣿⣿⣿⣿⠟⣻⣿⣿⣿⣿⢻⣿⡿⠇⣰⣿⣿⣿⣿⣴⢮⣈⡟⡟⠻⠿⠉⠀⠀⢠⣬⣿⡟⠁⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⡿⢿")
    print("⣿⣿⣿⢏⢌⠍⡩⠩⠈⠉⠠⠉⣠⣾⡿⣿⣛⣻⣿⡿⣧⣍⣾⣶⣶⣾⣶⠀⠀⠸⠝⠛⢆⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠿⢿⠻⠿⠈⠀")
    print("⣉⠋⠑⠁⠉⠈⠐⠁⠀⠀⣠⣾⣿⣻⠋⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠻⣶⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⠷⠚⠒⠒")
    print("⠿⠴⠡⠮⠼⠠⠂⠤⢠⣾⣿⣿⢾⣿⡴⠋⠴⠁⢉⠐⠛⣿⢿⠿⣿⠿⣿⣿⣿⣷⣦⡀⠀⠀⠀⢿⣿⣷⣀⣀⠀⠀⠀⠀⠀⢀⡀⠀⠀⠀")
    print("⣀⣀⠐⠀⠂⠂⠀⠀⢹⣿⣿⣷⡿⣿⢠⠥⠄⠀⢡⠂⠀⣠⠝⢹⡿⣧⣨⡟⠿⣙⣭⠉⡀⣀⡀⠀⠘⣿⡿⢿⡄⠀⠀⠀⠀⠀⠡⠀⠀⠀")
    print("⡀⢀⡀⠀⠀⠀⠀⠁⣾⣿⡿⣭⣶⣿⠞⡁⣶⣿⣿⣿⣿⣿⣴⣾⣽⣿⣿⣿⣶⣿⣷⣿⣿⣿⣷⠀⠀⢹⣿⣮⡌⣀⠀⠀⠀⠀⠀⠆⠀⠀")
    print("⠀⠘⠛⠁⣰⣿⡇⠀⣿⣿⣾⡿⣩⣾⣘⣸⣿⣿⣿⣿⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡄⠀⠀⠘⠛⣿⣟⠀⠀⠀⠀⠀⠀⡀⠀")
    print("⠀⠀⠀⠀⠉⠈⠀⣠⣿⣿⢋⠞⣹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠼⣿⣝⣦⢚⢀⠀⠀⢠⠙⠛")
    print("⣀⠀⠀⠀⠀⠀⠀⣿⣿⡿⣠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⢹⣯⠋⠈⢸⡟⡥⢪⠰⠦")
    print("⣿⣿⣿⣶⣶⣤⣼⣿⢛⣿⢳⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⣠⣼⣿⡯⠄⠀⢸⠇⠀⠘⠂⠀")
    print("⣿⣿⣿⣿⣿⣿⣿⣟⣼⣻⠈⠉⣸⣿⣿⢟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⢻⣿⣿⠁⠀⠀⠈⠀⡀⣀⢠⣴")
    print("⣿⣿⣿⣿⠟⠁⠠⣿⣿⠟⠁⡼⢻⣿⣿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⢹⠈⠀⣀⠁⠀⠀⠀⠁⠈⠈")
    print("⣿⡿⠟⠁⠀⠀⢠⣟⡍⠀⠈⣥⡟⢡⣿⡾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⠙⣤⠫⠄⠀⠀⠀⠀⠀⠀")

    for image_path in image_paths:
        save_prediction(model, image_path, device)
        print(f"Saved prediction for {image_path} to {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
