import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.dataset import SegmentationDataset
from common.loss import segmentation_loss
from common.model import FiberglassUNet
from common.validation import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fiberglass segmentation model.")
    parser.add_argument("--images", type=Path, default=Path("training/images"))
    parser.add_argument("--masks", type=Path, default=Path("training/masks"))
    parser.add_argument(
        "--validated-masks", type=Path, default=Path("training/validated_masks")
    )
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--patches-per-image", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--with-overlays",
        action="store_true",
        help="Allow mixed exact masks and corrected overlays in the masks folder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation = validate_dataset(
        args.images,
        args.masks,
        args.validated_masks,
        with_overlays=args.with_overlays,
    )
    for warning in validation.warnings:
        print(f"WARNING: {warning}")
    if not validation.ok:
        for error in validation.errors:
            print(f"ERROR: {error}")
        print(
            "Training stopped: fix the dataset and run validate_data.py first."
            " Use --with-overlays if the masks folder contains corrected overlays."
        )
        return 1

    print("    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⡄⠈⠀⢀⣀⡤⢄⠒⠒⡖⠒⡒⠒⡒⠒⢦⣤⡔⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⢀⢄⠐⡰⢞⠡⢂⢁⠂⠌⡐⠠⠡⠐⠡⡀⢣⢇⠯⣙⢦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⢉⣀⡠⢋⠠⡁⢂⡔⢦⣬⠴⠆⢃⠂⢡⠡⠐⢌⠾⣌⢣⠞⣳⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⡜⠔⠆⡈⢊⠱⠚⠓⠋⢉⠉⣉⢊⣥⣓⣲⣒⠤⡕⣣⠞⣡⢷⡐⠒⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠸⣆⣲⣸⡄⠂⠌⠤⢁⡘⢀⢢⡿⠊⠀⠀⠀⠈⠙⣵⠜⣜⣡⣞⣇⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⡔⠋⠉⠙⢻⡀⠒⡠⠐⠂⣿⠀⠀⠀⠀⣀⠀⠐⠘⡯⡔⢦⣉⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠸⠀⠀⠁⠀⢀⡇⡡⢄⢳⣣⢿⡄⠀⠀⠀⠀⠀⡐⢴⡟⡜⣢⢥⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⢱⣤⣤⣤⡾⢁⠸⠗⡈⠳⢁⠺⣦⣀⣀⣀⣠⣴⣿⣹⢒⣥⣾⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⢣⣉⣋⣐⡀⠬⡤⣤⣥⣤⣆⡙⢭⣛⢛⡻⣙⢸⢧⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠉⠉⠉⢀⣠⣹⣧⣀⣻⣷⣿⣦⠱⡣⠖⣍⣾⣛⢾⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⢀⣜⠛⣿⣿⣿⣿⣏⣀⣿⢃⡏⣭⣷⠟⠁⣾⣹⣿⣿⣿⣿⣿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠨⠌⡍⣉⢉⡉⣉⢉⠭⠯⣞⣼⣾⣆⠁⡆⣟⣯⣿⣿⣿⣿⣿⣿⣿⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠀⠀⠪⠤⠬⠦⠥⣬⣶⡿⣻⣿⣿⠯⣤⢼⣏⢷⣻⣿⣿⣿⣿⣿⣿⣿⣿⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣏⣷⣭⣽⡟⠰⡁⣾⢎⡯⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡿⢿⣿⣿⡇⡆⢼⢏⠮⣽⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣶⣿⠁⡀⠘⣏⡞⣵⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣬⣽⣛⣻⠿⣿⡀⠁⠀⢹⡜⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⢿⣿⣿⣿⣿⣿⡟⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⣷⣶⣭⣟⡻⢿⣿⣿⣧⠀⠀⢸⣿⣿⣿⣿⡭⢙⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠦⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⢽⣿⣷⣄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣷⣮⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}.")
    dataset = SegmentationDataset(
        validation.pairs,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    model = FiberglassUNet(pretrained=not args.no_pretrained).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"]

    args.checkpoints.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            loss = segmentation_loss(model(images), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {average_loss:.4f}")
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "classes": 4,
        }
        torch.save(checkpoint, args.checkpoints / "latest.pt")
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, args.checkpoints / f"epoch-{epoch + 1}.pt")

    torch.save(checkpoint, args.checkpoints / "final.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
