import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.labels import CLASS_COLORS_RGB, encode_mask, read_rgb, write_rgb


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PERCENT_COLUMNS = ("pores", "fibers", "resin", "undefined", "sumcheck")
PIXEL_COLUMNS = ("pores_pixels", "fibers_pixels", "resin_pixels", "undefined_pixels")


def find_masks(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    image_paths = [
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    mask_paths = [path for path in image_paths if path.stem.endswith("-mask")]
    return mask_paths if mask_paths else image_paths


def crop_bottom(image: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return image
    if pixels >= image.shape[0]:
        raise ValueError(
            f"Cannot crop {pixels} pixels from image with height {image.shape[0]}"
        )
    return image[:-pixels, :, :]


def get_mask_statistics(mask_path: Path, crop_bottom_pixels: int = 0) -> dict[str, object]:
    mask = crop_bottom(read_rgb(mask_path), crop_bottom_pixels)
    class_ids, invalid = encode_mask(mask)
    total_pixels = class_ids.size

    fiber_pixels = int(np.count_nonzero(class_ids == 0))
    resin_pixels = int(np.count_nonzero(class_ids == 1))
    pore_pixels = int(np.count_nonzero(class_ids == 2))
    undefined_pixels = int(np.count_nonzero(class_ids == 3))
    invalid_pixels = int(np.count_nonzero(invalid))

    counted_pixels = fiber_pixels + resin_pixels + pore_pixels + undefined_pixels
    return {
        "Image Paths": str(mask_path),
        "pores": pore_pixels / total_pixels * 100,
        "fibers": fiber_pixels / total_pixels * 100,
        "resin": resin_pixels / total_pixels * 100,
        "undefined": undefined_pixels / total_pixels * 100,
        "sumcheck": counted_pixels / total_pixels * 100,
        "total_pixels": total_pixels,
        "pores_pixels": pore_pixels,
        "fibers_pixels": fiber_pixels,
        "resin_pixels": resin_pixels,
        "undefined_pixels": undefined_pixels,
        "invalid_pixels": invalid_pixels,
        "invalid_percent": invalid_pixels / total_pixels * 100,
    }


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = (
        ["Image Paths"]
        + list(PERCENT_COLUMNS)
        + ["total_pixels"]
        + list(PIXEL_COLUMNS)
        + ["invalid_pixels", "invalid_percent"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_statistics_report(
    mask_path: Path,
    statistics: dict[str, object],
    output_path: Path,
    crop_bottom_pixels: int = 0,
) -> None:
    mask = crop_bottom(read_rgb(mask_path), crop_bottom_pixels)
    height, width = mask.shape[:2]
    panel_width = 500
    report = np.full((height, width + panel_width, 3), 255, dtype=np.uint8)
    report[:, :width] = mask

    x = width + 32
    cv2.putText(
        report,
        "Segmentation statistics",
        (x, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        report,
        mask_path.stem.removesuffix("-mask"),
        (x, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )

    labels = (
        ("Fiber", "fibers", CLASS_COLORS_RGB[0]),
        ("Resin", "resin", CLASS_COLORS_RGB[1]),
        ("Pore", "pores", CLASS_COLORS_RGB[2]),
        ("Unidentified", "undefined", CLASS_COLORS_RGB[3]),
    )
    for index, (label, key, color) in enumerate(labels):
        y = 155 + index * 72
        cv2.rectangle(report, (x, y - 24), (x + 36, y + 12), tuple(map(int, color)), -1)
        cv2.rectangle(report, (x, y - 24), (x + 36, y + 12), (90, 90, 90), 1)
        cv2.putText(
            report,
            f"{label}: {statistics[key]:.2f}%",
            (x + 54, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        report,
        f"Counted: {statistics['sumcheck']:.2f}%",
        (x, 480),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        report,
        f"Invalid colors: {statistics['invalid_percent']:.2f}%",
        (x, 525),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        report,
        f"Pixels: {statistics['total_pixels']:,}",
        (x, 570),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    write_rgb(output_path, report)


def write_statistics_reports(
    rows: list[dict[str, object]],
    output_dir: Path,
    crop_bottom_pixels: int,
) -> None:
    for row in rows:
        mask_path = Path(str(row["Image Paths"]))
        output_path = output_dir / f"{mask_path.stem.removesuffix('-mask')}-statistics.png"
        write_statistics_report(mask_path, row, output_path, crop_bottom_pixels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate class percentages from exact-color segmentation masks."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Mask image, or a folder containing predicted *-mask images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("statistics/statistics.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--crop-bottom",
        type=int,
        default=0,
        help="Pixels to remove from the bottom before counting.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Optional folder for mask-and-statistics PNG reports.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mask_paths = find_masks(args.input_path)
    if not mask_paths:
        print(f"No supported mask images found in {args.input_path}")
        return 1

    rows = [
        get_mask_statistics(mask_path, crop_bottom_pixels=args.crop_bottom)
        for mask_path in mask_paths
    ]
    write_csv(rows, args.output)
    print(f"Wrote statistics for {len(rows)} image(s) to {args.output}")
    if args.report_dir:
        write_statistics_reports(rows, args.report_dir, args.crop_bottom)
        print(f"Wrote report images to {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
