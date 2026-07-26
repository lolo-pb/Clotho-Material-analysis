from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.labels import (
    CLASS_COLORS_RGB,
    decode_mask,
    encode_mask,
    read_rgb,
    repair_invalid_colors,
    write_rgb,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_LIKE_COLOR_DISTANCE = 12
MASK_LIKE_RATIO = 0.99
OVERLAY_IMAGE_WEIGHT = 0.55
OVERLAY_MASK_WEIGHT = 0.45


@dataclass(frozen=True)
class Pair:
    name: str
    image_path: Path
    mask_path: Path


@dataclass
class ValidationResult:
    pairs: list[Pair]
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def find_images(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def nearest_label_colors(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = image.astype(np.int32)
    colors = CLASS_COLORS_RGB.astype(np.int32)
    distances = np.sum(
        (pixels[:, :, None, :] - colors[None, None, :, :]) ** 2,
        axis=3,
    )
    class_ids = np.argmin(distances, axis=2).astype(np.uint8)
    nearest_distance = np.sqrt(np.min(distances, axis=2))
    return class_ids, nearest_distance


def is_mask_like(mask_or_overlay: np.ndarray) -> tuple[bool, float]:
    _, nearest_distance = nearest_label_colors(mask_or_overlay)
    mask_like_ratio = float(np.mean(nearest_distance <= MASK_LIKE_COLOR_DISTANCE))
    return mask_like_ratio >= MASK_LIKE_RATIO, mask_like_ratio


def overlay_to_class_ids(image: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    recovered = (
        overlay.astype(np.float32) - OVERLAY_IMAGE_WEIGHT * image.astype(np.float32)
    ) / OVERLAY_MASK_WEIGHT
    recovered = np.clip(np.rint(recovered), 0, 255).astype(np.uint8)
    class_ids, _ = nearest_label_colors(recovered)
    return class_ids


def validate_dataset(
    image_dir: Path,
    mask_dir: Path,
    clean_dir: Path | None = None,
    with_overlays: bool = False,
) -> ValidationResult:
    images = find_images(image_dir)
    masks = {
        path.stem: path
        for path in sorted(mask_dir.glob("*.png"))
        if path.is_file()
    }
    errors: list[str] = []
    warnings: list[str] = []
    pairs: list[Pair] = []

    for name in sorted(images.keys() - masks.keys()):
        errors.append(f"{name}: missing PNG mask")
    for name in sorted(masks.keys() - images.keys()):
        errors.append(f"{name}: mask has no matching image")

    for name in sorted(images.keys() & masks.keys()):
        image_path = images[name]
        mask_path = masks[name]
        image = read_rgb(image_path)
        mask = read_rgb(mask_path)

        if image.shape[:2] != mask.shape[:2]:
            errors.append(
                f"{name}: image is {image.shape[1]}x{image.shape[0]}, "
                f"mask is {mask.shape[1]}x{mask.shape[0]}"
            )
            continue

        mean_difference = float(
            np.mean(np.abs(image.astype(np.int16) - mask.astype(np.int16)))
        )
        if mean_difference < 5:
            errors.append(f"{name}: mask appears to be a copy of the original image")
            continue

        mask_like, mask_like_ratio = is_mask_like(mask)
        converted_from_overlay = False
        if with_overlays and not mask_like:
            if clean_dir is None:
                errors.append(f"{name}: overlay conversion requires a clean output dir")
                continue
            class_ids = overlay_to_class_ids(image, mask)
            invalid = np.zeros(class_ids.shape, dtype=bool)
            invalid_count = 0
            converted_from_overlay = True
            warnings.append(
                f"{name}: converted overlay to mask "
                f"({mask_like_ratio:.1%} mask-like pixels)"
            )
        else:
            class_ids, invalid = encode_mask(mask)
            invalid_count = int(invalid.sum())

        valid_ratio = 1.0 - invalid_count / invalid.size

        if not with_overlays and valid_ratio < 0.5:
            errors.append(
                f"{name}: only {valid_ratio:.1%} of mask pixels use valid colors; "
                "this does not look like a painted mask"
            )
            continue
        if not converted_from_overlay and valid_ratio < 0.5:
            errors.append(
                f"{name}: only {valid_ratio:.1%} of mask pixels use valid colors; "
                "this does not look like a painted mask or overlay"
            )
            continue

        if invalid_count:
            class_ids = repair_invalid_colors(class_ids, invalid)
            warnings.append(
                f"{name}: repaired {invalid_count} invalid-color pixels from "
                "surrounding fiber/resin/pore colors"
            )

        cleaned_path = clean_dir / f"{name}.png" if clean_dir else mask_path
        if clean_dir:
            write_rgb(cleaned_path, decode_mask(class_ids))
        pairs.append(Pair(name, image_path, cleaned_path))

    return ValidationResult(pairs, errors, warnings)
