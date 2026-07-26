from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ("fiber", "resin", "pore", "unidentified")
CLASS_COLORS_RGB = np.array(
    [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (0, 0, 0),
    ],
    dtype=np.uint8,
)
UNIDENTIFIED_ID = 3


def read_rgb(path: Path | str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path | str, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def encode_mask(mask_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matches = np.all(
        mask_rgb[:, :, None, :] == CLASS_COLORS_RGB[None, None, :, :],
        axis=3,
    )
    valid = np.any(matches, axis=2)
    class_ids = np.argmax(matches, axis=2).astype(np.uint8)
    class_ids[~valid] = UNIDENTIFIED_ID
    return class_ids, ~valid


def repair_invalid_colors(
    class_ids: np.ndarray,
    invalid: np.ndarray,
) -> np.ndarray:
    repaired = class_ids.copy()
    unresolved = invalid.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0

    while unresolved.any():
        neighbor_counts = np.stack(
            [
                cv2.filter2D(
                    (repaired == class_id).astype(np.uint8),
                    cv2.CV_16U,
                    kernel,
                    borderType=cv2.BORDER_CONSTANT,
                )
                for class_id in range(UNIDENTIFIED_ID)
            ],
            axis=2,
        )
        most_common = np.argmax(neighbor_counts, axis=2).astype(np.uint8)
        has_colored_neighbor = np.max(neighbor_counts, axis=2) > 0
        fill = unresolved & has_colored_neighbor
        if not fill.any():
            break
        repaired[fill] = most_common[fill]
        unresolved[fill] = False

    return repaired


def decode_mask(class_ids: np.ndarray) -> np.ndarray:
    return CLASS_COLORS_RGB[class_ids]
