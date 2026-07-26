import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common.labels import encode_mask, read_rgb
from common.validation import Pair


IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def normalize_image(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def pad_to_patch(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    bottom = max(0, patch_size - height)
    right = max(0, patch_size - width)
    if bottom or right:
        image = cv2.copyMakeBorder(
            image, 0, bottom, 0, right, cv2.BORDER_REFLECT_101
        )
        mask = cv2.copyMakeBorder(
            mask, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=3
        )
    return image, mask


class SegmentationDataset(Dataset):
    def __init__(
        self,
        pairs: list[Pair],
        patch_size: int = 512,
        patches_per_image: int = 16,
        augment: bool = True,
    ):
        self.pairs = pairs
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index // self.patches_per_image]
        image = read_rgb(pair.image_path)
        mask_rgb = read_rgb(pair.mask_path)
        mask, _ = encode_mask(mask_rgb)
        image, mask = pad_to_patch(image, mask, self.patch_size)

        height, width = image.shape[:2]
        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)
        image = image[top : top + self.patch_size, left : left + self.patch_size]
        mask = mask[top : top + self.patch_size, left : left + self.patch_size]

        if self.augment:
            turns = random.randrange(4)
            image = np.rot90(image, turns)
            mask = np.rot90(mask, turns)
            if random.random() < 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)
            if random.random() < 0.5:
                image = np.flipud(image)
                mask = np.flipud(mask)
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            image = np.clip(
                (image.astype(np.float32) - 127.5) * contrast
                + 127.5 * brightness,
                0,
                255,
            ).astype(np.uint8)

        return normalize_image(np.ascontiguousarray(image)), torch.from_numpy(
            np.ascontiguousarray(mask)
        ).long()
