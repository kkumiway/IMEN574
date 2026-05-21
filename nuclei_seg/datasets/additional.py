"""
Dataset for additional_data format:
    additional_data/images_all/{id}.png   <- RGBA image, variable size
    additional_data/labels_all/{id}.tif   <- uint16 instance label map (0=bg, 1..N=nuclei)
    additional_data/masks_all/{id}.png    <- binary mask (not used)

Converts the pre-made instance label map to the same 2-channel body/border
float32 mask used by DSBDataset, so both datasets are interchangeable in training.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
import tifffile
from skimage.morphology import dilation, footprint_rectangle
from skimage.segmentation import watershed
from torch.utils.data import Dataset

import albumentations as A
from nuclei_seg.aug.transforms import IMAGENET_MEAN, IMAGENET_STD, get_val_transforms
from albumentations.pytorch import ToTensorV2


def get_train_transforms_padded(crop_size: int = 256) -> A.Compose:
    """Train transforms with PadIfNeeded before RandomCrop for small images."""
    return A.Compose([
        A.PadIfNeeded(min_height=crop_size, min_width=crop_size,
                      border_mode=cv2.BORDER_REFLECT, p=1.0),
        A.RandomCrop(crop_size, crop_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def labeled_to_body_border(labeled: np.ndarray) -> np.ndarray:
    """
    Convert a labeled instance map (H, W) uint to (H, W, 2) float32 body/border mask.
    Same logic as create_body_border_mask in dsb.py.
    """
    body_bin = labeled > 0

    dilated = dilation(body_bin, footprint_rectangle((9, 9)))
    ws      = watershed(dilated, labeled.astype(np.int32), mask=dilated, watershed_line=True)
    border  = (dilated ^ (ws > 0))
    border  = dilation(border, footprint_rectangle((3, 3)))

    body = body_bin.copy().astype(np.uint8)
    body[border] = 0

    return np.stack([body.astype(np.float32), border.astype(np.float32)], axis=-1)


class AdditionalDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        image_ids: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = Path(data_dir) / "images_all"
        self.labels_dir = Path(data_dir) / "labels_all"

        if image_ids is not None:
            self.image_ids = image_ids
        else:
            self.image_ids = [p.stem for p in sorted(self.images_dir.glob("*.png"))]

        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]

        # Load image — RGBA → RGB
        img = cv2.imread(str(self.images_dir / f"{img_id}.png"), cv2.IMREAD_UNCHANGED)
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load labeled instance map and build body/border mask
        labeled = tifffile.imread(str(self.labels_dir / f"{img_id}.tif"))
        mask = labeled_to_body_border(labeled)   # (H, W, 2) float32

        if self.transform:
            augmented   = self.transform(image=img, mask=mask)
            img_tensor  = augmented["image"]
            mask_tensor = augmented["mask"].permute(2, 0, 1).float()
        else:
            import torch
            img_tensor  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask.transpose(2, 0, 1)).float()

        return img_tensor, mask_tensor, img_id


def make_additional_datasets(
    data_dir: str,
    val_ratio: float = 0.15,
    crop_size: int = 256,
    seed: int = 42,
):
    """Split additional_data into train/val and return both datasets."""
    all_ids = [p.stem for p in sorted((Path(data_dir) / "images_all").glob("*.png"))]

    rng = random.Random(seed)
    shuffled = all_ids[:]
    rng.shuffle(shuffled)

    n_val    = int(len(shuffled) * val_ratio)
    val_ids  = shuffled[:n_val]
    train_ids = shuffled[n_val:]

    print(f"Additional data: {len(train_ids)} train / {len(val_ids)} val")

    train_ds = AdditionalDataset(data_dir, train_ids, transform=get_train_transforms_padded(crop_size))
    val_ds   = AdditionalDataset(data_dir, val_ids,   transform=get_val_transforms())
    return train_ds, val_ds