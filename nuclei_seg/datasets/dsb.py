"""
DSB 2018 dataset.

Each training sample lives at:
    stage1_train/{id}/images/{id}.png    ← RGB image
    stage1_train/{id}/masks/*.png        ← one PNG per nucleus (binary)

We build a 2-channel mask on the fly:
    channel 0 = nucleus body  (union of all masks, borders removed)
    channel 1 = nucleus border (thin boundary between touching nuclei)

This mirrors the body/border strategy from the selim topcoders branch.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from nuclei_seg.aug.transforms import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Mask creation
# ---------------------------------------------------------------------------

def _pad32(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Pad image so H and W are multiples of 32."""
    h, w = img.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    t = pad_h // 2
    b = pad_h - t
    l = pad_w // 2
    r = pad_w - l
    if img.ndim == 3:
        padded = np.pad(img, ((t, b), (l, r), (0, 0)), mode="reflect")
    else:
        padded = np.pad(img, ((t, b), (l, r)), mode="reflect")
    return padded, (t, b, l, r)


def create_body_border_mask(mask_paths: List[Path]) -> np.ndarray:
    """
    Return float32 array of shape (H, W, 2).
      ch0 = body  (nucleus pixels, boundary pixels set to 0)
      ch1 = border (boundary pixels between touching nuclei)
    """
    from skimage import measure
    from skimage.morphology import dilation, footprint_rectangle
    from skimage.segmentation import watershed

    if not mask_paths:
        raise ValueError("No mask files found")

    # Build labeled image (each nucleus = unique integer)
    sample = cv2.imread(str(mask_paths[0]), cv2.IMREAD_GRAYSCALE)
    h, w = sample.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    for i, mp in enumerate(mask_paths):
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        labeled[m > 127] = i + 1

    body_bin = labeled > 0

    # Dilate body slightly, then watershed to find shared boundaries
    dilated = dilation(body_bin, footprint_rectangle((9, 9)))
    ws = watershed(dilated, labeled, mask=dilated, watershed_line=True)
    border = (dilated ^ (ws > 0))
    border = dilation(border, footprint_rectangle((3, 3)))

    body = body_bin.copy().astype(np.uint8)
    body[border] = 0

    mask = np.stack([body.astype(np.float32), border.astype(np.float32)], axis=-1)
    return mask  # (H, W, 2)  values in {0, 1}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DSBDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        image_ids: List[str],
        transform: Optional[Callable] = None,
        is_test: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.image_ids = image_ids
        self.transform = transform
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]

        # Load image as RGB
        if self.is_test:
            img_path = self.data_dir / "stage1_test" / img_id / "images" / f"{img_id}.png"
        else:
            img_path = self.data_dir / "stage1_train" / img_id / "images" / f"{img_id}.png"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_test:
            img_padded, pad = _pad32(img)
            if self.transform:
                augmented = self.transform(image=img_padded)
                img_tensor = augmented["image"]
            else:
                img_tensor = img_padded
            return img_tensor, img_id, pad

        # Load masks and build body/border mask
        mask_dir = self.data_dir / "stage1_train" / img_id / "masks"
        mask_paths = sorted(mask_dir.glob("*.png"))
        mask = create_body_border_mask(mask_paths)  # (H, W, 2) float32

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img_tensor = augmented["image"]        # (3, H, W) float32 tensor
            mask_tensor = augmented["mask"]        # (H, W, 2) → permute below
            # albumentations ToTensorV2 does NOT convert mask channels, only HWC→CHW for image
            # mask stays as (H, W, C) tensor — move channels first
            mask_tensor = mask_tensor.permute(2, 0, 1).float()  # (2, H, W)
        else:
            import torch
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask.transpose(2, 0, 1)).float()

        return img_tensor, mask_tensor, img_id


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def get_train_val_ids(
    data_dir: str,
    fold: int = 0,
    n_folds: int = 4,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    train_root = Path(data_dir) / "stage1_train"
    all_ids = sorted(os.listdir(train_root))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(all_ids))
    train_idx, val_idx = splits[fold]

    train_ids = [all_ids[i] for i in train_idx]
    val_ids   = [all_ids[i] for i in val_idx]
    return train_ids, val_ids


def get_test_ids(data_dir: str) -> List[str]:
    test_root = Path(data_dir) / "stage1_test"
    return sorted(os.listdir(test_root))


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_datasets(
    data_dir: str,
    fold: int = 0,
    n_folds: int = 4,
    crop_size: int = 256,
    seed: int = 42,
) -> Tuple[DSBDataset, DSBDataset]:
    train_ids, val_ids = get_train_val_ids(data_dir, fold, n_folds, seed)
    print(f"Fold {fold}: {len(train_ids)} train / {len(val_ids)} val")

    train_ds = DSBDataset(data_dir, train_ids, transform=get_train_transforms(crop_size))
    val_ds   = DSBDataset(data_dir, val_ids,   transform=get_val_transforms())
    return train_ds, val_ds