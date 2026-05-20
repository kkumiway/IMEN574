"""
Shared utilities:
  - RLE encode / decode  (Kaggle DSB 2018 format)
  - Watershed post-processing  (binary mask → instance labels)
"""
from __future__ import annotations

import numpy as np
from skimage import measure
from skimage.morphology import dilation, footprint_rectangle
from skimage.segmentation import watershed
from scipy import ndimage as ndi

import os
import random
import torch
from torch.utils.data import get_worker_info


# ---------------------------------------------------------------------------
# RLE
# ---------------------------------------------------------------------------

def rle_encode(mask: np.ndarray) -> str:
    """
    Encode binary mask (H, W) to run-length string.
    Pixels are indexed column-major (Kaggle convention).
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs   = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(rle_string: str, shape: tuple[int, int]) -> np.ndarray:
    """
    Decode run-length string to binary mask of given (H, W).
    """
    h, w   = shape
    mask   = np.zeros(h * w, dtype=np.uint8)
    if not isinstance(rle_string, str) or rle_string.strip() == "":
        return mask.reshape(h, w)

    nums = list(map(int, rle_string.split()))
    starts, lengths = nums[::2], nums[1::2]
    for s, l in zip(starts, lengths):
        mask[s - 1 : s - 1 + l] = 1          # 1-indexed
    return mask.reshape(w, h).T               # column-major → row-major


# ---------------------------------------------------------------------------
# Post-processing: binary → instance label map
# ---------------------------------------------------------------------------

def postprocess_to_instance_map(
    body_prob: np.ndarray,
    body_border: np.ndarray | None = None,
    body_thresh: float = 0.5,
    border_thresh: float = 0.3,
    min_nucleus_size: int = 10,
) -> np.ndarray:
    """
    Convert predicted body (and optional border) probability maps to an
    instance label map where each nucleus gets a unique positive integer.

    Strategy (mirrors topcoders post-processing):
      1. Threshold body mask
      2. Subtract predicted border to separate touching nuclei
      3. Distance-transform to find seed points
      4. Watershed from seeds into body mask
      5. Remove tiny objects
    """
    body_bin = (body_prob >= body_thresh).astype(np.uint8)

    if body_border is not None:
        border_bin = (body_border >= border_thresh).astype(np.uint8)
        seeds_bin  = np.clip(body_bin - border_bin, 0, 1)
    else:
        seeds_bin = body_bin.copy()

    # Distance transform → seeds
    distance = ndi.distance_transform_edt(seeds_bin)
    seeds_labeled = measure.label(distance > 0.3 * distance.max())

    # Watershed into the full body mask
    ws = watershed(-distance, seeds_labeled, mask=body_bin)

    # Remove very small objects
    for region in measure.regionprops(ws):
        if region.area < min_nucleus_size:
            ws[ws == region.label] = 0

    return ws.astype(np.int32)

# ---------------------------------------------------------------------------
# Set Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """
    Fix the random seed of all Library (Python, numpy, torch .. etc.),
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    """
    Fix the random.seed in each worker in DataLoader
    including np.random / random 
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # Albumentations v2 worker별 RNG 분리
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, 'transform') and info.dataset.transform is not None:
        info.dataset.transform.set_random_seed(worker_seed)
