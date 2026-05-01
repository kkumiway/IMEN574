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