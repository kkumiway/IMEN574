"""
Kaggle DSB 2018 evaluation metric.

mean Average Precision at IoU thresholds 0.50 : 0.05 : 0.95
(same formula as selim's metric.py / Kaggle leaderboard)
"""
from __future__ import annotations

import numpy as np
from skimage import measure


THRESHOLDS = np.arange(0.5, 1.0, 0.05)


def precision_at(iou_matrix: np.ndarray, threshold: float) -> float:
    matches   = iou_matrix > threshold
    tp = np.sum(np.sum(matches, axis=1) == 1)
    fp = np.sum(np.sum(matches, axis=0) == 0)
    fn = np.sum(np.sum(matches, axis=1) == 0)
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def instance_map_score(gt_labeled: np.ndarray, pred_labeled: np.ndarray) -> float:
    """
    Compute mean AP for one image.

    Parameters
    ----------
    gt_labeled   : 2-D int array, each nucleus = unique positive int (0 = background)
    pred_labeled : same format, output of skimage.measure.label on predicted binary mask
    """
    true_objects = len(np.unique(gt_labeled))    # includes background (0)
    pred_objects = len(np.unique(pred_labeled))

    if true_objects <= 1:   # only background in GT
        return 1.0 if pred_objects <= 1 else 0.0

    intersection = np.histogram2d(
        gt_labeled.flatten(), pred_labeled.flatten(),
        bins=(true_objects, pred_objects),
    )[0]

    area_true = np.histogram(gt_labeled,   bins=true_objects)[0]
    area_pred = np.histogram(pred_labeled, bins=pred_objects)[0]

    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred,  0)

    union = area_true + area_pred - intersection
    union[union == 0] = 1e-9

    iou = intersection / union
    iou = iou[1:, 1:]   # remove background row/col

    prec = [precision_at(iou, t) for t in THRESHOLDS]
    return float(np.mean(prec))


def mean_ap(gt_list: list[np.ndarray], pred_list: list[np.ndarray]) -> float:
    """Mean AP across a list of images."""
    scores = [instance_map_score(g, p) for g, p in zip(gt_list, pred_list)]
    return float(np.mean(scores))