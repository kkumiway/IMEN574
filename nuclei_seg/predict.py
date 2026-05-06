"""
Run inference on stage1_test and evaluate against stage1_solution.csv.

Usage (from haein/ directory):
    python -m nuclei_seg.predict \\
        --data_dir      data-science-bowl-2018 \\
        --weights       nuclei_seg/weights/best_fold0.pth \\
        --encoder       resnet34 \\
        --solution_csv  data-science-bowl-2018/stage1_solution.csv/stage1_solution.csv \\
        --out_csv       nuclei_seg/predictions/submission.csv
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage import measure
from tqdm import tqdm

from nuclei_seg.aug.transforms import get_val_transforms
from nuclei_seg.datasets.dsb import _pad32
from nuclei_seg.metric import instance_map_score, mean_ap
from nuclei_seg.models.unet import load_model
from nuclei_seg.utils import postprocess_to_instance_map, rle_decode, rle_encode


# ---------------------------------------------------------------------------
# TTA helpers (flip × rot like selim's pred_test.py)
# ---------------------------------------------------------------------------

def _tta_predict(model, img_tensor: torch.Tensor, device: str) -> np.ndarray:
    """
    8-fold TTA: 2 flips × 4 rotations.
    Returns averaged (H, W, 2) numpy array in [0, 1].
    """
    img = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
    acc = None
    count = 0

    for flip in range(2):
        x = img.flip(-2) if flip else img       # vertical flip
        for rot in range(4):
            inp = torch.rot90(x, rot, dims=(-2, -1))
            with torch.no_grad():
                pred = model(inp)               # (1, 2, H, W)
            # Undo rotation + flip
            pred = torch.rot90(pred, -rot, dims=(-2, -1))
            if flip:
                pred = pred.flip(-2)
            pred_np = pred[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
            acc = pred_np if acc is None else acc + pred_np
            count += 1

    return acc / count


# ---------------------------------------------------------------------------
# Solution CSV parsing → GT instance maps
# ---------------------------------------------------------------------------

def load_gt_instance_maps(
    solution_csv: str,
) -> dict[str, np.ndarray]:
    """
    Parse stage1_solution.csv and return a dict:
        {image_id: labeled_instance_map (H, W int32)}

    Each row in the CSV is one nucleus (one RLE mask).
    """
    df = pd.read_csv(solution_csv)
    gt_maps: dict[str, np.ndarray] = {}

    for img_id, group in df.groupby("ImageId"):
        h = int(group["Height"].iloc[0])
        w = int(group["Width"].iloc[0])
        labeled = np.zeros((h, w), dtype=np.int32)
        nucleus_idx = 1
        for _, row in group.iterrows():
            if not isinstance(row["EncodedPixels"], str):
                continue
            single_mask = rle_decode(row["EncodedPixels"], (h, w))
            labeled[single_mask > 0] = nucleus_idx
            nucleus_idx += 1
        gt_maps[str(img_id)] = labeled

    return gt_maps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     default="data-science-bowl-2018")
    p.add_argument("--weights",      default="nuclei_seg/weights/best_fold0.pth")
    p.add_argument("--encoder",      default="resnet34")
    p.add_argument("--solution_csv", default="data-science-bowl-2018/stage1_solution.csv/stage1_solution.csv")
    p.add_argument("--out_csv",      default="nuclei_seg/predictions/submission.csv")
    p.add_argument("--no_tta",       action="store_true", help="Disable test-time augmentation")
    return p.parse_args()


def _predict_one(img_path: Path, model, transform, device: str, no_tta: bool):
    """Load image, run inference, return instance label map."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_padded, (t, b, l, r) = _pad32(img)
    img_tensor = transform(image=img_padded)["image"]

    if no_tta:
        with torch.no_grad():
            pred = model(img_tensor.unsqueeze(0).to(device))
        pred_np = pred[0].permute(1, 2, 0).cpu().numpy()
    else:
        pred_np = _tta_predict(model, img_tensor, device)

    ph, pw = img_padded.shape[:2]
    pred_np = pred_np[t: ph - b if b else ph, l: pw - r if r else pw, :]

    return postprocess_to_instance_map(pred_np[:, :, 0], pred_np[:, :, 1])


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(args.weights, encoder=args.encoder, device=device)
    transform = get_val_transforms()

    data_root = Path(args.data_dir)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    # Load stage1 GT for local evaluation
    gt_maps = load_gt_instance_maps(args.solution_csv)
    print(f"GT instances loaded for {len(gt_maps)} images")

    # Collect all test images: stage1_test + stage2_test_final
    test_subdirs = [
        ("stage1_test",       "Stage 1"),
        ("stage2_test_final", "Stage 2"),
    ]
    all_test_ids: list[tuple[str, str]] = []   # (subdir, img_id)
    for subdir, label in test_subdirs:
        d = data_root / subdir
        if d.exists():
            ids = sorted(p.name for p in d.iterdir() if p.is_dir())
            print(f"{label} ({subdir}): {len(ids)} images")
            all_test_ids.extend((subdir, img_id) for img_id in ids)

    print(f"Total test images: {len(all_test_ids)}")

    submission_rows = []
    per_image_scores = []

    for subdir, img_id in tqdm(all_test_ids, desc="Predicting"):
        img_path = data_root / subdir / img_id / "images" / f"{img_id}.png"
        pred_inst = _predict_one(img_path, model, transform, device, args.no_tta)

        # RLE-encode each predicted nucleus
        # If no nuclei detected, add a dummy 1-pixel entry so the image ID
        # appears in the submission (Kaggle requires all solution IDs present)
        nucleus_ids = np.unique(pred_inst)
        nucleus_ids = nucleus_ids[nucleus_ids > 0]
        if len(nucleus_ids) == 0:
            submission_rows.append({"ImageId": img_id, "EncodedPixels": "1 1"})
        else:
            for nid in nucleus_ids:
                mask = (pred_inst == nid).astype(np.uint8)
                submission_rows.append({
                    "ImageId": img_id,
                    "EncodedPixels": rle_encode(mask),
                })

        # Evaluate against GT if available
        if img_id in gt_maps:
            gt_inst = gt_maps[img_id]
            # Resize pred_inst to GT size if needed
            if pred_inst.shape != gt_inst.shape:
                pred_inst_resized = cv2.resize(
                    pred_inst.astype(np.float32),
                    (gt_inst.shape[1], gt_inst.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)
            else:
                pred_inst_resized = pred_inst
            score = instance_map_score(gt_inst, pred_inst_resized)
            per_image_scores.append({"ImageId": img_id, "AP": score})

    # Save submission CSV
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(args.out_csv, index=False)
    print(f"\nSubmission saved → {args.out_csv}")
    print(f"  Total rows (nuclei): {len(submission_df)}")
    print(f"  Images predicted   : {submission_df['ImageId'].nunique()}")

    # Print evaluation results
    if per_image_scores:
        scores_df = pd.DataFrame(per_image_scores).sort_values("AP")
        mean_score = scores_df["AP"].mean()

        print("\n" + "=" * 52)
        print(f"  Evaluation vs stage1_solution.csv")
        print("=" * 52)
        print(f"  Mean AP (IoU 0.5:0.05:0.95) : {mean_score:.4f}")
        print(f"  Min AP                       : {scores_df['AP'].min():.4f}")
        print(f"  Max AP                       : {scores_df['AP'].max():.4f}")
        print(f"  Median AP                    : {scores_df['AP'].median():.4f}")
        print("=" * 52)

        print("\n  Per-image results (sorted by AP, worst first):")
        print(f"  {'ImageId[:16]':<18} {'AP':>6}")
        print("  " + "-" * 26)
        for _, row in scores_df.iterrows():
            flag = " ← worst" if row["AP"] == scores_df["AP"].min() else ""
            print(f"  {str(row['ImageId'])[:16]:<18} {row['AP']:>6.4f}{flag}")

        # Save per-image scores
        score_path = Path(args.out_csv).parent / "per_image_scores.csv"
        scores_df.to_csv(score_path, index=False)
        print(f"\n  Per-image scores saved → {score_path}")
    else:
        print("\nNo test images found in solution CSV — cannot evaluate.")


if __name__ == "__main__":
    main()