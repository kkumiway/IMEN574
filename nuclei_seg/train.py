"""
Train nuclei segmentation model.

Usage (from haein/ directory):
    python -m nuclei_seg.train \\
        --data_dir  data-science-bowl-2018 \\
        --fold      0 \\
        --epochs    30 \\
        --batch_size 8 \\
        --crop_size  256 \\
        --encoder   resnet34 \\
        --weights_dir nuclei_seg/weights
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from nuclei_seg.datasets.dsb import make_datasets
from nuclei_seg.losses import make_loss
from nuclei_seg.metric import mean_ap, instance_map_score
from nuclei_seg.models.unet import make_model
from nuclei_seg.utils import postprocess_to_instance_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data-science-bowl-2018")
    p.add_argument("--fold",        type=int,   default=0)
    p.add_argument("--n_folds",     type=int,   default=4)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--crop_size",   type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--encoder",     default="resnet34")
    p.add_argument("--loss",        default="double_head")
    p.add_argument("--weights_dir", default="nuclei_seg/weights")
    p.add_argument("--num_workers", type=int,   default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total_loss = 0.0
    for imgs, masks, _ in tqdm(loader, desc="  train", leave=False):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss  = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, loss_fn, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    ap_scores  = []

    for imgs, masks, _ in tqdm(loader, desc="  val  ", leave=False):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        # Pad to 32-multiple, run model, then crop back
        h, w = imgs.shape[2], imgs.shape[3]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            imgs  = F.pad(imgs,  (0, pad_w, 0, pad_h))
            masks = F.pad(masks, (0, pad_w, 0, pad_h))
        preds = model(imgs)[:, :, :h, :w]
        masks = masks[:, :, :h, :w]
        loss  = loss_fn(preds, masks)
        total_loss += loss.item()

        # Compute Kaggle mAP on body channel (ch 0)
        for i in range(imgs.shape[0]):
            body_pred = preds[i, 0].cpu().numpy()
            body_gt   = masks[i, 0].cpu().numpy()
            pred_inst = postprocess_to_instance_map(body_pred, body_border=preds[i, 1].cpu().numpy())
            gt_inst   = postprocess_to_instance_map(body_gt)
            ap_scores.append(instance_map_score(gt_inst, pred_inst))

    return total_loss / len(loader), float(sum(ap_scores) / len(ap_scores)) if ap_scores else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.weights_dir, exist_ok=True)

    train_ds, val_ds = make_datasets(
        args.data_dir, fold=args.fold, n_folds=args.n_folds, crop_size=args.crop_size
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=1,              shuffle=False,
                              num_workers=args.num_workers)

    model    = make_model(encoder=args.encoder).to(device)
    loss_fn  = make_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_weights  = Path(args.weights_dir) / f"best_fold{args.fold}.pth"

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val mAP':>8}")
    print("-" * 42)

    for epoch in range(1, args.epochs + 1):
        tr_loss           = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_map = val_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weights)

        marker = " *" if improved else ""
        print(f"{epoch:>5}  {tr_loss:>10.4f}  {val_loss:>9.4f}  {val_map:>8.4f}{marker}")

    print(f"\nBest val loss: {best_val_loss:.4f}  → saved to {best_weights}")


if __name__ == "__main__":
    main()