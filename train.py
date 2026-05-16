"""
Train the baseline and equivariant segmentation models.

Usage:
    python train.py --model baseline
    python train.py --model equivariant
    python train.py --model both          (default)
"""

import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_loaders
from models import BaselineUNet, EquivariantUNet


CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def iou_score(logits, masks, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * masks).sum(dim=(1, 2, 3))
    union = (preds + masks).clamp(max=1).sum(dim=(1, 2, 3))
    return (inter / union.clamp(min=1e-6)).mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou = 0.0, 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item() * imgs.size(0)
        total_iou += iou_score(logits, masks) * imgs.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_iou / n


def train(model_name, epochs=20, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} on {device}")

    train_loader, val_loader, _ = get_loaders(
        root=os.path.join(os.path.dirname(__file__), "data"),
        size=128,
        batch_size=16,
    )

    if model_name == "baseline":
        model = BaselineUNet().to(device)
    else:
        model = EquivariantUNet().to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_iou = 0.0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.pt")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = evaluate(model, val_loader, criterion, device)
        print(f"  Epoch {epoch:2d}/{epochs}  loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_iou={val_iou:.4f}")
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), ckpt_path)

    print(f"  Best val IoU: {best_val_iou:.4f}  ->  saved to {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "equivariant", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    names = ["baseline", "equivariant"] if args.model == "both" else [args.model]
    for name in names:
        train(name, epochs=args.epochs, lr=args.lr)
