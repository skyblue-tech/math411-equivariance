"""
Evaluate both models and print the results table used in the paper.

Usage:
    python evaluate.py

Outputs:
  - Equivariance error (%) for baseline and equivariant model
  - Mean IoU on original test set
  - Mean IoU on D4-transformed test set
"""

import os
import torch
import torch.nn.functional as F_nn
from tqdm import tqdm

from dataset import get_loaders
from d4_transforms import D4, D4_NAMES
from models import BaselineUNet, EquivariantUNet

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def load_model(name, device):
    if name == "baseline":
        model = BaselineUNet()
    else:
        model = EquivariantUNet()
    ckpt = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    return model


@torch.no_grad()
def equivariance_error(model, loader, device):
    """
    For each test image x and each g in D4, compute the pixel-wise L1
    distance between sigmoid(f(g(x))) and g(sigmoid(f(x))).
    Returns the mean over all pixels, images, and group elements (as %).
    """
    total_err = 0.0
    total_pixels = 0
    for imgs, _ in tqdm(loader, desc="  equivariance error", leave=False):
        imgs = imgs.to(device)
        f_x = torch.sigmoid(model(imgs))   # B×1×H×W

        for g in D4:
            g_x = g(imgs)
            f_gx = torch.sigmoid(model(g_x))
            g_fx = g(f_x)
            err = (f_gx - g_fx).abs().mean().item()
            total_err += err * imgs.size(0)
            total_pixels += imgs.size(0)

    return 100.0 * total_err / total_pixels


@torch.no_grad()
def mean_iou(model, loader, device, transform=None):
    """
    Compute mean IoU over the loader.  If transform is given, apply it to
    each image and mask before evaluating (for the transformed test set).
    """
    total_iou = 0.0
    n = 0
    for imgs, masks in tqdm(loader, desc="  IoU", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        if transform is not None:
            imgs = transform(imgs)
            masks = transform(masks)
        logits = model(imgs)
        preds = (torch.sigmoid(logits) > 0.5).float()
        inter = (preds * masks).sum(dim=(1, 2, 3))
        union = (preds + masks).clamp(max=1).sum(dim=(1, 2, 3))
        iou = (inter / union.clamp(min=1e-6)).sum().item()
        total_iou += iou
        n += imgs.size(0)
    return total_iou / n


@torch.no_grad()
def mean_iou_all_transforms(model, loader, device):
    """Average IoU over all 8 D4 transforms of the test set."""
    total = 0.0
    for g in D4:
        total += mean_iou(model, loader, device, transform=g)
    return total / len(D4)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}\n")

    _, _, test_loader = get_loaders(
        root=os.path.join(os.path.dirname(__file__), "data"),
        size=128,
        batch_size=16,
    )

    results = {}
    for name in ["baseline", "equivariant"]:
        print(f"--- {name} ---")
        model = load_model(name, device)
        eq_err = equivariance_error(model, test_loader, device)
        iou_orig = mean_iou(model, test_loader, device)
        iou_trans = mean_iou_all_transforms(model, test_loader, device)
        results[name] = {"eq_err": eq_err, "iou_orig": iou_orig, "iou_trans": iou_trans}
        print(f"  Equivariance error : {eq_err:.2f}%")
        print(f"  IoU (original)     : {iou_orig:.4f}")
        print(f"  IoU (transformed)  : {iou_trans:.4f}")
        print()

    # Print LaTeX-ready table row for pasting into the paper
    print("LaTeX table snippet:")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Metric & Baseline & Equivariant \\")
    print(r"\midrule")
    b, e = results["baseline"], results["equivariant"]
    print(f"Equivariance error (\\%) & {b['eq_err']:.1f} & {e['eq_err']:.1f} \\\\")
    print(f"Mean IoU (original test) & {b['iou_orig']:.3f} & {e['iou_orig']:.3f} \\\\")
    print(f"Mean IoU (transformed test) & {b['iou_trans']:.3f} & {e['iou_trans']:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
