"""
Generate the equivariance comparison figure for the paper.

Layout (3 rows x 3 columns):

              f(x)              f(gx)           |f(gx) - gf(x)|
  Input     [ image x ]      [ image gx ]       [ empty ]
  Baseline  [ mask f(x) ]    [ mask f(gx) ]     [ diff image ]
  Equiv     [ mask f_G(x) ]  [ mask f_G(gx) ]   [ diff image ]

The difference column shows pixel-wise absolute error between the model's
prediction on the rotated input f(gx) and the rotated prediction gf(x).
For an equivariant model this is identically zero; for the baseline it is not.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataset import PetSegDataset
from models import BaselineUNet, EquivariantUNet
from d4_transforms import D4

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
TEST_IDX = 2609   # Ragdoll — clean foreground, highest equivariance error among solid-mask candidates
G_IDX = 1         # g = 90° CCW rotation


def load_model(name, device):
    cls = BaselineUNet if name == "baseline" else EquivariantUNet
    m = cls()
    m.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, f"{name}.pt"), map_location=device
    ))
    return m.to(device).eval()


@torch.no_grad()
def predict_prob(model, img_tensor, device):
    """img_tensor: C×H×W -> returns H×W numpy probability map in [0,1]."""
    logits = model(img_tensor.unsqueeze(0).to(device))
    return torch.sigmoid(logits).squeeze().cpu().numpy()


def binarize(prob, threshold=0.5):
    return (prob > threshold).astype(np.uint8)


def to_numpy_img(t):
    return t.permute(1, 2, 0).cpu().numpy().clip(0, 1)


def apply_g_to_mask(mask_np, g):
    """Apply a D4 transform g to a 2-D numpy mask via torch."""
    t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()
    t = g(t)
    return t.squeeze().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PetSegDataset("data", split="test", size=128)
    img, _ = ds[TEST_IDX]        # C×H×W
    g = D4[G_IDX]                # 90° CCW rotation
    img_rot = g(img)             # C×H×W

    baseline    = load_model("baseline",    device)
    equivariant = load_model("equivariant", device)

    # Probability maps: f(x) and f(gx) for each model
    base_prob_orig  = predict_prob(baseline,    img,     device)
    base_prob_rot   = predict_prob(baseline,    img_rot, device)   # f(gx)
    equiv_prob_orig = predict_prob(equivariant, img,     device)
    equiv_prob_rot  = predict_prob(equivariant, img_rot, device)   # f_G(gx)

    # Rotated predictions: g(f(x)) and g(f_G(x))
    base_g_f_x  = apply_g_to_mask(base_prob_orig,  g)   # g(f(x))
    equiv_g_f_x = apply_g_to_mask(equiv_prob_orig, g)   # g(f_G(x))

    # Equivariance error maps: |f(gx) - g(f(x))|
    base_diff  = np.abs(base_prob_rot  - base_g_f_x)
    equiv_diff = np.abs(equiv_prob_rot - equiv_g_f_x)

    base_err  = 100.0 * base_diff.mean()
    equiv_err = 100.0 * equiv_diff.mean()

    # Binary masks for display (cols 0-1)
    base_bin_orig  = binarize(base_prob_orig)
    base_bin_rot   = binarize(base_prob_rot)
    equiv_bin_orig = binarize(equiv_prob_orig)
    equiv_bin_rot  = binarize(equiv_prob_rot)

    # --- Figure layout -------------------------------------------------
    img_np     = to_numpy_img(img)
    img_rot_np = to_numpy_img(img_rot)

    fig = plt.figure(figsize=(6.0, 5.0))

    # Manual GridSpec: 3 rows × 3 cols, third col slightly narrower (colorbar)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 3, figure=fig,
                  hspace=0.08, wspace=0.07,
                  width_ratios=[1, 1, 1])

    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]

    # Mask colormap: dark bg, near-white fg
    from matplotlib.colors import ListedColormap
    mask_cmap = ListedColormap(["#111111", "#e8f4fd"])

    # --- Row 0: input images -------------------------------------------
    axes[0][0].imshow(img_np)
    axes[0][1].imshow(img_rot_np)
    axes[0][2].axis("off")   # empty top-right corner

    # --- Row 1: baseline -----------------------------------------------
    axes[1][0].imshow(base_bin_orig, cmap=mask_cmap, vmin=0, vmax=1)
    axes[1][1].imshow(base_bin_rot,  cmap=mask_cmap, vmin=0, vmax=1)
    im_base = axes[1][2].imshow(base_diff, cmap="Reds", vmin=0, vmax=0.5)

    # --- Row 2: equivariant --------------------------------------------
    axes[2][0].imshow(equiv_bin_orig, cmap=mask_cmap, vmin=0, vmax=1)
    axes[2][1].imshow(equiv_bin_rot,  cmap=mask_cmap, vmin=0, vmax=1)
    im_equiv = axes[2][2].imshow(equiv_diff, cmap="Reds", vmin=0, vmax=0.5)

    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Column headers
    axes[0][0].set_title(r"$x$", fontsize=10, pad=3)
    axes[0][1].set_title(r"$gx$", fontsize=10, pad=3)
    axes[0][2].set_title(r"$|f(gx) - gf(x)|$", fontsize=8.5, pad=3)

    # Row labels
    row_labels = ["Input", "Baseline", "Equivariant"]
    for r, label in enumerate(row_labels):
        axes[r][0].set_ylabel(label, fontsize=8.5, labelpad=5,
                               rotation=90, va="center")

    # Equivariance error text in corner of difference panels
    axes[1][2].text(0.97, 0.05, f"{base_err:.1f}% error",
                    transform=axes[1][2].transAxes, fontsize=7,
                    color="#c0392b", ha="right", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.8, pad=1.5, edgecolor="none"))
    axes[2][2].text(0.97, 0.05, f"{equiv_err:.2f}% error",
                    transform=axes[2][2].transAxes, fontsize=7,
                    color="#27ae60", ha="right", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.8, pad=1.5, edgecolor="none"))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_pdf = os.path.join(OUT_DIR, "equivariance_figure.pdf")
    out_png = os.path.join(OUT_DIR, "equivariance_figure.png")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    print(f"Saved {out_pdf}")
    print(f"Baseline equivariance error:    {base_err:.2f}%")
    print(f"Equivariant equivariance error: {equiv_err:.4f}%")


if __name__ == "__main__":
    main()
