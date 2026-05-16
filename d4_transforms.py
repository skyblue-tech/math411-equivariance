"""
The 8 elements of D4 acting on square image tensors (B×C×H×W or C×H×W).
Each element is a callable that permutes pixels via rotation/reflection.
"""

import torch


def _rot90(x, k):
    return torch.rot90(x, k, dims=[-2, -1])


def _flip_h(x):
    return torch.flip(x, dims=[-1])


def _flip_v(x):
    return torch.flip(x, dims=[-2])


def _transpose(x):
    return x.transpose(-2, -1)


# D4 = {e, r, r^2, r^3, s, sr, sr^2, sr^3}
# where r = 90-degree CCW rotation, s = horizontal flip
D4 = [
    lambda x: x,                                     # e  (identity)
    lambda x: _rot90(x, 1),                          # r  (90° CCW)
    lambda x: _rot90(x, 2),                          # r^2 (180°)
    lambda x: _rot90(x, 3),                          # r^3 (270° CCW)
    lambda x: _flip_h(x),                            # s  (horizontal flip)
    lambda x: _flip_h(_rot90(x, 1)),                 # sr
    lambda x: _flip_h(_rot90(x, 2)),                 # sr^2
    lambda x: _flip_h(_rot90(x, 3)),                 # sr^3
]

D4_NAMES = ["e", "r", "r²", "r³", "s", "sr", "sr²", "sr³"]

# Inverse of each D4 element, indexed to match D4 above.
# All reflections and r^2 are self-inverse; r and r^3 swap.
D4_INV = [
    D4[0],   # e^{-1}   = e
    D4[3],   # r^{-1}   = r^3
    D4[2],   # (r^2)^{-1} = r^2
    D4[1],   # (r^3)^{-1} = r
    D4[4],   # s^{-1}   = s
    D4[5],   # (sr)^{-1} = sr
    D4[6],   # (sr^2)^{-1} = sr^2
    D4[7],   # (sr^3)^{-1} = sr^3
]
