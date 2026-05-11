"""Loss and visualisation helpers used by train.py / infer.py."""
import math
import os

import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Adversarial losses (non-saturating logistic — StyleGAN2 default)
# ---------------------------------------------------------------------------
def d_logistic_loss(real_logits, fake_logits):
    """E[softplus(-D(real))] + E[softplus(D(fake))]."""
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def g_nonsaturating_loss(fake_logits):
    """E[softplus(-D(fake))]."""
    return F.softplus(-fake_logits).mean()


# ---------------------------------------------------------------------------
# R1 regularizer (Mescheder et al., 2018) for the discriminator
# ---------------------------------------------------------------------------
def r1_gradient_penalty(real_imgs, real_logits):
    """E[||grad_x D(x)||^2] on real samples; computed with create_graph=True."""
    grad = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_imgs,
        create_graph=True,
    )[0]
    return grad.pow(2).reshape(grad.size(0), -1).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Path-length regularizer (Karras et al., 2020) for the generator
# ---------------------------------------------------------------------------
def path_length_lengths(fake_imgs, w):
    """L2 norm of d(fake * y) / d w with y a random unit vector in pixel space."""
    noise = torch.randn_like(fake_imgs) / math.sqrt(
        fake_imgs.shape[2] * fake_imgs.shape[3]
    )
    grad = torch.autograd.grad(
        outputs=(fake_imgs * noise).sum(),
        inputs=w,
        create_graph=True,
    )[0]
    return grad.pow(2).sum(dim=1).sqrt()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def save_image_grid(tensor, save_path, nrow=4, padding=2, pad_value=0.0):
    """Save a [-1, 1] RGB image batch as a tiled PNG grid.

    Numpy-free implementation — torchvision's save_image goes through numpy
    which can fail in environments with broken numpy installs.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    assert tensor.dim() == 4, "expected NCHW tensor"
    n, c, h, w = tensor.shape
    assert c in (1, 3), "save_image_grid expects 1 or 3 channels"
    ncol = max(1, int(nrow))
    nrow_grid = (n + ncol - 1) // ncol

    # Map [-1, 1] -> [0, 1]
    img = tensor.detach().clamp(-1, 1).add(1).div(2).cpu()

    grid_h = nrow_grid * h + (nrow_grid + 1) * padding
    grid_w = ncol     * w + (ncol     + 1) * padding
    grid = torch.full((c, grid_h, grid_w), float(pad_value))
    for idx in range(n):
        r, k = divmod(idx, ncol)
        y0 = padding + r * (h + padding)
        x0 = padding + k * (w + padding)
        grid[:, y0:y0 + h, x0:x0 + w] = img[idx]

    # Tensor [C, H, W] in [0, 1] -> uint8 -> PIL -> PNG, all without numpy
    arr = grid.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
    if c == 1:
        arr = arr.expand(3, -1, -1)
    arr = arr.permute(1, 2, 0).contiguous()      # [H, W, 3]
    h_, w_, _ = arr.shape
    pil = Image.frombytes("RGB", (w_, h_), bytes(arr.flatten().tolist()))
    pil.save(save_path)
