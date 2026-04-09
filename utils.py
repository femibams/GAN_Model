import math
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# Bounding box & ROI utilities
# ------------------------------------------------------------
def bboxes_to_mask(bboxes, image_size=128, device=None):
    if device is None:
        device = bboxes.device

    B = bboxes.shape[0]
    # Vectorized: build coordinate grids and compare against bbox coords
    # No Python loops, no .item() calls — stays entirely on GPU
    coords = torch.linspace(0, 1, image_size, device=device)
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")  # [H, W] each
    # bboxes: [B, 4] with (x1, y1, x2, y2) normalized
    x1 = bboxes[:, 0].view(B, 1, 1)
    y1 = bboxes[:, 1].view(B, 1, 1)
    x2 = bboxes[:, 2].view(B, 1, 1)
    y2 = bboxes[:, 3].view(B, 1, 1)
    mask = ((gx >= x1) & (gx < x2) & (gy >= y1) & (gy < y2)).float().unsqueeze(1)
    return mask

def crop_roi_from_bboxes(images, bboxes, out_size=64):
    """Crop per-sample RoIs and resize to out_size using grid_sample (fully vectorized)."""
    B, C, H, W = images.shape
    # Convert normalized (x1,y1,x2,y2) → grid_sample sampling grid in [-1, 1]
    x1 = bboxes[:, 0].clamp(0, 1)
    y1 = bboxes[:, 1].clamp(0, 1)
    x2 = bboxes[:, 2].clamp(0, 1)
    y2 = bboxes[:, 3].clamp(0, 1)

    # Build a sampling grid for each sample: shape [B, out_size, out_size, 2]
    ty = torch.linspace(0, 1, out_size, device=images.device, dtype=images.dtype)
    tx = torch.linspace(0, 1, out_size, device=images.device, dtype=images.dtype)
    gy, gx = torch.meshgrid(ty, tx, indexing="ij")       # [out_size, out_size]

    # Map [0,1] grid coords to each bbox's [x1,x2] / [y1,y2] range
    sample_x = x1.view(B, 1, 1) + gx.unsqueeze(0) * (x2 - x1).view(B, 1, 1)
    sample_y = y1.view(B, 1, 1) + gy.unsqueeze(0) * (y2 - y1).view(B, 1, 1)

    # grid_sample expects coordinates in [-1, 1]
    grid = torch.stack([sample_x * 2 - 1, sample_y * 2 - 1], dim=-1)  # [B, S, S, 2]
    return F.grid_sample(images, grid, mode="bilinear", align_corners=False, padding_mode="border")

# ------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------
def d_hinge_loss(real_logits, fake_logits):
    loss_real = torch.mean(F.relu(1.0 - real_logits))
    loss_fake = torch.mean(F.relu(1.0 + fake_logits))
    return loss_real + loss_fake

def g_hinge_loss(fake_logits):
    return -torch.mean(fake_logits)

def leakage_loss_simple(fake_imgs, bbox_mask):
    outside_mask = 1.0 - bbox_mask
    return (fake_imgs.abs() * outside_mask).mean()

def leakage_loss_background(fake_imgs, real_imgs, bbox_mask):
    outside_mask = 1.0 - bbox_mask
    return (torch.abs(fake_imgs - real_imgs) * outside_mask).mean()

def path_length_penalty(fake_imgs, latents):
    """StyleGAN2 path length regularization.

    Encourages a fixed-size step in latent space to result in a fixed-magnitude
    change in image space, making latent-space interpolation smooth and
    consistent across the full generator.

    The penalty is (||J^T y||_2 - a)^2 where:
      - J is the Jacobian of generated pixels w.r.t. latents
      - y is a random unit vector in pixel space (Monte Carlo estimator)
      - a is a running mean of path lengths (updated by caller via EMA)

    Returns:
        path_lengths: per-sample L2 path lengths, shape [B].
        Caller computes (path_lengths - ema_mean).pow(2).mean() and adds
        it to the generator loss.

    Note: `latents` must have requires_grad=True before calling G.forward().
    """
    noise = torch.randn_like(fake_imgs) / math.sqrt(
        fake_imgs.shape[2] * fake_imgs.shape[3]
    )
    grad = torch.autograd.grad(
        outputs=(fake_imgs * noise).sum(),
        inputs=latents,
        create_graph=True,
    )[0]
    return grad.pow(2).sum(dim=1).sqrt()   # [B]


def r1_gradient_penalty(real_imgs, real_logits):
    """R1 gradient penalty (Mescheder et al., 2018): E[||∇D(x)||²] where x ~ p_data.
    Computed in float32 to avoid precision issues with second-order gradients."""
    gradients = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_imgs,
        create_graph=True,
    )[0]
    return gradients.pow(2).reshape(gradients.shape[0], -1).sum(1).mean()

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
def save_or_show_images(tensor, title="Generated Images", save_path=None):
    grid = vutils.make_grid(tensor[:16], nrow=4, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.title(title)
    plt.axis("off")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()