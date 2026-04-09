import torch
import torch.nn as nn


class AugmentPipe(nn.Module):
    """Adaptive Data Augmentation pipeline (StyleGAN2-ADA).

    Both real and fake images are passed through the same stochastic
    augmentations before the discriminator sees them.  This prevents D
    from overfitting to dataset statistics (e.g. exact pixel distributions
    of CelebA) rather than learning genuine realism.

    The global strength `p` (0 → 1) is adjusted automatically via
    `update_p()`:  when D is too confident on real images, p increases
    (harder task); when D under-discriminates, p decreases.

    Augmentations applied independently per image:
      - Brightness  : additive Gaussian shift
      - Contrast    : multiplicative rescaling around per-image mean
      - Cutout      : random rectangle zeroed out

    Note: x-flip is intentionally omitted.  Flipping would misalign the
    bounding-box crop coordinates used by the ROI discriminator.  Add it
    at the dataset level instead if desired.

    Gradients flow through all operations, so this pipe can be used during
    both discriminator training (on detached tensors) and generator training
    (where gradients must reach G's parameters).
    """

    def __init__(self,
                 brightness: float = 1.0,
                 brightness_std: float = 0.2,
                 contrast: float = 1.0,
                 contrast_range: tuple = (0.5, 2.0),
                 cutout: float = 0.5,
                 cutout_fraction: float = 0.5):
        super().__init__()
        self.brightness       = brightness
        self.brightness_std   = brightness_std
        self.contrast         = contrast
        self.contrast_lo, self.contrast_hi = contrast_range
        self.cutout           = cutout
        self.cutout_fraction  = cutout_fraction
        # Master probability multiplier — updated each training step via update_p()
        self.register_buffer('p', torch.zeros([]))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Augment a batch of images.

        Args:
            images: float tensor [B, C, H, W] in the range [-1, 1]
        Returns:
            Augmented images, same shape, clamped to [-1, 1]
        """
        if self.p.item() == 0.0:
            return images

        B, C, H, W = images.shape
        device = images.device
        images = images.clone()

        def coin(base_prob: float) -> torch.Tensor:
            """Per-image boolean mask: True where this aug fires."""
            return torch.rand(B, device=device) < (base_prob * self.p.item())

        # ------ Brightness jitter ------
        mask = coin(self.brightness)
        if mask.any():
            delta = torch.randn(B, 1, 1, 1, device=device) * self.brightness_std
            images = images + delta * mask.float().view(B, 1, 1, 1)

        # ------ Contrast jitter ------
        mask = coin(self.contrast)
        if mask.any():
            lo, hi = self.contrast_lo, self.contrast_hi
            factor = torch.empty(B, 1, 1, 1, device=device).uniform_(lo, hi)
            img_mean = images.mean(dim=[2, 3], keepdim=True)
            adjusted = (images - img_mean) * factor + img_mean
            m = mask.view(B, 1, 1, 1)
            images = torch.where(m, adjusted, images)

        # ------ Cutout — zero a random rectangle ------
        mask = coin(self.cutout)
        if mask.any():
            frac = self.cutout_fraction
            ch = max(1, int(H * frac))
            cw = max(1, int(W * frac))
            y0 = torch.randint(0, max(1, H - ch + 1), (B,), device=device)
            x0 = torch.randint(0, max(1, W - cw + 1), (B,), device=device)
            for idx in mask.nonzero(as_tuple=True)[0]:
                images[idx, :, y0[idx]:y0[idx] + ch, x0[idx]:x0[idx] + cw] = 0.0

        return images.clamp(-1.0, 1.0)

    def update_p(self, real_sign_mean: float, batch_size: int,
                 interval: int, ada_kimg: float, ada_target: float) -> None:
        """Adapt augmentation probability p (StyleGAN2-ADA §3.3).

        Tracks mean(sign(D(real))) as an overfitting indicator:
          - If > ada_target  → D is too confident on reals → raise p
          - If < ada_target  → D is under-discriminating  → lower p

        The magnitude of each adjustment is calibrated so that p moves by
        ~1 over ada_kimg thousand images when the sign is fully wrong.

        Args:
            real_sign_mean: mean(sign(D(real_images))) averaged over `interval` steps
            batch_size:     images per step
            interval:       steps between p updates (to match accumulation window)
            ada_kimg:       adjustment speed in thousands of images (larger = slower)
            ada_target:     target overfitting level; 0.6 ≈ D correct 80% of the time
        """
        import math
        adjust = math.copysign(1.0, real_sign_mean - ada_target) \
                 * (batch_size * interval) / (ada_kimg * 1000.0)
        self.p.copy_((self.p + adjust).clamp_(0.0, 1.0))
