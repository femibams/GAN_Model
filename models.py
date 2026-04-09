import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Equalized Learning Rate primitives
# ---------------------------------------------------------------------------
class EqualizedConv2d(nn.Module):
    """Conv2d with equalized learning rate (StyleGAN2)."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.scale = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias,
                        self.stride, self.padding)


class EqualizedLinear(nn.Module):
    """Linear with equalized learning rate (StyleGAN2)."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = 1.0 / math.sqrt(in_features)

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)


# ---------------------------------------------------------------------------
# PixelNorm — normalize each vector to unit length along the channel dim
# ---------------------------------------------------------------------------
class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # eps=1e-4: avoids 0/0 NaN in fp16 where 1e-8 rounds to zero
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-4)


# ---------------------------------------------------------------------------
# Minibatch Standard Deviation
# ---------------------------------------------------------------------------
class MinibatchStdDev(nn.Module):
    """Appends per-group standard deviation as an extra feature channel.

    Directly penalises mode-collapsed generators that produce low-variance
    batches. Appended before the final discriminator conv.
    """
    def __init__(self, group_size: int = 4, num_features: int = 1):
        super().__init__()
        self.group_size = group_size
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        G = min(self.group_size, B)
        while G > 1 and B % G != 0:
            G -= 1
        F_feat = self.num_features
        c = C // F_feat

        y = x.reshape(G, B // G, F_feat, c, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = (y.pow(2).mean(dim=0) + 1e-8).sqrt()
        y = y.mean(dim=[2, 3, 4], keepdim=True)
        y = y.squeeze(2).repeat(G, 1, H, W)
        return torch.cat([x, y], dim=1)


# ---------------------------------------------------------------------------
# Anti-aliased blur upsampling
# ---------------------------------------------------------------------------
def _blur_kernel(channels: int, device, dtype) -> torch.Tensor:
    k = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
    k = k[:, None] * k[None, :]
    k = k / k.sum()
    return k.unsqueeze(0).unsqueeze(0).expand(channels, 1, 3, 3)


def blur_upsample(x: torch.Tensor) -> torch.Tensor:
    """2× bilinear upsample followed by a [1,2,1] FIR blur (StyleGAN2)."""
    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    C = x.shape[1]
    k = _blur_kernel(C, x.device, x.dtype)
    return F.conv2d(x, k, padding=1, groups=C)


# ---------------------------------------------------------------------------
# StyleGAN2 Modulated Convolution with Weight Demodulation
# ---------------------------------------------------------------------------
class ModulatedConv2d(nn.Module):
    """StyleGAN2 weight-modulated convolution with demodulation.

    Unlike AdaIN (which normalises *features* then rescales), this modulates
    the *convolution weights* by a per-sample style vector and then
    demodulates to normalise the expected output standard deviation to 1.

        w_mod[b,o,i,k,l] = w[o,i,k,l] * s[b,i]          (modulate)
        w_dem[b,o,…]      = w_mod / sqrt(sum w_mod² + ε)  (demodulate)
        y = grouped_conv(x, w_dem)

    Benefits over AdaIN:
    - No feature-normalisation instability on small spatial maps (e.g. 4×4).
    - Eliminates the characteristic AdaIN "droplet/blob" artifacts.
    - Style conditioning acts on the frequency content, not just the mean/var.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 style_dim: int, demodulate: bool = True,
                 upsample: bool = False):
        super().__init__()
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.padding = kernel_size // 2

        # Weight stored at unit scale; equalized LR scale applied at runtime
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel_size, kernel_size)
        )
        self.lr_scale = 1.0 / math.sqrt(in_ch * kernel_size * kernel_size)

        # Affine: w → per-input-channel modulation scale.
        # bias=1 at init so initial modulation is near identity.
        self.affine = EqualizedLinear(style_dim, in_ch, bias=True)
        nn.init.ones_(self.affine.bias)

        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape

        # Affine: style → per-input-channel scale [B, in_ch]
        s = self.affine(style)

        # Equalized weight [out_ch, in_ch, k, k]
        w = self.weight * self.lr_scale

        # Modulate: broadcast over batch and spatial dims
        w_mod = w[None] * s[:, None, :, None, None]   # [B, out_ch, in_ch, k, k]

        if self.demodulate:
            # Demodulate: normalise per output channel
            d = w_mod.pow(2).sum(dim=[2, 3, 4], keepdim=True).add(1e-8).rsqrt()
            w_mod = w_mod * d

        if self.upsample:
            x = blur_upsample(x)
            H, W = H * 2, W * 2

        # Grouped conv: fold batch into channel axis for a single conv call
        x = x.reshape(1, B * C_in, H, W)
        w_mod = w_mod.reshape(B * self.out_ch, C_in, self.kernel_size, self.kernel_size)
        x = F.conv2d(x, w_mod, padding=self.padding, groups=B)
        x = x.reshape(B, self.out_ch, H, W)

        return x + self.bias.view(1, -1, 1, 1)


# ---------------------------------------------------------------------------
# Synthesis Layer: ModulatedConv2d + stochastic noise + activation
# ---------------------------------------------------------------------------
class SynthesisLayer(nn.Module):
    """One StyleGAN2 synthesis step: modulated conv → additive noise → LReLU."""
    def __init__(self, in_ch: int, out_ch: int, style_dim: int,
                 upsample: bool = False):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, 3, style_dim,
                                    upsample=upsample)
        # Per-channel learned noise scale (zero-init: starts silent)
        self.noise_weight = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, style: torch.Tensor,
                noise: torch.Tensor = None) -> torch.Tensor:
        x = self.conv(x, style)
        if noise is None:
            b, _, h, w = x.shape
            noise = torch.randn(b, 1, h, w, device=x.device, dtype=x.dtype)
        x = x + self.noise_weight * noise
        return self.act(x)


# ---------------------------------------------------------------------------
# ToRGB: modulated 1×1 conv projecting feature map to partial RGB
# ---------------------------------------------------------------------------
class ToRGB(nn.Module):
    """Project a feature map to RGB at the current resolution (no demodulation).

    Accumulated across resolutions via upsample-and-add (skip-to-RGB), giving
    the gradient a short path to the output loss at every scale.
    """
    def __init__(self, in_ch: int, num_channels: int, style_dim: int):
        super().__init__()
        # No demodulation for the output projection (StyleGAN2 design choice)
        self.conv = ModulatedConv2d(in_ch, num_channels, 1, style_dim,
                                    demodulate=False)

    def forward(self, x: torch.Tensor, style: torch.Tensor,
                skip: torch.Tensor = None) -> torch.Tensor:
        rgb = self.conv(x, style)
        if skip is not None:
            skip_up = F.interpolate(skip, size=rgb.shape[-2:],
                                    mode="bilinear", align_corners=False)
            rgb = rgb + skip_up
        return rgb


# ---------------------------------------------------------------------------
# Mapping Network: (z, text_emb) → w
# ---------------------------------------------------------------------------
class MappingNetwork(nn.Module):
    """8-layer fully-connected mapping network: (z, text) → style vector w.

    z is PixelNorm-normalised to prevent its magnitude from dominating.
    Text is projected to z_dim so both inputs are the same size before
    concatenation. The resulting 2*z_dim vector is then passed through 8
    equalized-LR layers to produce the final disentangled w.

    Using 8 layers (vs. the previous 2) gives the network much more capacity
    to learn a curved, disentangled latent space where each direction
    corresponds to a distinct and separable image attribute.
    """
    def __init__(self, z_dim: int, text_dim: int, w_dim: int,
                 num_layers: int = 8):
        super().__init__()
        self.pixel_norm = PixelNorm()

        # Project text into the same dimensionality as z for concatenation
        self.text_proj = nn.Sequential(
            EqualizedLinear(text_dim, z_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 8-layer FC: (z || text_proj) → w
        in_dim = z_dim * 2   # z_dim from z + z_dim from text_proj
        layers = []
        for _ in range(num_layers):
            layers += [EqualizedLinear(in_dim, w_dim),
                       nn.LeakyReLU(0.2, inplace=True)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        z = self.pixel_norm(z)
        t = self.text_proj(text_emb)
        x = torch.cat([z, t], dim=1)   # [B, z_dim * 2]
        return self.net(x)              # [B, w_dim]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class GeneratorWithMask(nn.Module):
    """StyleGAN2-architecture text-conditioned generator with bbox guidance.

    Architecture overview
    ─────────────────────
    MappingNetwork (8 layers)
      (z, text_emb) → w ∈ R^{w_dim}

    Synthesis Network (learned 4×4 constant → 128×128 via 5 upsample blocks)
      Each block has two SynthesisLayers (modulated conv + noise + LReLU)
      and one ToRGB projection.  The ToRGB outputs are accumulated via
      upsample-and-add (skip-to-RGB), providing the gradient a short path
      to the loss at every resolution.

    Bbox mask conditioning
      A 1×1 EqualizedConv2d encodes the resized mask to feature-space channels
      and adds it as a spatial bias after the first SynthesisLayer of each
      block.  Zero-initialised weights mean the mask has no effect at the start
      of training and the network learns to use it only as needed.

    Key improvements over the previous (AdaIN) version
    ────────────────────────────────────────────────────
    • ModulatedConv2d + demodulation instead of AdaIN:
        - Modulates conv *weights* per-sample; demodulation normalises output std.
        - Eliminates the AdaIN blob/droplet artifacts.
        - Numerically stable at any spatial resolution in fp16.
    • 8-layer mapping network → disentangled w space (was 2 layers).
    • Two synthesis layers per resolution block (was one).
    • Separate mask spatial encoder per block instead of channel concatenation.
    """
    def __init__(self, noise_size: int, feature_size: int, num_channels: int,
                 embedding_size: int, reduced_dim_size: int):
        super().__init__()
        F_ = feature_size
        w_dim = reduced_dim_size

        # --- Mapping network ---
        self.mapping = MappingNetwork(noise_size, embedding_size, w_dim)

        # --- Stem: learned 4×4 constant → first synthesis layer ---
        self.const = nn.Parameter(torch.randn(1, F_ * 8, 4, 4))
        self.stem_l1 = SynthesisLayer(F_ * 8, F_ * 8, w_dim)
        self.stem_l2 = SynthesisLayer(F_ * 8, F_ * 8, w_dim)
        self.stem_rgb = ToRGB(F_ * 8, num_channels, w_dim)
        self.stem_mask = EqualizedConv2d(1, F_ * 8, 1, bias=False)
        nn.init.zeros_(self.stem_mask.weight)

        # --- Block 1: 4 → 8,  F*8 → F*8 ---
        self.up1_l1   = SynthesisLayer(F_ * 8, F_ * 8, w_dim, upsample=True)
        self.up1_l2   = SynthesisLayer(F_ * 8, F_ * 8, w_dim)
        self.up1_rgb  = ToRGB(F_ * 8, num_channels, w_dim)
        self.up1_mask = EqualizedConv2d(1, F_ * 8, 1, bias=False)
        nn.init.zeros_(self.up1_mask.weight)

        # --- Block 2: 8 → 16,  F*8 → F*4 ---
        self.up2_l1   = SynthesisLayer(F_ * 8, F_ * 4, w_dim, upsample=True)
        self.up2_l2   = SynthesisLayer(F_ * 4, F_ * 4, w_dim)
        self.up2_rgb  = ToRGB(F_ * 4, num_channels, w_dim)
        self.up2_mask = EqualizedConv2d(1, F_ * 4, 1, bias=False)
        nn.init.zeros_(self.up2_mask.weight)

        # --- Block 3: 16 → 32,  F*4 → F*2 ---
        self.up3_l1   = SynthesisLayer(F_ * 4, F_ * 2, w_dim, upsample=True)
        self.up3_l2   = SynthesisLayer(F_ * 2, F_ * 2, w_dim)
        self.up3_rgb  = ToRGB(F_ * 2, num_channels, w_dim)
        self.up3_mask = EqualizedConv2d(1, F_ * 2, 1, bias=False)
        nn.init.zeros_(self.up3_mask.weight)

        # --- Block 4: 32 → 64,  F*2 → F ---
        self.up4_l1   = SynthesisLayer(F_ * 2, F_,     w_dim, upsample=True)
        self.up4_l2   = SynthesisLayer(F_,     F_,     w_dim)
        self.up4_rgb  = ToRGB(F_,     num_channels, w_dim)
        self.up4_mask = EqualizedConv2d(1, F_,     1, bias=False)
        nn.init.zeros_(self.up4_mask.weight)

        # --- Block 5: 64 → 128,  F → F ---
        self.up5_l1   = SynthesisLayer(F_,     F_,     w_dim, upsample=True)
        self.up5_l2   = SynthesisLayer(F_,     F_,     w_dim)
        self.up5_rgb  = ToRGB(F_,     num_channels, w_dim)
        self.up5_mask = EqualizedConv2d(1, F_,     1, bias=False)
        nn.init.zeros_(self.up5_mask.weight)

    def _inject_mask(self, x: torch.Tensor, bbox_mask: torch.Tensor,
                     mask_enc: nn.Module) -> torch.Tensor:
        """Resize bbox_mask to x's spatial size and add encoded spatial bias."""
        mask_r = F.interpolate(bbox_mask, size=x.shape[-2:], mode="nearest")
        return x + mask_enc(mask_r)

    def forward(self, noise: torch.Tensor, text_embeddings: torch.Tensor,
                bbox_mask: torch.Tensor) -> torch.Tensor:
        B = noise.size(0)

        # (z, text) → style vector w used at every synthesis layer
        w = self.mapping(noise, text_embeddings)            # [B, w_dim]

        # Stem: two synthesis layers on the learned 4×4 constant
        x = self.const.expand(B, -1, -1, -1)               # [B, F*8, 4, 4]
        x = self._inject_mask(x, bbox_mask, self.stem_mask)
        x = self.stem_l1(x, w)
        x = self.stem_l2(x, w)
        rgb = self.stem_rgb(x, w)                           # [B, 3,   4, 4]

        # Block 1: 4 → 8
        x = self.up1_l1(x, w)                              # [B, F*8, 8, 8]
        x = self._inject_mask(x, bbox_mask, self.up1_mask)
        x = self.up1_l2(x, w)
        rgb = self.up1_rgb(x, w, skip=rgb)                 # [B, 3,   8, 8]

        # Block 2: 8 → 16
        x = self.up2_l1(x, w)                              # [B, F*4, 16, 16]
        x = self._inject_mask(x, bbox_mask, self.up2_mask)
        x = self.up2_l2(x, w)
        rgb = self.up2_rgb(x, w, skip=rgb)                 # [B, 3,   16, 16]

        # Block 3: 16 → 32
        x = self.up3_l1(x, w)                              # [B, F*2, 32, 32]
        x = self._inject_mask(x, bbox_mask, self.up3_mask)
        x = self.up3_l2(x, w)
        rgb = self.up3_rgb(x, w, skip=rgb)                 # [B, 3,   32, 32]

        # Block 4: 32 → 64
        x = self.up4_l1(x, w)                              # [B, F,   64, 64]
        x = self._inject_mask(x, bbox_mask, self.up4_mask)
        x = self.up4_l2(x, w)
        rgb = self.up4_rgb(x, w, skip=rgb)                 # [B, 3,   64, 64]

        # Block 5: 64 → 128
        x = self.up5_l1(x, w)                              # [B, F,   128, 128]
        x = self._inject_mask(x, bbox_mask, self.up5_mask)
        x = self.up5_l2(x, w)
        rgb = self.up5_rgb(x, w, skip=rgb)                 # [B, 3,   128, 128]

        return torch.tanh(rgb)


# ---------------------------------------------------------------------------
# Residual Discriminator Block
# ---------------------------------------------------------------------------
class DiscriminatorBlock(nn.Module):
    """Residual downsampling block with spectral normalisation.

    Pre-activation residual design:
        skip = avgpool(1×1_conv(x))
        x    = act(conv3x3(x)) → act(conv3x3) → avgpool
        out  = (x + skip) / sqrt(2)

    The skip connection keeps the gradient signal alive even when D is very
    confident, which matters most in early training when D trivially separates
    real from fake and the residual path is the only gradient that reaches G.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = sn(EqualizedConv2d(in_ch, in_ch,  3, 1, 1))
        self.conv2 = sn(EqualizedConv2d(in_ch, out_ch, 3, 1, 1))
        self.skip  = nn.Sequential(
            sn(EqualizedConv2d(in_ch, out_ch, 1, bias=False)),
            nn.AvgPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        # Normalise the sum so the combined signal stays at unit variance
        return (x + residual) * (1.0 / math.sqrt(2))


# ---------------------------------------------------------------------------
# Global Discriminator — residual blocks + projection text conditioning
# ---------------------------------------------------------------------------
class GlobalDiscriminator(nn.Module):
    """Full-image discriminator with residual blocks and text projection.

    Residual blocks (DiscriminatorBlock) improve gradient flow at all depths.
    Text conditioning uses the projection discriminator (Miyato & Koyama, 2018):
        D(x | c) = D_base(x) + <φ(c), pool(features)>
    which penalises text-image misalignment independently of visual quality.
    """
    def __init__(self, num_channels: int, base_ch: int = 64,
                 mask_ch: int = 1, text_dim: int = 0):
        super().__init__()
        sn = nn.utils.spectral_norm
        in_ch = num_channels + mask_ch

        # Initial 1×1 projection (no spatial change, just channel expansion)
        self.from_input = sn(EqualizedConv2d(in_ch, base_ch, 1))
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Residual downsampling: 128→64→32→16→8
        self.block1 = DiscriminatorBlock(base_ch,      base_ch * 2)   # →64
        self.block2 = DiscriminatorBlock(base_ch * 2,  base_ch * 4)   # →32
        self.block3 = DiscriminatorBlock(base_ch * 4,  base_ch * 8)   # →16
        self.block4 = DiscriminatorBlock(base_ch * 8,  base_ch * 8)   # →8

        feat_ch = base_ch * 8 + 1    # +1 channel from MinibatchStdDev
        self.mbstd      = MinibatchStdDev(group_size=4, num_features=1)
        self.final_conv = sn(EqualizedConv2d(feat_ch, base_ch * 8, 3, 1, 1))
        # Flatten [B, base_ch*8, 8, 8] → [B, base_ch*8*64] → [B, 1]
        self.final_linear = sn(EqualizedLinear(base_ch * 8 * 8 * 8, 1))

        # Optional projection conditioning on text
        if text_dim > 0:
            self.text_proj = sn(EqualizedLinear(text_dim, base_ch * 8))
        else:
            self.text_proj = None

    def forward(self, img: torch.Tensor, bbox_mask: torch.Tensor,
                text_emb: torch.Tensor = None) -> torch.Tensor:
        x = torch.cat([img, bbox_mask], dim=1)
        x = self.act(self.from_input(x))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)                            # [B, base_ch*8, 8, 8]

        # Pool spatial features before mbstd for text projection
        feat_pooled = x.mean(dim=[2, 3])              # [B, base_ch*8]

        x = self.mbstd(x)                             # [B, feat_ch, 8, 8]
        x = self.act(self.final_conv(x))              # [B, base_ch*8, 8, 8]
        feat_flat = x.reshape(x.size(0), -1)          # [B, base_ch*8*64]
        logits = self.final_linear(feat_flat)          # [B, 1]

        # Add projection term: <φ(text), pool(features)>
        if self.text_proj is not None and text_emb is not None:
            logits = logits + (
                feat_pooled * self.text_proj(text_emb)
            ).sum(dim=1, keepdim=True)

        return logits


# ---------------------------------------------------------------------------
# RoI Discriminator — residual blocks on the face crop
# ---------------------------------------------------------------------------
class RoIDiscriminator(nn.Module):
    """Face-crop (RoI) discriminator with residual blocks.

    Operates on 64×64 face crops regardless of text.  Its sole job is to
    enforce high-frequency realism in the face region; text-image alignment
    is handled by GlobalDiscriminator.
    """
    def __init__(self, num_channels: int, base_ch: int = 64):
        super().__init__()
        sn = nn.utils.spectral_norm

        self.from_input = sn(EqualizedConv2d(num_channels, base_ch, 1))
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Residual downsampling: 64→32→16→8→4
        self.block1 = DiscriminatorBlock(base_ch,     base_ch * 2)    # →32
        self.block2 = DiscriminatorBlock(base_ch * 2, base_ch * 4)    # →16
        self.block3 = DiscriminatorBlock(base_ch * 4, base_ch * 8)    # →8
        self.block4 = DiscriminatorBlock(base_ch * 8, base_ch * 8)    # →4

        feat_ch = base_ch * 8 + 1
        self.mbstd      = MinibatchStdDev(group_size=4, num_features=1)
        self.final_conv = sn(EqualizedConv2d(feat_ch, base_ch * 8, 3, 1, 1))
        # Flatten [B, base_ch*8, 4, 4] → [B, base_ch*8*16] → [B, 1]
        self.final_linear = sn(EqualizedLinear(base_ch * 8 * 4 * 4, 1))

    def forward(self, roi_img: torch.Tensor) -> torch.Tensor:
        x = self.act(self.from_input(roi_img))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)                            # [B, base_ch*8, 4, 4]
        x = self.mbstd(x)
        x = self.act(self.final_conv(x))
        x = x.reshape(x.size(0), -1)                 # [B, base_ch*8*16]
        return self.final_linear(x)                   # [B, 1]
