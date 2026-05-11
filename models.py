"""Small-scale StyleGAN2-inspired generator and discriminator.

Faithful to the StyleGAN2 architecture (Karras et al., 2020):
  - 8-layer mapping network z → w
  - Synthesis network from a learned 4x4 constant via N upsample blocks
  - Modulated convolutions with weight demodulation (no AdaIN, no instance norm)
  - Per-layer learned noise injection
  - Skip-to-RGB output accumulation
  - Residual discriminator with MinibatchStdDev

Scaled down vs the full reference by reducing channel_base / channel_max,
so the network trains at 64x64 or 128x128 on a single GPU.

The architecture (block count and channels per resolution) is derived from
img_resolution and channel_{base,max} — no hard-coded blocks.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Equalized-LR primitives
# ---------------------------------------------------------------------------
class EqualizedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.scale = 1.0 / math.sqrt(in_ch * kernel_size * kernel_size)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)


class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = (1.0 / math.sqrt(in_features)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        b = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, self.weight * self.scale, b)


class PixelNorm(nn.Module):
    def forward(self, x):
        # eps=1e-4: avoids 0/0 NaN in fp16 where 1e-8 rounds to zero
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-4)


# ---------------------------------------------------------------------------
# Anti-aliased blur upsample
# ---------------------------------------------------------------------------
_BLUR_KERNEL_CACHE: dict = {}


def _blur_kernel(channels, device, dtype):
    key = (channels, device, dtype)
    if key not in _BLUR_KERNEL_CACHE:
        k = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
        k = k[:, None] * k[None, :]
        k = k / k.sum()
        _BLUR_KERNEL_CACHE[key] = k.unsqueeze(0).unsqueeze(0).expand(channels, 1, 3, 3).contiguous()
    return _BLUR_KERNEL_CACHE[key]


def blur_upsample(x):
    """2x bilinear upsample followed by a [1,2,1] FIR blur."""
    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    C = x.shape[1]
    k = _blur_kernel(C, x.device, x.dtype)
    return F.conv2d(x, k, padding=1, groups=C)


# ---------------------------------------------------------------------------
# StyleGAN2 modulated conv with demodulation
# ---------------------------------------------------------------------------
class ModulatedConv2d(nn.Module):
    """Modulate conv weights by a per-sample style; demodulate to unit output std.

        w_mod[b,o,i,k,l] = w[o,i,k,l] * s[b,i]
        w_dem[b,o,...]   = w_mod / sqrt(sum w_mod**2 + eps)
        y                = grouped_conv(x, w_dem)
    """
    def __init__(self, in_ch, out_ch, kernel_size, style_dim,
                 demodulate=True, upsample=False):
        super().__init__()
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size))
        self.lr_scale = 1.0 / math.sqrt(in_ch * kernel_size * kernel_size)

        # Affine: w -> per-input-channel modulation. Init bias=1 so initial mod is identity.
        self.affine = EqualizedLinear(style_dim, in_ch, bias=True)
        nn.init.ones_(self.affine.bias)

        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x, style):
        B, C_in, H, W = x.shape

        s = self.affine(style)                           # [B, in_ch]
        w = self.weight * self.lr_scale                  # [out_ch, in_ch, k, k]
        w_mod = w[None] * s[:, None, :, None, None]      # [B, out_ch, in_ch, k, k]

        if self.demodulate:
            d = w_mod.pow(2).sum(dim=[2, 3, 4], keepdim=True).add(1e-8).rsqrt()
            w_mod = w_mod * d

        if self.upsample:
            x = blur_upsample(x)
            H, W = H * 2, W * 2

        # Grouped conv: fold batch into channels for a single conv call
        x = x.reshape(1, B * C_in, H, W)
        w_mod = w_mod.reshape(B * self.out_ch, C_in, self.kernel_size, self.kernel_size)
        x = F.conv2d(x, w_mod, padding=self.padding, groups=B)
        x = x.reshape(B, self.out_ch, H, W)
        return x + self.bias.view(1, -1, 1, 1)


# ---------------------------------------------------------------------------
# Synthesis layer = modulated conv + noise + LReLU
# ---------------------------------------------------------------------------
class SynthesisLayer(nn.Module):
    def __init__(self, in_ch, out_ch, style_dim, upsample=False):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, 3, style_dim, upsample=upsample)
        self.noise_weight = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, style, noise=None):
        x = self.conv(x, style)
        if noise is None:
            b, _, h, w = x.shape
            noise = torch.randn(b, 1, h, w, device=x.device, dtype=x.dtype)
        x = x + self.noise_weight * noise
        return self.act(x)


class ToRGB(nn.Module):
    """Project a feature map to RGB at the current resolution (no demodulation).

    Outputs are accumulated across resolutions via upsample-and-add (skip-to-RGB).
    """
    def __init__(self, in_ch, num_channels, style_dim):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, num_channels, 1, style_dim, demodulate=False)

    def forward(self, x, style, skip=None):
        rgb = self.conv(x, style)
        if skip is not None:
            skip_up = F.interpolate(skip, size=rgb.shape[-2:],
                                    mode="bilinear", align_corners=False)
            rgb = rgb + skip_up
        return rgb


# ---------------------------------------------------------------------------
# Mapping network: z -> w
# ---------------------------------------------------------------------------
class MappingNetwork(nn.Module):
    """8-layer FC that maps PixelNormed z to the disentangled style space w.

    A reduced lr_mul (0.01) is applied on the mapping FC layers, matching the
    StyleGAN2 reference — this stabilises training of the mapping network
    relative to the much larger synthesis network.
    """
    def __init__(self, z_dim, w_dim, num_layers=8, lr_mul=0.01):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.pixel_norm = PixelNorm()
        layers = []
        in_dim = z_dim
        for _ in range(num_layers):
            layers += [EqualizedLinear(in_dim, w_dim, lr_mul=lr_mul),
                       nn.LeakyReLU(0.2, inplace=True)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(self.pixel_norm(z))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class SynthesisBlock(nn.Module):
    """One resolution step: optional upsample-conv, plain-conv, ToRGB."""
    def __init__(self, in_ch, out_ch, style_dim, num_channels, is_first=False):
        super().__init__()
        self.is_first = is_first
        if not is_first:
            self.conv0 = SynthesisLayer(in_ch, out_ch, style_dim, upsample=True)
        self.conv1 = SynthesisLayer(out_ch, out_ch, style_dim)
        self.to_rgb = ToRGB(out_ch, num_channels, style_dim)

    def forward(self, x, w, skip_rgb=None):
        if not self.is_first:
            x = self.conv0(x, w)
        x = self.conv1(x, w)
        rgb = self.to_rgb(x, w, skip=skip_rgb)
        return x, rgb


def _channels_for_resolution(res, channel_base, channel_max):
    """StyleGAN2 NF schedule: clip(channel_base // res, 1, channel_max)."""
    return min(channel_max, max(1, channel_base // res))


class Generator(nn.Module):
    """Unconditional StyleGAN2 generator.

    Args:
        z_dim: latent dimensionality.
        w_dim: style-vector dimensionality.
        img_resolution: output resolution (must be a power of two, >= 4).
        img_channels: number of output channels (3 for RGB).
        channel_base, channel_max: control the per-resolution feature widths.
        mapping_layers: number of FC layers in the mapping network.
    """
    def __init__(self, z_dim=256, w_dim=256, img_resolution=128, img_channels=3,
                 channel_base=8192, channel_max=256, mapping_layers=8):
        super().__init__()
        assert img_resolution >= 4 and (img_resolution & (img_resolution - 1)) == 0, \
            "img_resolution must be a power of two >= 4"
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        # Resolutions: 4, 8, 16, ..., img_resolution
        self.resolutions = [2 ** i for i in range(2, int(math.log2(img_resolution)) + 1)]
        channels = [_channels_for_resolution(r, channel_base, channel_max) for r in self.resolutions]

        self.mapping = MappingNetwork(z_dim, w_dim, num_layers=mapping_layers)

        # Learned 4x4 constant for the first block input
        self.const = nn.Parameter(torch.randn(1, channels[0], 4, 4))

        # Build synthesis blocks
        self.blocks = nn.ModuleList()
        prev_ch = channels[0]
        for i, (res, ch) in enumerate(zip(self.resolutions, channels)):
            block = SynthesisBlock(
                in_ch=prev_ch, out_ch=ch, style_dim=w_dim,
                num_channels=img_channels, is_first=(i == 0),
            )
            self.blocks.append(block)
            prev_ch = ch

        # Total style inputs (used for the path-length penalty if mixing styles).
        # Each block uses 1 (first) or 2 styles (upsample + plain) plus 1 ToRGB.
        self.num_ws = 0
        for i, _ in enumerate(self.resolutions):
            self.num_ws += (1 if i == 0 else 2) + 1

    def forward(self, z, return_w=False):
        w = self.mapping(z)                                  # [B, w_dim]
        x = self.const.expand(z.size(0), -1, -1, -1)
        rgb = None
        for block in self.blocks:
            x, rgb = block(x, w, skip_rgb=rgb)
        img = torch.tanh(rgb)
        if return_w:
            return img, w
        return img


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------
class MinibatchStdDev(nn.Module):
    """Append per-group stddev as a feature channel — penalises mode collapse."""
    def __init__(self, group_size=4, num_features=1):
        super().__init__()
        self.group_size = group_size
        self.num_features = num_features

    def forward(self, x):
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


class DiscriminatorBlock(nn.Module):
    """Pre-activation residual downsample block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = EqualizedConv2d(in_ch, in_ch,  3, 1, 1)
        self.conv2 = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)
        self.skip  = nn.Sequential(
            EqualizedConv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        residual = self.skip(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        return (x + residual) * (1.0 / math.sqrt(2))


class Discriminator(nn.Module):
    """Unconditional StyleGAN2 discriminator.

    Mirrors the generator schedule: residual blocks downsample from
    img_resolution to 4x4, then a MinibatchStdDev + 3x3 conv + linear head.
    """
    def __init__(self, img_resolution=128, img_channels=3,
                 channel_base=8192, channel_max=256, mbstd_group=4):
        super().__init__()
        assert img_resolution >= 4 and (img_resolution & (img_resolution - 1)) == 0
        self.img_resolution = img_resolution

        # Resolutions from input down to 4: e.g. [128, 64, 32, 16, 8, 4]
        resolutions = [img_resolution >> i for i in range(int(math.log2(img_resolution)) - 1)]
        channels = [_channels_for_resolution(r, channel_base, channel_max) for r in resolutions]

        # Initial 1x1 from RGB to first resolution channels
        self.from_rgb = EqualizedConv2d(img_channels, channels[0], 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Build downsample blocks: channels[i] -> channels[i+1]
        self.blocks = nn.ModuleList()
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            self.blocks.append(DiscriminatorBlock(in_ch, out_ch))

        # Final 4x4 head
        last_ch = channels[-1]
        self.mbstd = MinibatchStdDev(group_size=mbstd_group)
        self.final_conv = EqualizedConv2d(last_ch + 1, last_ch, 3, 1, 1)
        self.final_linear = EqualizedLinear(last_ch * 4 * 4, last_ch)
        self.out_linear = EqualizedLinear(last_ch, 1)

    def forward(self, img):
        x = self.act(self.from_rgb(img))
        for block in self.blocks:
            x = block(x)
        x = self.mbstd(x)
        x = self.act(self.final_conv(x))
        x = x.reshape(x.size(0), -1)
        x = self.act(self.final_linear(x))
        return self.out_linear(x)
