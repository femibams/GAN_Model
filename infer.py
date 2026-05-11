"""Inference: generate sample faces from a trained checkpoint."""
import argparse
import os

import torch

import config
from models import Generator
from utils import save_image_grid


def parse_args():
    p = argparse.ArgumentParser(description="Generate faces from a trained StyleGAN2-small checkpoint")
    p.add_argument("--ckpt", type=str,
                   default=os.path.join("outputs", "checkpoints", "latest.pth"),
                   help="Path to checkpoint produced by train.py")
    p.add_argument("--out", type=str, default=os.path.join("outputs", "samples.png"),
                   help="Output grid PNG path")
    p.add_argument("--num-samples", type=int, default=64,
                   help="Number of samples in the grid (square root rounded for nrow)")
    p.add_argument("--nrow", type=int, default=8, help="Images per grid row")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--truncation", type=float, default=0.7,
                   help="Truncation psi in [0, 1]; lower = higher quality, less diversity")
    p.add_argument("--use-ema", action="store_true", default=True,
                   help="Use the EMA generator weights if present (default True)")
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    return p.parse_args()


def _build_generator_from_ckpt(ckpt: dict, device) -> Generator:
    """Re-create G with the architecture saved in the checkpoint config."""
    cfg = ckpt.get("config", {})
    return Generator(
        z_dim=cfg.get("NOISE_SIZE", config.NOISE_SIZE),
        w_dim=cfg.get("W_DIM", config.W_DIM),
        img_resolution=cfg.get("IMAGE_SIZE", config.IMAGE_SIZE),
        img_channels=cfg.get("NUM_CHANNELS", config.NUM_CHANNELS),
        channel_base=cfg.get("CHANNEL_BASE", config.CHANNEL_BASE),
        channel_max=cfg.get("CHANNEL_MAX", config.CHANNEL_MAX),
        mapping_layers=cfg.get("MAPPING_LAYERS", config.MAPPING_LAYERS),
    ).to(device)


def main():
    args = parse_args()
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    device = config.DEVICE
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device)
    G = _build_generator_from_ckpt(ckpt, device)

    if args.use_ema and "G_ema" in ckpt:
        G.load_state_dict(ckpt["G_ema"], strict=True)
        which = "G_ema"
    else:
        G.load_state_dict(ckpt["G"], strict=True)
        which = "G"
    G.eval()
    print(f"Loaded {which} from {args.ckpt} (step={ckpt.get('step', '?')})")

    z_dim = G.z_dim
    n = int(args.num_samples)

    # Truncation in W space: w_truncated = w_avg + psi * (w - w_avg).
    # Approximated here by sampling many z's, mapping them, averaging, then
    # blending — done once on the fly.
    if args.truncation < 1.0:
        with torch.no_grad():
            z_avg = torch.randn(10000, z_dim, device=device)
            w_avg = G.mapping(z_avg).mean(dim=0, keepdim=True)
            z = torch.randn(n, z_dim, device=device)
            w = G.mapping(z)
            w = w_avg + args.truncation * (w - w_avg)
            x = G.const.expand(n, -1, -1, -1)
            rgb = None
            for block in G.blocks:
                x, rgb = block(x, w, skip_rgb=rgb)
            imgs = torch.tanh(rgb)
    else:
        with torch.no_grad():
            z = torch.randn(n, z_dim, device=device)
            imgs = G(z)

    save_image_grid(imgs, args.out, nrow=args.nrow)
    print(f"Saved {n} samples -> {args.out}")


if __name__ == "__main__":
    main()
