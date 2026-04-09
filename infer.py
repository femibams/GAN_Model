import argparse
import os

import torch

import config
from models import GeneratorWithMask
from utils import bboxes_to_mask, save_or_show_images


def get_clip_text_embeddings(prompt, num_samples, device, expected_dim, clip_model_name, clip_pretrained):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "open-clip-torch is required for text-conditioned inference. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    clip_model, _, _ = open_clip.create_model_and_transforms(
        clip_model_name,
        pretrained=clip_pretrained,
        device=device,
    )
    clip_model.eval()

    with torch.no_grad():
        tokens = open_clip.tokenize([prompt]).to(device)
        text_features = clip_model.encode_text(tokens).float()

    if text_features.shape[-1] != expected_dim:
        raise ValueError(
            f"CLIP embedding dim ({text_features.shape[-1]}) does not match "
            f"config.EMBEDDING_SIZE ({expected_dim})."
        )

    return text_features.repeat(num_samples, 1)


def parse_args():
    p = argparse.ArgumentParser(description="GAN inference: generate an image from a bbox mask")
    p.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join("outputs", "checkpoints", "latest.pth"),
        help="Path to checkpoint created by train.py",
    )
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join("outputs", "inference.png"),
        help="Output image path",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of samples to generate (rendered as a grid; first 16 are shown)",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    p.add_argument(
        "--truncation",
        type=float,
        default=0.7,
        help="Truncation psi for noise sampling (0 < psi <= 1). Lower = better quality, less diversity.",
    )

    # bbox coords are normalized to [0, 1], matching the training code.
    p.add_argument("--x1", type=float, default=0.30)
    p.add_argument("--y1", type=float, default=0.20)
    p.add_argument("--x2", type=float, default=0.70)
    p.add_argument("--y2", type=float, default=0.80)
    p.add_argument(
        "--prompt",
        type=str,
        default=getattr(config, "DEFAULT_PROMPT", "a portrait photo of a person"),
        help="Text prompt used to create CLIP text embedding for conditioning.",
    )
    p.add_argument(
        "--clip-model",
        type=str,
        default=getattr(config, "INFER_CLIP_MODEL_NAME", "ViT-B-32"),
        help="OpenCLIP model name used for text embedding.",
    )
    p.add_argument(
        "--clip-pretrained",
        type=str,
        default=getattr(config, "INFER_CLIP_PRETRAINED", "openai"),
        help="OpenCLIP pretrained tag (e.g., openai, laion2b_s34b_b79k).",
    )

    p.add_argument("--save-only", action="store_true", help="Do not display the image (always saves)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    device = config.DEVICE
    if device.type != "cuda":
        print("Warning: CUDA not available; running on CPU (will be slow).")

    # Deterministic-ish inference: seed both CPU and CUDA RNGs.
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device)
    if "G" not in ckpt:
        raise KeyError("Checkpoint missing 'G' state dict. Is this from train.py?")

    # Recreate generator with the same architecture/hyperparams as training.
    G = GeneratorWithMask(
        noise_size=config.NOISE_SIZE,
        feature_size=config.FEATURE_SIZE,
        num_channels=config.NUM_CHANNELS,
        embedding_size=config.EMBEDDING_SIZE,
        reduced_dim_size=config.REDUCED_DIM_SIZE,
    ).to(device)
    G.load_state_dict(ckpt["G"], strict=True)
    G.eval()

    # Build bbox tensors/mask. Shape must be (B, 4).
    b = int(args.num_samples)
    bboxes = torch.tensor([[args.x1, args.y1, args.x2, args.y2]] * b, dtype=torch.float32, device=device)
    bbox_mask = bboxes_to_mask(bboxes, image_size=config.IMAGE_SIZE, device=device)

    # Use real text conditioning from CLIP instead of placeholder embeddings.
    text_embeddings = get_clip_text_embeddings(
        prompt=args.prompt,
        num_samples=b,
        device=device,
        expected_dim=config.EMBEDDING_SIZE,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )
    if args.truncation < 1.0:
        # Truncation trick: reject noise vectors with norm above threshold, keeping higher-quality region
        noise = torch.zeros(b, config.NOISE_SIZE, dtype=torch.float32, device=device)
        remaining = torch.ones(b, dtype=torch.bool, device=device)
        threshold = args.truncation * (config.NOISE_SIZE ** 0.5)
        for _ in range(100):
            candidate = torch.randn(b, config.NOISE_SIZE, dtype=torch.float32, device=device)
            accepted = candidate.norm(dim=1) <= threshold
            fill = remaining & accepted
            noise[fill] = candidate[fill]
            remaining = remaining & ~accepted
            if not remaining.any():
                break
        # Fill any remaining with scaled-down noise
        if remaining.any():
            noise[remaining] = torch.randn(remaining.sum(), config.NOISE_SIZE, dtype=torch.float32, device=device) * args.truncation
    else:
        noise = torch.randn(b, config.NOISE_SIZE, dtype=torch.float32, device=device)

    with torch.no_grad():
        fake_imgs = G(noise, text_embeddings, bbox_mask)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_or_show_images(fake_imgs, title="Inference", save_path=args.out if not args.save_only else args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

