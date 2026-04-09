import argparse
import csv
import os
from datetime import datetime

import torch
from tqdm import tqdm

import config
from dataset import get_dataloader
from models import GeneratorWithMask
from utils import bboxes_to_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GAN outputs with FID, CLIP score, KID, and IS.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=getattr(config, "EVAL_CKPT_PATH", os.path.join("outputs", "checkpoints", "latest.pth")),
        help="Path to training checkpoint.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(getattr(config, "EVAL_MAX_SAMPLES", 2048)),
        help="Maximum number of real/fake images to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Evaluation batch size used for generation/metric updates.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=getattr(config, "EVAL_OUTPUT_CSV", os.path.join("outputs", "eval_results.csv")),
        help="CSV file to append evaluation metrics.",
    )
    parser.add_argument(
        "--clip-prompt",
        type=str,
        default=getattr(config, "DEFAULT_PROMPT", "a portrait photo of a person"),
        help="Prompt used for CLIP score.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default=getattr(config, "EVAL_CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"),
        help="CLIP model name for torchmetrics CLIPScore.",
    )
    parser.add_argument(
        "--truncation",
        type=float,
        default=0.7,
        help="Truncation psi for noise sampling (0 < psi <= 1). Lower = better quality, less diversity.",
    )
    return parser.parse_args()


def to_uint8_for_inception(images):
    # Input expected in [-1, 1]. Convert to uint8 [0, 255] for inception-based metrics.
    images_01 = ((images + 1.0) / 2.0).clamp(0, 1)
    return (images_01 * 255.0).to(torch.uint8)


def load_generator(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if "G" not in ckpt:
        raise KeyError("Checkpoint missing 'G' state dict. Is this from train.py?")

    model = GeneratorWithMask(
        noise_size=config.NOISE_SIZE,
        feature_size=config.FEATURE_SIZE,
        num_channels=config.NUM_CHANNELS,
        embedding_size=config.EMBEDDING_SIZE,
        reduced_dim_size=config.REDUCED_DIM_SIZE,
    ).to(device)
    model.load_state_dict(ckpt["G"], strict=True)
    model.eval()
    return model


def build_metrics(device, clip_model):
    metrics = {}
    warnings = []

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        metrics["fid"] = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    except Exception as exc:
        warnings.append(f"FID disabled: {exc}")

    try:
        from torchmetrics.image.kid import KernelInceptionDistance

        metrics["kid"] = KernelInceptionDistance(
            subset_size=100,
            normalize=False,
        ).to(device)
    except Exception as exc:
        warnings.append(f"KID disabled: {exc}")

    try:
        from torchmetrics.image.inception import InceptionScore

        metrics["is"] = InceptionScore(normalize=False).to(device)
    except Exception as exc:
        warnings.append(f"Inception Score disabled: {exc}")

    try:
        from torchmetrics.multimodal.clip_score import CLIPScore

        metrics["clip_fake"] = CLIPScore(model_name_or_path=clip_model).to(device)
        metrics["clip_real"] = CLIPScore(model_name_or_path=clip_model).to(device)
    except Exception as exc:
        warnings.append(f"CLIP score disabled: {exc}")

    return metrics, warnings


def build_clip_text_encoder(device):
    if not bool(getattr(config, "USE_CLIP_TEXT", True)):
        return None, None
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "USE_CLIP_TEXT=True requires open-clip-torch. "
            "Install with: pip install -r requirements.txt"
        ) from exc
    model_name = getattr(config, "TRAIN_CLIP_MODEL_NAME", "ViT-B-32")
    pretrained = getattr(config, "TRAIN_CLIP_PRETRAINED", "openai")
    clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(model_name)
    return clip_model, tokenizer


@torch.no_grad()
def encode_text_prompts(prompts, clip_model, tokenizer, device):
    if clip_model is None or tokenizer is None:
        bsz = len(prompts)
        return torch.zeros(bsz, config.EMBEDDING_SIZE, dtype=torch.float32, device=device)
    tokens = tokenizer(list(prompts)).to(device)
    text_embeddings = clip_model.encode_text(tokens).float()
    if text_embeddings.shape[-1] != config.EMBEDDING_SIZE:
        raise ValueError(
            f"CLIP embedding dim ({text_embeddings.shape[-1]}) does not match "
            f"config.EMBEDDING_SIZE ({config.EMBEDDING_SIZE})."
        )
    return text_embeddings


@torch.no_grad()
def run_eval(args):
    device = config.DEVICE
    if device.type != "cuda":
        print("Warning: CUDA not available; evaluation can be slow on CPU.")

    G = load_generator(args.ckpt, device)
    dataloader = get_dataloader(return_prompt=True, batch_size=args.batch_size, shuffle=False)
    clip_model, clip_tokenizer = build_clip_text_encoder(device)
    metrics, warnings = build_metrics(device, args.clip_model)

    if warnings:
        print("Metric availability warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if not metrics:
        raise RuntimeError(
            "No metrics could be initialized. Install optional deps from requirements.txt "
            "and retry."
        )

    target_samples = int(args.max_samples)
    processed = 0
    progress = tqdm(total=target_samples, desc="Evaluating", unit="img")

    for real_imgs, text_prompts, bboxes in dataloader:
        if processed >= target_samples:
            break

        take = min(real_imgs.size(0), target_samples - processed)
        real_imgs = real_imgs[:take].to(device, non_blocking=True)
        text_embeddings = encode_text_prompts(text_prompts[:take], clip_model, clip_tokenizer, device)
        bboxes = bboxes[:take].to(device, non_blocking=True)

        if args.truncation < 1.0:
            z = torch.zeros(take, config.NOISE_SIZE, device=device)
            remaining = torch.ones(take, dtype=torch.bool, device=device)
            threshold = args.truncation * (config.NOISE_SIZE ** 0.5)
            for _ in range(100):
                candidate = torch.randn(take, config.NOISE_SIZE, device=device)
                accepted = candidate.norm(dim=1) <= threshold
                fill = remaining & accepted
                z[fill] = candidate[fill]
                remaining = remaining & ~accepted
                if not remaining.any():
                    break
            if remaining.any():
                z[remaining] = torch.randn(remaining.sum(), config.NOISE_SIZE, device=device) * args.truncation
        else:
            z = torch.randn(take, config.NOISE_SIZE, device=device)
        bbox_mask = bboxes_to_mask(bboxes, image_size=config.IMAGE_SIZE, device=device)
        fake_imgs = G(z, text_embeddings, bbox_mask)

        real_u8 = to_uint8_for_inception(real_imgs)
        fake_u8 = to_uint8_for_inception(fake_imgs)

        if "fid" in metrics:
            metrics["fid"].update(real_u8, real=True)
            metrics["fid"].update(fake_u8, real=False)

        if "kid" in metrics:
            metrics["kid"].update(real_u8, real=True)
            metrics["kid"].update(fake_u8, real=False)

        if "is" in metrics:
            metrics["is"].update(fake_u8)

        if "clip_fake" in metrics and "clip_real" in metrics:
            prompt_batch = [args.clip_prompt] * take
            fake_01 = ((fake_imgs + 1.0) / 2.0).clamp(0, 1)
            real_01 = ((real_imgs + 1.0) / 2.0).clamp(0, 1)
            metrics["clip_fake"].update(fake_01, prompt_batch)
            metrics["clip_real"].update(real_01, prompt_batch)

        processed += take
        progress.update(take)

    progress.close()

    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ckpt": args.ckpt,
        "samples": processed,
    }

    if "fid" in metrics:
        results["fid"] = float(metrics["fid"].compute().item())

    if "kid" in metrics:
        kid_mean, kid_std = metrics["kid"].compute()
        results["kid_mean"] = float(kid_mean.item())
        results["kid_std"] = float(kid_std.item())

    if "is" in metrics:
        is_mean, is_std = metrics["is"].compute()
        results["inception_score_mean"] = float(is_mean.item())
        results["inception_score_std"] = float(is_std.item())

    if "clip_fake" in metrics and "clip_real" in metrics:
        clip_fake = float(metrics["clip_fake"].compute().item())
        clip_real = float(metrics["clip_real"].compute().item())
        results["clip_score_fake"] = clip_fake
        results["clip_score_real"] = clip_real
        results["clip_score_gap_real_minus_fake"] = clip_real - clip_fake

    return results


def append_results_csv(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path)

    headers = [
        "timestamp",
        "ckpt",
        "samples",
        "fid",
        "kid_mean",
        "kid_std",
        "inception_score_mean",
        "inception_score_std",
        "clip_score_fake",
        "clip_score_real",
        "clip_score_gap_real_minus_fake",
    ]

    with open(path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in headers})


def main():
    args = parse_args()
    results = run_eval(args)

    print("\nEvaluation Results")
    for key, value in results.items():
        print(f"{key}: {value}")

    append_results_csv(args.out_csv, results)
    print(f"\nSaved metrics to: {args.out_csv}")


if __name__ == "__main__":
    main()
