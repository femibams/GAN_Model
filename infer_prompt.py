"""Prompt-guided inference via CLIP-guided latent optimization.

The trained generator is unconditional. To bias generation toward a text prompt
we keep G frozen and optimize a single style vector w in W-space so that the
generated image's CLIP embedding matches the prompt's CLIP embedding.

Quality ceiling = whatever the generator can already produce. Prompts that
land outside the learned face distribution (e.g. "astronaut") will not work.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image

import config
from models import Generator
from infer import _build_generator_from_ckpt, _save_single_image

try:
    import open_clip
except ImportError as e:
    raise SystemExit(
        "open_clip_torch is required for prompt-guided inference.\n"
        "Install it with:  pip install open_clip_torch"
    ) from e


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_RES = 224


def parse_args():
    p = argparse.ArgumentParser(description="Generate a face matching a text prompt")
    p.add_argument("--prompt", type=str, required=True,
                   help="Text prompt to steer generation, e.g. 'a smiling woman with red hair'")
    p.add_argument("--ckpt", type=str,
                   default=os.path.join("outputs", "checkpoints", "latest.pth"),
                   help="Path to checkpoint produced by train.py")
    p.add_argument("--out", type=str, default=os.path.join("outputs", "sample_prompt.png"),
                   help="Output image PNG path")
    p.add_argument("--steps", type=int, default=200,
                   help="Number of optimization steps")
    p.add_argument("--lr", type=float, default=0.05,
                   help="Adam learning rate for the latent w")
    p.add_argument("--truncation", type=float, default=0.7,
                   help="Initial truncation psi for w; lower = stay closer to the average face")
    p.add_argument("--w-reg", type=float, default=0.05,
                   help="Strength of L2 regularizer pulling w toward w_avg (prevents drift off-manifold)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed; if omitted, a fresh random seed is used each run")
    p.add_argument("--clip-model", type=str, default="ViT-B-16",
                   help="open_clip model name")
    p.add_argument("--clip-pretrained", type=str, default="openai",
                   help="open_clip pretrained tag (run `python -c \"import open_clip; "
                        "print(open_clip.list_pretrained())\"` to see what's available "
                        "for your installed version)")
    p.add_argument("--use-ema", action="store_true", default=True,
                   help="Use the EMA generator weights if present (default True)")
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    p.add_argument("--log-every", type=int, default=25,
                   help="Print loss every N steps (0 to disable)")
    return p.parse_args()


def _synthesize(G: Generator, w: torch.Tensor) -> torch.Tensor:
    """Run G's synthesis blocks for a given w (skipping the mapping network)."""
    x = G.const.expand(w.size(0), -1, -1, -1)
    rgb = None
    for block in G.blocks:
        x, rgb = block(x, w, skip_rgb=rgb)
    return torch.tanh(rgb)


def _to_clip_input(img: torch.Tensor, device) -> torch.Tensor:
    """Convert generator output in [-1, 1] to CLIP-normalized 224x224 input."""
    img01 = (img.clamp(-1, 1) + 1) / 2.0
    img01 = F.interpolate(img01, size=CLIP_RES, mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    return (img01 - mean) / std


def main():
    args = parse_args()
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    device = config.DEVICE
    seed = args.seed if args.seed is not None else torch.seed()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")

    # ---- Generator ---------------------------------------------------------
    ckpt = torch.load(args.ckpt, map_location=device)
    G = _build_generator_from_ckpt(ckpt, device)
    state_key = "G_ema" if (args.use_ema and "G_ema" in ckpt) else "G"
    G.load_state_dict(ckpt[state_key], strict=True)
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    print(f"Loaded {state_key} from {args.ckpt} (step={ckpt.get('step', '?')})")

    # ---- CLIP --------------------------------------------------------------
    print(f"Loading CLIP {args.clip_model} / {args.clip_pretrained}...")
    try:
        clip_model, _, _ = open_clip.create_model_and_transforms(
            args.clip_model, pretrained=args.clip_pretrained, device=device
        )
    except RuntimeError as e:
        available = [t for m, t in open_clip.list_pretrained() if m == args.clip_model]
        raise SystemExit(
            f"{e}\n\nAvailable pretrained tags for {args.clip_model} in this open_clip version:\n"
            f"  {available}\n"
            f"Pass one with --clip-pretrained, or upgrade open_clip_torch."
        ) from e
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    with torch.no_grad():
        tokens = tokenizer([args.prompt]).to(device)
        text_feat = clip_model.encode_text(tokens).float()
        text_feat = F.normalize(text_feat, dim=-1)

    # ---- Initialize w in the truncated region ------------------------------
    with torch.no_grad():
        z_avg = torch.randn(10000, G.z_dim, device=device)
        w_avg = G.mapping(z_avg).mean(dim=0, keepdim=True)
        z0 = torch.randn(1, G.z_dim, device=device)
        w0 = G.mapping(z0)
        w_init = w_avg + args.truncation * (w0 - w_avg)
    w = w_init.detach().clone().requires_grad_(True)
    w_avg_const = w_avg.detach()

    # ---- Optimize ----------------------------------------------------------
    opt = torch.optim.Adam([w], lr=args.lr)
    for step in range(1, args.steps + 1):
        img = _synthesize(G, w)
        clip_in = _to_clip_input(img, device)
        img_feat = clip_model.encode_image(clip_in).float()
        img_feat = F.normalize(img_feat, dim=-1)

        clip_loss = 1.0 - (img_feat * text_feat).sum(dim=-1).mean()
        reg = ((w - w_avg_const) ** 2).mean()
        loss = clip_loss + args.w_reg * reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if args.log_every and (step == 1 or step % args.log_every == 0 or step == args.steps):
            print(f"  step {step:4d}/{args.steps}  clip={clip_loss.item():.4f}  reg={reg.item():.4f}")

    # ---- Final image -------------------------------------------------------
    with torch.no_grad():
        final = _synthesize(G, w)
    _save_single_image(final[0], args.out)
    print(f"Saved sample -> {args.out}")


if __name__ == "__main__":
    main()
