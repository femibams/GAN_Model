import copy
import os
import time

# Must be set before any CUDA allocation to reduce fragmentation.
# max_split_size_mb=128 prevents the allocator from holding large cached blocks
# that can't satisfy subsequent 128 MiB allocation requests.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
import torch.optim as optim
from tqdm import tqdm

import config
from augment import AugmentPipe
from dataset import get_dataloader, precompute_clip_embeddings
from models import GeneratorWithMask, GlobalDiscriminator, RoIDiscriminator
from utils import (
    bboxes_to_mask, crop_roi_from_bboxes, d_hinge_loss, g_hinge_loss,
    leakage_loss_simple, leakage_loss_background, r1_gradient_penalty,
    path_length_penalty, save_or_show_images
)

def save_checkpoint(*, epoch, G, D_global, D_roi, g_optimizer, d_global_optimizer, d_roi_optimizer, out_path,
                    g_scheduler=None, d_global_scheduler=None, d_roi_scheduler=None,
                    fixed_noise=None, fixed_text=None, fixed_bboxes=None,
                    G_ema=None, augment_pipe=None):
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "config": {k: getattr(config, k) for k in dir(config) if k.isupper()},
        "G": G.state_dict(),
        "D_global": D_global.state_dict(),
        "D_roi": D_roi.state_dict(),
        "g_optimizer": g_optimizer.state_dict(),
        "d_global_optimizer": d_global_optimizer.state_dict(),
        "d_roi_optimizer": d_roi_optimizer.state_dict(),
    }
    if g_scheduler is not None:
        payload["g_scheduler"] = g_scheduler.state_dict()
    if d_global_scheduler is not None:
        payload["d_global_scheduler"] = d_global_scheduler.state_dict()
    if d_roi_scheduler is not None:
        payload["d_roi_scheduler"] = d_roi_scheduler.state_dict()
    if fixed_noise is not None:
        payload["fixed_noise"] = fixed_noise.cpu()
    if fixed_text is not None:
        payload["fixed_text"] = fixed_text.cpu()
    if fixed_bboxes is not None:
        payload["fixed_bboxes"] = fixed_bboxes.cpu()
    if G_ema is not None:
        payload["G_ema"] = G_ema.state_dict()
    if augment_pipe is not None:
        payload["augment_pipe_p"] = augment_pipe.p.item()
    torch.save(payload, out_path)

def load_checkpoint(*, ckpt_path, G, D_global, D_roi, g_optimizer, d_global_optimizer, d_roi_optimizer, device,
                    g_scheduler=None, d_global_scheduler=None, d_roi_scheduler=None,
                    G_ema=None, augment_pipe=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt["G"])
    D_global.load_state_dict(ckpt["D_global"])
    D_roi.load_state_dict(ckpt["D_roi"])
    g_optimizer.load_state_dict(ckpt["g_optimizer"])
    d_global_optimizer.load_state_dict(ckpt["d_global_optimizer"])
    d_roi_optimizer.load_state_dict(ckpt["d_roi_optimizer"])
    if g_scheduler is not None and "g_scheduler" in ckpt:
        g_scheduler.load_state_dict(ckpt["g_scheduler"])
    if d_global_scheduler is not None and "d_global_scheduler" in ckpt:
        d_global_scheduler.load_state_dict(ckpt["d_global_scheduler"])
    if d_roi_scheduler is not None and "d_roi_scheduler" in ckpt:
        d_roi_scheduler.load_state_dict(ckpt["d_roi_scheduler"])
    if G_ema is not None and "G_ema" in ckpt:
        G_ema.load_state_dict(ckpt["G_ema"])
    if augment_pipe is not None and "augment_pipe_p" in ckpt:
        augment_pipe.p.fill_(ckpt["augment_pipe_p"])
    start_epoch = int(ckpt.get("epoch", 0))
    return start_epoch


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


def train_step(real_imgs, text_embeddings, bboxes,
               G, D_global, D_roi,
               g_optimizer, d_global_optimizer, d_roi_optimizer,
               lambda_roi, lambda_leak, use_background_leak=True,
               lambda_r1=0.0, r1_interval=16, step=0,
               lambda_pl=0.0, pl_interval=4, pl_ema_ref=None,
               augment_pipe=None,
               *, scaler=None, use_amp=False):

    real_imgs = real_imgs.to(config.DEVICE, non_blocking=True)
    text_embeddings = text_embeddings.to(config.DEVICE, non_blocking=True)
    bboxes = bboxes.to(config.DEVICE, non_blocking=True)

    B = real_imgs.size(0)
    bbox_mask = bboxes_to_mask(bboxes, image_size=real_imgs.shape[-1], device=config.DEVICE)

    amp_enabled = bool(use_amp and config.DEVICE.type == "cuda")
    use_scaler = scaler is not None and amp_enabled

    # 1) Train Global Discriminator
    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
        z = torch.randn(B, config.NOISE_SIZE, device=config.DEVICE)
        fake_imgs = G(z, text_embeddings, bbox_mask).detach()

        # ADA: augment real and fake before D — prevents D overfitting to raw pixel stats.
        # R1 penalty (below) uses the original real_imgs, not the augmented version.
        if augment_pipe is not None:
            real_imgs_d = augment_pipe(real_imgs)
            fake_imgs_d = augment_pipe(fake_imgs)
        else:
            real_imgs_d, fake_imgs_d = real_imgs, fake_imgs

        real_logits_global = D_global(real_imgs_d, bbox_mask, text_embeddings)
        fake_logits_global = D_global(fake_imgs_d, bbox_mask, text_embeddings)
        d_global_loss = d_hinge_loss(real_logits_global, fake_logits_global)

    d_global_optimizer.zero_grad(set_to_none=True)
    if use_scaler:
        scaler.scale(d_global_loss).backward()
        scaler.step(d_global_optimizer)
    else:
        d_global_loss.backward()
        d_global_optimizer.step()

    # R1 gradient penalty for D_global (lazy: every r1_interval steps)
    # Computed in float32 with a separate forward pass — avoids fp16 precision issues
    # with second-order gradients. Scaled by r1_interval so effective weight stays LAMBDA_R1.
    if lambda_r1 > 0 and (step % r1_interval == 0):
        real_imgs_r1 = real_imgs.detach().float().requires_grad_(True)
        real_logits_r1 = D_global(real_imgs_r1, bbox_mask.float(),
                                  text_embeddings.float())
        r1_pen_global = r1_gradient_penalty(real_imgs_r1, real_logits_r1)
        # Multiply by r1_interval: lazy regularization computes every N steps,
        # so we scale up to maintain the same effective weight as every-step reg.
        r1_loss_global = (lambda_r1 * r1_interval / 2.0) * r1_pen_global
        d_global_optimizer.zero_grad(set_to_none=True)
        r1_loss_global.backward()
        d_global_optimizer.step()
    else:
        r1_pen_global = torch.zeros(1)

    # 2) Train ROI Discriminator
    # Crop from augmented images so D_roi and D_global see consistent inputs.
    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
        real_roi = crop_roi_from_bboxes(real_imgs_d, bboxes, out_size=config.ROI_SIZE)
        fake_roi = crop_roi_from_bboxes(fake_imgs_d, bboxes, out_size=config.ROI_SIZE)

        real_logits_roi = D_roi(real_roi)
        fake_logits_roi = D_roi(fake_roi)
        d_roi_loss = d_hinge_loss(real_logits_roi, fake_logits_roi)

    d_roi_optimizer.zero_grad(set_to_none=True)
    if use_scaler:
        scaler.scale(d_roi_loss).backward()
        scaler.step(d_roi_optimizer)
    else:
        d_roi_loss.backward()
        d_roi_optimizer.step()

    # R1 gradient penalty for D_roi
    if lambda_r1 > 0 and (step % r1_interval == 0):
        real_roi_r1 = real_roi.detach().float().requires_grad_(True)
        real_logits_roi_r1 = D_roi(real_roi_r1)
        r1_pen_roi = r1_gradient_penalty(real_roi_r1, real_logits_roi_r1)
        r1_loss_roi = (lambda_r1 * r1_interval / 2.0) * r1_pen_roi
        d_roi_optimizer.zero_grad(set_to_none=True)
        r1_loss_roi.backward()
        d_roi_optimizer.step()

    # 3) Train Generator
    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
        z = torch.randn(B, config.NOISE_SIZE, device=config.DEVICE)
        fake_imgs = G(z, text_embeddings, bbox_mask)

        # G must fool D even under augmentation — apply pipe to fakes here too.
        # Augmentation ops are differentiable so gradients still flow back to G.
        fake_imgs_g = augment_pipe(fake_imgs) if augment_pipe is not None else fake_imgs

        fake_logits_global = D_global(fake_imgs_g, bbox_mask, text_embeddings)
        fake_roi = crop_roi_from_bboxes(fake_imgs_g, bboxes, out_size=config.ROI_SIZE)
        fake_logits_roi = D_roi(fake_roi)

        g_adv_global = g_hinge_loss(fake_logits_global)
        g_adv_roi = g_hinge_loss(fake_logits_roi)

        if use_background_leak:
            g_leak = leakage_loss_background(fake_imgs, real_imgs, bbox_mask)
        else:
            g_leak = leakage_loss_simple(fake_imgs, bbox_mask)

        g_loss = g_adv_global + lambda_roi * g_adv_roi + lambda_leak * g_leak

    g_optimizer.zero_grad(set_to_none=True)
    if use_scaler:
        scaler.scale(g_loss).backward()
        scaler.unscale_(g_optimizer)
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        scaler.step(g_optimizer)
        scaler.update()
    else:
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        g_optimizer.step()

    # --- StyleGAN2 Path Length Regularization (lazy: every pl_interval steps) ---
    # Penalises changes in path length relative to a running EMA, encouraging
    # a fixed step in latent space to produce a fixed-magnitude pixel change.
    pl_pen_val = 0.0
    if lambda_pl > 0 and pl_ema_ref is not None and (step % pl_interval == 0):
        z_pl = torch.randn(B, config.NOISE_SIZE, device=config.DEVICE,
                           requires_grad=True)
        fake_pl = G(z_pl, text_embeddings, bbox_mask)
        pl_lengths = path_length_penalty(fake_pl, z_pl)   # [B]
        pl_mean = pl_lengths.mean().detach()

        # Warm-start EMA on first call, then exponential decay
        if pl_ema_ref[0] == 0.0:
            pl_ema_ref[0] = pl_mean.item()
        else:
            pl_ema_ref[0] = 0.99 * pl_ema_ref[0] + 0.01 * pl_mean.item()

        pl_penalty = (pl_lengths - pl_ema_ref[0]).pow(2).mean()
        # Same lazy-reg scaling as R1: multiply by interval to maintain
        # the same effective weight as if computed every step.
        pl_loss = lambda_pl * pl_interval * pl_penalty

        g_optimizer.zero_grad(set_to_none=True)
        pl_loss.backward()
        g_optimizer.step()
        pl_pen_val = float(pl_penalty.item())

    return {
        "d_global_loss": float(d_global_loss.item()),
        "d_roi_loss": float(d_roi_loss.item()),
        "g_loss": float(g_loss.item()),
        "r1_penalty": float(r1_pen_global.item()),
        "pl_penalty": pl_pen_val,
        # ADA overfitting signal: mean(sign(D(real))).  Ranges in [-1, 1].
        # Values near +1 mean D is very confident on reals → augment more.
        "real_sign": float(real_logits_global.detach().sign().mean().item()),
    }

def main():
    print(f"Using device: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
    else:
        print("CUDA is unavailable in this environment; training runs on CPU.")
    clip_model, clip_tokenizer = build_clip_text_encoder(config.DEVICE)

    # Pre-compute all CLIP embeddings once (cached to disk after first run).
    # This eliminates per-batch CLIP inference in the training loop.
    clip_embeddings = None
    if clip_model is not None:
        clip_embeddings = precompute_clip_embeddings(
            clip_model=clip_model,
            tokenizer=clip_tokenizer,
            attr_file=getattr(config, "ATTR_FILE", None),
            bbox_file=config.BBOX_FILE,
            device=config.DEVICE,
            embedding_size=config.EMBEDDING_SIZE,
        )

    dataloader = get_dataloader(return_prompt=True, clip_embeddings=clip_embeddings)

    # Init Models
    G = GeneratorWithMask(
        noise_size=config.NOISE_SIZE, feature_size=config.FEATURE_SIZE, 
        num_channels=config.NUM_CHANNELS, embedding_size=config.EMBEDDING_SIZE, 
        reduced_dim_size=config.REDUCED_DIM_SIZE
    ).to(config.DEVICE)

    # D_TEXT_DIM=0 disables projection conditioning — the inner product term
    # amplifies R1 gradients ~500× at init causing immediate NaN. Re-enable
    # only after confirming stable training.
    d_text_dim = getattr(config, "D_TEXT_DIM", 0)
    D_global = GlobalDiscriminator(
        num_channels=config.NUM_CHANNELS, text_dim=d_text_dim
    ).to(config.DEVICE)
    D_roi = RoIDiscriminator(num_channels=config.NUM_CHANNELS).to(config.DEVICE)

    # EMA copy of G — updated every step, used for all visualisation and eval.
    # Produces visibly smoother outputs than the live training weights.
    G_ema = copy.deepcopy(G).eval().to(config.DEVICE)

    # ADA augmentation pipe — p starts at 0 (no augmentation) and is adapted
    # automatically based on discriminator overfitting.
    augment_pipe = AugmentPipe(
        brightness=1.0, brightness_std=0.2,
        contrast=1.0,   contrast_range=(0.5, 2.0),
        cutout=0.5,     cutout_fraction=0.5,
    ).to(config.DEVICE)

    # Init Optimizers
    lr_d = getattr(config, "LR_D", config.LR)
    g_optimizer = optim.Adam(G.parameters(), lr=config.LR, betas=config.BETAS)
    d_global_optimizer = optim.Adam(D_global.parameters(), lr=lr_d, betas=config.BETAS)
    d_roi_optimizer = optim.Adam(D_roi.parameters(), lr=lr_d, betas=config.BETAS)

    # LR schedulers: linear decay from LR_DECAY_START epoch → 0 at NUM_EPOCHS
    lr_decay_start = getattr(config, "LR_DECAY_START", 150)
    def lr_lambda(epoch):
        if epoch < lr_decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - lr_decay_start) / max(config.NUM_EPOCHS - lr_decay_start, 1))
    g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda)
    d_global_scheduler = optim.lr_scheduler.LambdaLR(d_global_optimizer, lr_lambda)
    d_roi_scheduler = optim.lr_scheduler.LambdaLR(d_roi_optimizer, lr_lambda)

    ckpt_path = os.path.join("outputs", "checkpoints", "latest.pth")
    start_epoch = 0  # epoch index to start from (0-based)
    if os.path.exists(ckpt_path):
        try:
            # ckpt stores the last completed epoch number (1-based); next epoch index is that number
            start_epoch = load_checkpoint(
                ckpt_path=ckpt_path,
                G=G,
                D_global=D_global,
                D_roi=D_roi,
                g_optimizer=g_optimizer,
                d_global_optimizer=d_global_optimizer,
                d_roi_optimizer=d_roi_optimizer,
                g_scheduler=g_scheduler,
                d_global_scheduler=d_global_scheduler,
                d_roi_scheduler=d_roi_scheduler,
                device=config.DEVICE,
                G_ema=G_ema,
                augment_pipe=augment_pipe,
            )
            if start_epoch >= config.NUM_EPOCHS:
                print(
                    f"Checkpoint epoch ({start_epoch}) is already >= NUM_EPOCHS ({config.NUM_EPOCHS}). "
                    "No additional epochs to train."
                )
                return
            print(f"Resuming from checkpoint `{ckpt_path}` at epoch {start_epoch + 1}/{config.NUM_EPOCHS}...")
            # Fast-forward schedulers if checkpoint didn't have them saved
            if "g_scheduler" not in torch.load(ckpt_path, map_location="cpu"):
                for _ in range(start_epoch):
                    g_scheduler.step()
                    d_global_scheduler.step()
                    d_roi_scheduler.step()
        except Exception as e:
            print(f"Warning: failed to load checkpoint at `{ckpt_path}`. Starting fresh. Error: {e}")
            start_epoch = 0

    # Setup Fixed Noise for consistent eval tracking
    fixed_noise = torch.randn(16, config.NOISE_SIZE, device=config.DEVICE)
    fixed_prompts = [getattr(config, "DEFAULT_PROMPT", "a portrait photo of a person")] * 16
    fixed_text = encode_text_prompts(fixed_prompts, clip_model, clip_tokenizer, config.DEVICE)
    fixed_bboxes = torch.tensor([[0.30, 0.20, 0.70, 0.80]] * 16, dtype=torch.float32, device=config.DEVICE)
    fixed_mask = bboxes_to_mask(fixed_bboxes, image_size=config.IMAGE_SIZE, device=config.DEVICE)

    use_amp = bool(getattr(config, "USE_AMP", False)) and config.DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Path length EMA state — a single float tracked across all batches
    pl_ema_ref = [0.0]

    # Create outputs directory and initialize results file
    os.makedirs(os.path.dirname(config.RESULT_FILE), exist_ok=True)
    if start_epoch == 0 or not os.path.exists(config.RESULT_FILE):
        with open(config.RESULT_FILE, "w") as f:
            f.write("Epoch,D_global_loss,D_roi_loss,G_loss,Time_s\n")

    print("Starting Training Loop...")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # --- NEW: Record the start time of the epoch ---
        epoch_start_time = time.time()
        
        G.train()
        D_global.train()
        D_roi.train()

        epoch_stats = {"d_global": 0, "d_roi": 0, "g_loss": 0}
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")

        # Gradual ROI warm-up: linear ramp from ROI_WARMUP_START → ROI_WARMUP_END
        roi_warmup_start = getattr(config, "ROI_WARMUP_START", 10)
        roi_warmup_end = getattr(config, "ROI_WARMUP_END", 30)
        if epoch < roi_warmup_start:
            current_lambda_roi = 0.0
        elif epoch < roi_warmup_end:
            current_lambda_roi = config.LAMBDA_ROI * (epoch - roi_warmup_start) / (roi_warmup_end - roi_warmup_start)
        else:
            current_lambda_roi = config.LAMBDA_ROI

        lambda_r1 = getattr(config, "LAMBDA_R1", 0.0)
        r1_interval = getattr(config, "R1_INTERVAL", 16)
        use_background_leak = getattr(config, "USE_BACKGROUND_LEAK", False)
        lambda_pl = getattr(config, "LAMBDA_PL", 2.0)
        pl_interval = getattr(config, "PL_INTERVAL", 4)
        ada_target   = getattr(config, "ADA_TARGET",   0.6)
        ada_interval = getattr(config, "ADA_INTERVAL", 4)
        ada_kimg     = getattr(config, "ADA_KIMG",     500.0)
        ema_kimg     = getattr(config, "EMA_KIMG",     10.0)
        ema_nimg     = ema_kimg * 1000
        ema_beta     = 0.5 ** (config.BATCH_SIZE / max(ema_nimg, 1e-8))

        # Accumulator for ADA overfitting signal
        ada_sign_acc   = 0.0
        ada_sign_count = 0

        for i, (real_imgs, text_prompts, bboxes) in enumerate(progress_bar):
            global_step = epoch * len(dataloader) + i
            # If the dataset returned pre-computed tensors, move them to device directly.
            # Otherwise fall back to on-the-fly CLIP encoding (slower).
            if isinstance(text_prompts, torch.Tensor):
                text_embeddings = text_prompts.to(config.DEVICE, non_blocking=True)
            else:
                text_embeddings = encode_text_prompts(text_prompts, clip_model, clip_tokenizer, config.DEVICE)

            stats = train_step(
                real_imgs, text_embeddings, bboxes,
                G, D_global, D_roi,
                g_optimizer, d_global_optimizer, d_roi_optimizer,
                lambda_roi=current_lambda_roi, lambda_leak=config.LAMBDA_LEAK,
                use_background_leak=use_background_leak,
                lambda_r1=lambda_r1, r1_interval=r1_interval, step=global_step,
                lambda_pl=lambda_pl, pl_interval=pl_interval, pl_ema_ref=pl_ema_ref,
                augment_pipe=augment_pipe,
                scaler=scaler, use_amp=use_amp,
            )

            # --- EMA update (every step) ---
            with torch.no_grad():
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                    b_ema.copy_(b)

            # --- ADA p update (every ada_interval steps) ---
            ada_sign_acc   += stats["real_sign"]
            ada_sign_count += 1
            if (global_step + 1) % ada_interval == 0:
                augment_pipe.update_p(
                    real_sign_mean=ada_sign_acc / ada_sign_count,
                    batch_size=config.BATCH_SIZE,
                    interval=ada_interval,
                    ada_kimg=ada_kimg,
                    ada_target=ada_target,
                )
                ada_sign_acc   = 0.0
                ada_sign_count = 0

            epoch_stats["d_global"] += stats["d_global_loss"]
            epoch_stats["d_roi"] += stats["d_roi_loss"]
            epoch_stats["g_loss"] += stats["g_loss"]

            progress_bar.set_postfix({
                "Dg": f"{stats['d_global_loss']:.3f}",
                "Dr": f"{stats['d_roi_loss']:.3f}",
                "G": f"{stats['g_loss']:.3f}",
                "R1": f"{stats['r1_penalty']:.3f}",
                "PL": f"{stats['pl_penalty']:.3f}",
                "p":  f"{augment_pipe.p.item():.3f}",
            })

        # --- NEW: Calculate total time taken for the epoch ---
        epoch_duration = time.time() - epoch_start_time

        num_batches = len(dataloader)
        avg_dg = epoch_stats['d_global'] / num_batches
        avg_dr = epoch_stats['d_roi'] / num_batches
        avg_g = epoch_stats['g_loss'] / num_batches
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  D_global: {avg_dg:.4f}")
        print(f"  D_roi   : {avg_dr:.4f}")
        print(f"  G_loss  : {avg_g:.4f}")
        print(f"  LR_G    : {g_scheduler.get_last_lr()[0]:.2e}")
        print(f"  ADA p   : {augment_pipe.p.item():.4f}")
        print(f"  Time    : {epoch_duration:.2f} seconds")

        with open(config.RESULT_FILE, "a") as f:
            f.write(f"{epoch+1},{avg_dg:.4f},{avg_dr:.4f},{avg_g:.4f},{epoch_duration:.2f}\n")

        # Visualise with G_ema — its averaged weights give cleaner samples than live G
        with torch.no_grad():
            fake_imgs = G_ema(fixed_noise, fixed_text, fixed_mask)

        save_or_show_images(fake_imgs, title=f"Epoch {epoch+1} (EMA)", save_path=f"outputs/epoch_{epoch+1}.png")

        # Step LR schedulers at end of each epoch
        g_scheduler.step()
        d_global_scheduler.step()
        d_roi_scheduler.step()

        # Save checkpoint after each epoch
        save_checkpoint(
            epoch=epoch + 1,
            G=G,
            D_global=D_global,
            D_roi=D_roi,
            g_optimizer=g_optimizer,
            d_global_optimizer=d_global_optimizer,
            d_roi_optimizer=d_roi_optimizer,
            g_scheduler=g_scheduler,
            d_global_scheduler=d_global_scheduler,
            d_roi_scheduler=d_roi_scheduler,
            out_path=os.path.join("outputs", "checkpoints", "latest.pth"),
            G_ema=G_ema,
            augment_pipe=augment_pipe,
        )

if __name__ == "__main__":
    main()