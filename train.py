"""Iteration-based training loop for the small-scale StyleGAN2 face GAN.

Key features
  - Non-saturating logistic GAN loss (StyleGAN2 default)
  - R1 gradient penalty on D (lazy, every R1_INTERVAL iterations)
  - Path-length regularization on G (lazy, every PL_INTERVAL iterations)
  - Mixed-precision forward passes via torch.amp.autocast + GradScaler
  - Optional gradient accumulation (GRAD_ACCUM_STEPS in config)
  - Generator EMA, used for sample grids and saved alongside G
  - Periodic sample grid saves and checkpoints with full resume support
"""
import argparse
import copy
import math
import os
import time

# Set before any CUDA allocation to reduce fragmentation under AMP
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
import torch.optim as optim

import config
from dataset import get_dataloader, infinite_loader
from models import Generator, Discriminator
from utils import (
    d_logistic_loss, g_nonsaturating_loss,
    r1_gradient_penalty, path_length_lengths,
    save_image_grid,
)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def _config_snapshot() -> dict:
    return {k: getattr(config, k) for k in dir(config) if k.isupper()}


def save_checkpoint(path, *, step, G, D, G_ema, g_opt, d_opt, scaler,
                    pl_ema, fixed_z):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "step": int(step),
        "config": _config_snapshot(),
        "G": G.state_dict(),
        "D": D.state_dict(),
        "G_ema": G_ema.state_dict(),
        "g_opt": g_opt.state_dict(),
        "d_opt": d_opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "pl_ema": float(pl_ema),
        "fixed_z": fixed_z.detach().cpu(),
    }, path)


def load_checkpoint(path, *, G, D, G_ema, g_opt, d_opt, scaler, device):
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    G_ema.load_state_dict(ckpt["G_ema"])
    g_opt.load_state_dict(ckpt["g_opt"])
    d_opt.load_state_dict(ckpt["d_opt"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def _format_eta(seconds: float) -> str:
    """Format a duration as Hh:MMm or MMm:SSs — short, monotonic, easy to scan."""
    if seconds <= 0:
        return "  -  "
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:>2d}h{m:02d}m"
    return f"{m:>2d}m{s:02d}s"


# ---------------------------------------------------------------------------
# Generator EMA
# ---------------------------------------------------------------------------
def update_ema(G_ema, G, beta):
    with torch.no_grad():
        for p_ema, p in zip(G_ema.parameters(), G.parameters()):
            p_ema.copy_(p.lerp(p_ema, beta))
        for b_ema, b in zip(G_ema.buffers(), G.buffers()):
            b_ema.copy_(b)


# ---------------------------------------------------------------------------
# Training step components
# ---------------------------------------------------------------------------
def d_step(real_imgs, *, G, D, d_opt, scaler, use_amp,
           lambda_r1, r1_interval, step):
    """One discriminator step: hinge-style logistic loss, with lazy R1 regulariser."""
    z = torch.randn(real_imgs.size(0), config.NOISE_SIZE, device=real_imgs.device)

    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        with torch.no_grad():
            fake_imgs = G(z)
        real_logits = D(real_imgs)
        fake_logits = D(fake_imgs)
        d_loss = d_logistic_loss(real_logits, fake_logits)

    d_opt.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(d_loss).backward()
        scaler.step(d_opt)
    else:
        d_loss.backward()
        d_opt.step()

    r1_val = 0.0
    if lambda_r1 > 0 and (step % r1_interval == 0):
        # R1 in fp32 with a fresh forward — second-order grads are precision-sensitive
        real_r1 = real_imgs.detach().float().requires_grad_(True)
        real_logits_r1 = D(real_r1)
        r1 = r1_gradient_penalty(real_r1, real_logits_r1)
        r1_loss = (lambda_r1 * r1_interval / 2.0) * r1
        d_opt.zero_grad(set_to_none=True)
        r1_loss.backward()
        d_opt.step()
        r1_val = float(r1.detach().item())

    return {
        "d_loss": float(d_loss.detach().item()),
        "real_logit": float(real_logits.detach().mean().item()),
        "fake_logit": float(fake_logits.detach().mean().item()),
        "r1": r1_val,
    }


def g_step(*, G, D, g_opt, scaler, use_amp, batch_size, device,
           lambda_pl, pl_interval, pl_ema_ref, step):
    """One generator step: non-saturating logistic adv loss + lazy path-length penalty."""
    z = torch.randn(batch_size, config.NOISE_SIZE, device=device)
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        fake_imgs, _ = G(z, return_w=True)
        fake_logits = D(fake_imgs)
        g_loss = g_nonsaturating_loss(fake_logits)

    g_opt.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(g_loss).backward()
        scaler.step(g_opt)
        scaler.update()
    else:
        g_loss.backward()
        g_opt.step()

    pl_val = 0.0
    if lambda_pl > 0 and (step % pl_interval == 0):
        # Path length in fp32 — uses second-order autograd
        z_pl = torch.randn(max(batch_size // 2, 1), config.NOISE_SIZE, device=device)
        w_pl = G.mapping(z_pl).float().requires_grad_(True)
        # Re-run the synthesis manually with w_pl so autograd.grad can target it
        x = G.const.expand(z_pl.size(0), -1, -1, -1).float()
        rgb = None
        for block in G.blocks:
            x, rgb = block(x.float(), w_pl, skip_rgb=rgb)
        fake_pl = torch.tanh(rgb)
        lengths = path_length_lengths(fake_pl, w_pl)
        mean_len = lengths.mean().detach()
        if pl_ema_ref[0] == 0.0:
            pl_ema_ref[0] = float(mean_len.item())
        else:
            pl_ema_ref[0] = 0.99 * pl_ema_ref[0] + 0.01 * float(mean_len.item())
        pl_pen = (lengths - pl_ema_ref[0]).pow(2).mean()
        pl_loss = lambda_pl * pl_interval * pl_pen
        g_opt.zero_grad(set_to_none=True)
        pl_loss.backward()
        g_opt.step()
        pl_val = float(pl_pen.detach().item())

    return {"g_loss": float(g_loss.detach().item()), "pl": pl_val}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train small-scale StyleGAN2 on FFHQ")
    p.add_argument("--resume", action="store_true",
                   help=f"Resume from {os.path.join(config.CKPT_DIR, 'latest.pth')} if present")
    p.add_argument("--no-resume", action="store_true",
                   help="Start fresh even if a checkpoint exists")
    p.add_argument("--total-steps", type=int, default=None,
                   help="Override config.TOTAL_STEPS")
    return p.parse_args()


def main():
    args = parse_args()
    device = config.DEVICE
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = bool(config.USE_AMP) and device.type == "cuda"
    total_steps = int(args.total_steps if args.total_steps else config.TOTAL_STEPS)
    accum = max(1, int(getattr(config, "GRAD_ACCUM_STEPS", 1)))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    loader = get_dataloader()
    data_iter = infinite_loader(loader)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    G = Generator(
        z_dim=config.NOISE_SIZE, w_dim=config.W_DIM,
        img_resolution=config.IMAGE_SIZE, img_channels=config.NUM_CHANNELS,
        channel_base=config.CHANNEL_BASE, channel_max=config.CHANNEL_MAX,
        mapping_layers=config.MAPPING_LAYERS,
    ).to(device)
    D = Discriminator(
        img_resolution=config.IMAGE_SIZE, img_channels=config.NUM_CHANNELS,
        channel_base=config.CHANNEL_BASE, channel_max=config.CHANNEL_MAX,
    ).to(device)
    G_ema = copy.deepcopy(G).eval()
    for p in G_ema.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in G.parameters()) + sum(p.numel() for p in D.parameters())
    print(f"Models: G={sum(p.numel() for p in G.parameters())/1e6:.1f}M  "
          f"D={sum(p.numel() for p in D.parameters())/1e6:.1f}M  "
          f"total={n_params/1e6:.1f}M")

    # StyleGAN2 lazy-regularization LR scaling: regular optimizer LR is
    # multiplied by interval/(interval+1) when applying lazy reg every N steps.
    # Skipped here — we instead bake the interval into the loss weight.
    g_opt = optim.Adam(G.parameters(), lr=config.LR_G, betas=config.BETAS)
    d_opt = optim.Adam(D.parameters(), lr=config.LR_D, betas=config.BETAS)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    # Fixed-noise grid for visual progress tracking
    fixed_z = torch.randn(int(config.NUM_SAMPLE_IMAGES), config.NOISE_SIZE, device=device)
    pl_ema_ref = [0.0]

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLES_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CKPT_DIR, "latest.pth")

    start_step = 0
    if args.no_resume:
        pass
    elif (args.resume or os.path.exists(ckpt_path)) and os.path.exists(ckpt_path):
        try:
            ckpt = load_checkpoint(ckpt_path, G=G, D=D, G_ema=G_ema,
                                   g_opt=g_opt, d_opt=d_opt, scaler=scaler,
                                   device=device)
            start_step = int(ckpt.get("step", 0))
            pl_ema_ref[0] = float(ckpt.get("pl_ema", 0.0))
            if "fixed_z" in ckpt:
                fz = ckpt["fixed_z"].to(device)
                if fz.shape == fixed_z.shape:
                    fixed_z = fz
            print(f"Resumed from {ckpt_path} at step {start_step}")
        except Exception as exc:
            print(f"WARN: could not load checkpoint ({exc}); starting fresh")
            start_step = 0

    if start_step >= total_steps:
        print(f"start_step ({start_step}) >= TOTAL_STEPS ({total_steps}). Nothing to do.")
        return

    # ------------------------------------------------------------------
    # Training-log CSV
    # ------------------------------------------------------------------
    if start_step == 0 or not os.path.exists(config.RESULT_FILE):
        os.makedirs(os.path.dirname(config.RESULT_FILE) or ".", exist_ok=True)
        with open(config.RESULT_FILE, "w") as f:
            f.write("step,d_loss,g_loss,r1,pl,real_logit,fake_logit,"
                    "sec_per_step,img_per_sec,gpu_mem_gb,gpu_peak_gb,eta_sec\n")

    # EMA decay derived from EMA_KIMG and (effective) batch size
    eff_batch = config.BATCH_SIZE * accum
    ema_beta = 0.5 ** (eff_batch / max(config.EMA_KIMG * 1000, 1e-8))
    print(f"Training: total_steps={total_steps}  batch={config.BATCH_SIZE}x{accum}  "
          f"AMP={use_amp}  ema_beta={ema_beta:.4f}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    G.train()
    D.train()

    # Running averages for smoother console logging
    ravg = {"d_loss": 0.0, "g_loss": 0.0, "r1": 0.0, "pl": 0.0,
            "real_logit": 0.0, "fake_logit": 0.0, "n": 0}

    last_t = time.time()
    for step in range(start_step + 1, total_steps + 1):
        # ---- Discriminator (with optional grad accumulation) ----
        d_stats_acc = {"d_loss": 0.0, "real_logit": 0.0, "fake_logit": 0.0, "r1": 0.0}
        for _ in range(accum):
            real_imgs = next(data_iter).to(device, non_blocking=True)
            stats = d_step(
                real_imgs, G=G, D=D, d_opt=d_opt, scaler=scaler,
                use_amp=use_amp, lambda_r1=config.LAMBDA_R1,
                r1_interval=config.R1_INTERVAL, step=step,
            )
            for k, v in stats.items():
                d_stats_acc[k] += v / accum

        # ---- Generator ----
        g_stats_acc = {"g_loss": 0.0, "pl": 0.0}
        for _ in range(accum):
            stats = g_step(
                G=G, D=D, g_opt=g_opt, scaler=scaler, use_amp=use_amp,
                batch_size=config.BATCH_SIZE, device=device,
                lambda_pl=config.LAMBDA_PL, pl_interval=config.PL_INTERVAL,
                pl_ema_ref=pl_ema_ref, step=step,
            )
            for k, v in stats.items():
                g_stats_acc[k] += v / accum

        # ---- EMA ----
        update_ema(G_ema, G, ema_beta)

        # ---- Logging ----
        for k, v in d_stats_acc.items():
            ravg[k] += v
        for k, v in g_stats_acc.items():
            ravg[k] += v
        ravg["n"] += 1

        if step % config.LOG_EVERY == 0 or step == start_step + 1:
            n = max(ravg["n"], 1)
            now = time.time()
            # First log line of a run has no prior interval to time against
            first_line = (step == start_step + 1)
            interval_steps = 1 if first_line else config.LOG_EVERY
            sec = 0.0 if first_line else (now - last_t) / max(interval_steps, 1)
            last_t = now

            img_per_sec = (config.BATCH_SIZE * accum / sec) if sec > 0 else 0.0
            remaining = max(0, total_steps - step)
            eta_sec = remaining * sec if sec > 0 else 0.0
            eta_str = _format_eta(eta_sec)

            if device.type == "cuda":
                gpu_mem_gb  = torch.cuda.memory_allocated(device)     / (1024 ** 3)
                gpu_peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                gpu_str = f"mem={gpu_mem_gb:4.1f}/peak={gpu_peak_gb:4.1f}GB  "
                # Reset peak so each window's peak is meaningful
                torch.cuda.reset_peak_memory_stats(device)
            else:
                gpu_mem_gb = gpu_peak_gb = 0.0
                gpu_str = ""

            print(
                f"[{step:>7d}/{total_steps}] "
                f"D={ravg['d_loss']/n:.3f}  G={ravg['g_loss']/n:.3f}  "
                f"R1={ravg['r1']/n:.3f}  PL={ravg['pl']/n:.3f}  "
                f"D(real)={ravg['real_logit']/n:+.2f}  D(fake)={ravg['fake_logit']/n:+.2f}  "
                f"sec/it={sec:.3f}  img/s={img_per_sec:5.1f}  "
                f"{gpu_str}eta={eta_str}",
                flush=True,
            )
            with open(config.RESULT_FILE, "a") as f:
                f.write(
                    f"{step},{ravg['d_loss']/n:.4f},{ravg['g_loss']/n:.4f},"
                    f"{ravg['r1']/n:.4f},{ravg['pl']/n:.4f},"
                    f"{ravg['real_logit']/n:.4f},{ravg['fake_logit']/n:.4f},"
                    f"{sec:.4f},{img_per_sec:.2f},"
                    f"{gpu_mem_gb:.3f},{gpu_peak_gb:.3f},{eta_sec:.1f}\n"
                )
                f.flush()
            for k in ravg:
                ravg[k] = 0 if k == "n" else 0.0

        # ---- Sample grid ----
        if step % config.SAMPLE_EVERY == 0:
            G_ema.eval()
            with torch.no_grad():
                imgs = G_ema(fixed_z)
            nrow = max(1, int(round(math.sqrt(fixed_z.size(0)))))
            save_image_grid(imgs, os.path.join(config.SAMPLES_DIR,
                                               f"step_{step:07d}.png"),
                            nrow=nrow)
            G_ema.train()  # safe (no BN/Dropout); keeps semantics consistent

        # ---- Checkpoint ----
        if step % config.CHECKPOINT_EVERY == 0 or step == total_steps:
            save_checkpoint(ckpt_path, step=step, G=G, D=D, G_ema=G_ema,
                            g_opt=g_opt, d_opt=d_opt, scaler=scaler,
                            pl_ema=pl_ema_ref[0], fixed_z=fixed_z)
            # Also save a versioned copy of milestone checkpoints
            milestone = os.path.join(config.CKPT_DIR, f"step_{step:07d}.pth")
            save_checkpoint(milestone, step=step, G=G, D=D, G_ema=G_ema,
                            g_opt=g_opt, d_opt=d_opt, scaler=scaler,
                            pl_ema=pl_ema_ref[0], fixed_z=fixed_z)
            print(f"Saved checkpoint -> {ckpt_path} (and {milestone})")

    # ------------------------------------------------------------------
    # Final sample grid
    # ------------------------------------------------------------------
    G_ema.eval()
    with torch.no_grad():
        imgs = G_ema(fixed_z)
    final_path = os.path.join(config.OUTPUT_DIR, "final_samples.png")
    save_image_grid(imgs, final_path,
                    nrow=max(1, int(round(math.sqrt(fixed_z.size(0))))))
    print(f"Training complete. Final samples -> {final_path}")


if __name__ == "__main__":
    main()
