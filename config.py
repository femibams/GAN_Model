"""Configuration for the small-scale StyleGAN2 face GAN.

All hyperparameters live here so train.py / infer.py / dataset.py can read
them directly.  Values are tuned for a single mid-range GPU (~12 GB).
"""
import torch

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
FFHQ_IMG_DIR = "data/ffhq/thumbnails128x128"  # any folder of face PNGs
MAX_IMAGES = None                              # None => use all images on disk

# ---------------------------------------------------------------------------
# Model architecture (scaled-down StyleGAN2)
# ---------------------------------------------------------------------------
IMAGE_SIZE     = 128         # 64 or 128 — must be a power of two
NUM_CHANNELS   = 3
NOISE_SIZE     = 256         # z_dim
W_DIM          = 256         # mapping-network output / synthesis style dim
MAPPING_LAYERS = 8

# Per-resolution feature widths follow StyleGAN2's NF schedule
#   nf(res) = min(channel_max, channel_base // res)
# Reference uses (16384, 512); we shrink to ~1/4 the parameter count.
CHANNEL_BASE = 8192
CHANNEL_MAX  = 256

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE         = 16
GRAD_ACCUM_STEPS   = 1        # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
TOTAL_STEPS        = 150_000  # task target: 100k-200k iterations
LR_G               = 2.5e-3
LR_D               = 2.5e-3
BETAS              = (0.0, 0.99)   # StyleGAN2 standard

# R1 regularization (lazy)
LAMBDA_R1   = 1.0
R1_INTERVAL = 16

# Path-length regularization (lazy)
LAMBDA_PL   = 2.0
PL_INTERVAL = 4

# Generator EMA — ema_beta = 0.5 ** (batch_size / (ema_kimg * 1000))
EMA_KIMG    = 10.0

# Logging / sampling / checkpointing
LOG_EVERY        = 100         # console / CSV log frequency in iters
SAMPLE_EVERY     = 1_000       # write a sample grid every N iters
CHECKPOINT_EVERY = 5_000       # save a checkpoint every N iters
NUM_SAMPLE_IMAGES = 32         # samples per grid

# ---------------------------------------------------------------------------
# Augmentation (handled in dataset.py)
# ---------------------------------------------------------------------------
AUG_HFLIP       = True
AUG_CROP_JITTER = 0.05         # resize to (1 + jitter)*IMAGE_SIZE then random-crop

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
USE_AMP            = True
NUM_WORKERS        = 8
PERSISTENT_WORKERS = True
PREFETCH_FACTOR    = 4

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUT_DIR  = "outputs"
SAMPLES_DIR = "outputs/samples"
CKPT_DIR    = "outputs/checkpoints"
RESULT_FILE = "outputs/training_log.csv"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
