import torch

# Dataset selection: "celeba" or "ffhq"
DATASET = "ffhq"

# CelebA paths
IMG_DIR = "data/img_align_celeba/img_align_celeba"
BBOX_FILE = "data/list_bbox_celeba.csv"
ATTR_FILE = "data/list_attr_celeba.csv"

# FFHQ paths (thumbnails128x128 matches IMAGE_SIZE=128 directly)
FFHQ_IMG_DIR = "data/ffhq/thumbnails128x128"
FFHQ_JSON_FILE = "data/ffhq/ffhq-dataset-v2.json"

RESULT_FILE = "outputs/training_results.csv"

# Hyperparameters
NOISE_SIZE = 256            # was 100; larger latent = more expressive variation
FEATURE_SIZE = 64           # base channel multiplier (F*8=512 at 4×4 stem)
NUM_CHANNELS = 3
EMBEDDING_SIZE = 512
REDUCED_DIM_SIZE = 256      # was 128; style_dim for AdaIN mapping network
IMAGE_SIZE = 128
ROI_SIZE = 64
BATCH_SIZE = 16             # was 32; reduced to fit within GPU VRAM at 128×128 resolution
NUM_WORKERS = 8
NUM_EPOCHS = 300

LAMBDA_ROI = 1.0
LAMBDA_LEAK = 0.01          # was 0.05; leakage was over-constraining the generator
USE_BACKGROUND_LEAK = True  # use real-image background as reference instead of zeroing
LR = 5e-4                   # 2e-3 caused NaN; lower is safer for new architecture
LR_D = 5e-4                 # match G LR — D must keep pace with G to provide clean gradients
BETAS = (0.0, 0.99)         # was (0.5, 0.999); StyleGAN2 standard — β1=0 kills momentum

# R1 gradient penalty (prevents discriminator from dominating)
LAMBDA_R1 = 0.5    # effective weight = 0.5 * 16 / 2 = 4 ≈ original (10/2=5)
R1_INTERVAL = 16   # lazy regularization: compute every N steps (reduces cost ~16x)
D_TEXT_DIM = 0     # projection discriminator disabled: amplifies R1 gradient ~500x at init

# StyleGAN2 path length regularization (encourages smooth latent-to-image mapping)
LAMBDA_PL = 0.0    # disabled: PL w.r.t. raw noise z is ill-defined when z feeds two paths
PL_INTERVAL = 4    # lazy: compute every N generator steps

# Adaptive Data Augmentation (ADA) — adapts augmentation strength automatically
ADA_TARGET   = 0.6    # target mean(sign(D(real))); 0.6 ≈ D correct ~80% of the time
ADA_INTERVAL = 4      # steps between p updates
ADA_KIMG     = 200.0  # adjustment speed in k-images; lower = faster ramp

# Generator EMA — smoothed weights used for all visualisation and evaluation
EMA_KIMG = 10.0        # smoothing window in k-images (StyleGAN2 default)

# ROI loss warm-up: ramp linearly from 0 → LAMBDA_ROI between these epochs
ROI_WARMUP_START = 10
ROI_WARMUP_END = 30

# LR decay: linear decay from LR_DECAY_START epoch to 0 at NUM_EPOCHS
LR_DECAY_START = 200        # decay in final third only — model needs full LR for most of training

# Perf
USE_AMP = True  # mixed precision on CUDA
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation defaults
EVAL_CKPT_PATH = "outputs/checkpoints/latest.pth"
EVAL_OUTPUT_CSV = "outputs/eval_results.csv"
EVAL_MAX_SAMPLES = 2048
EVAL_CLIP_PROMPT = "a high-quality portrait photo of a person"
EVAL_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
INFER_CLIP_PROMPT = "a high-quality portrait photo of a person"
INFER_CLIP_MODEL_NAME = "ViT-B-32-quickgelu"
INFER_CLIP_PRETRAINED = "openai"

# Text conditioning for training/evaluation
USE_CLIP_TEXT = True
TRAIN_CLIP_MODEL_NAME = "ViT-B-32-quickgelu"
TRAIN_CLIP_PRETRAINED = "openai"
DEFAULT_PROMPT = "a portrait photo of a person"