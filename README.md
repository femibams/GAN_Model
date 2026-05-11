# Small-scale StyleGAN2 — FFHQ Faces

A scaled-down PyTorch implementation of StyleGAN2 (Karras et al., 2020) for
face generation on FFHQ at **64×64 or 128×128**.  All the core StyleGAN2
mechanisms are preserved, with channel widths and parameter counts shrunk
~4× so training fits on a single mid-range GPU.

What's preserved from StyleGAN2:

- 8-layer mapping network `z → w` with PixelNorm and reduced-LR FC layers
- Modulated convolutions with **weight demodulation** (no AdaIN, no instance norm)
- Per-layer learned **noise injection** for stochastic detail
- **Skip-to-RGB** synthesis: each block emits an RGB tensor that's added to
  the upsampled output of the previous block
- Residual discriminator with a **MinibatchStdDev** layer
- **R1 gradient penalty** on D (lazy regularisation, every `R1_INTERVAL` steps)
- **Path-length regularisation** on G (lazy, every `PL_INTERVAL` steps)
- **Generator EMA**, used for all sample saves and inference

Training uses non-saturating logistic GAN loss, mixed precision, and writes a
sample grid every `SAMPLE_EVERY` iterations so progress is visible.

## Repo layout

```
config.py        all hyperparameters in one place
models.py        Generator + Discriminator (StyleGAN2 building blocks)
dataset.py       FFHQDataset: scans a folder, hflip + crop-jitter
utils.py         losses, R1 / path-length penalties, image-grid saver
train.py         iteration-based training loop with AMP / R1 / PL / EMA / resume
infer.py         load a checkpoint and generate a single face from random noise
infer_prompt.py  prompt-guided generation via CLIP-guided latent optimization
prepare_ffhq.py  verify the FFHQ folder is set up
```

## 1. Setup

```bash
pip install -r requirements.txt
```

PyTorch 1.12+ with CUDA is recommended; AMP + 128×128 + ~25M params fit in
~6 GB of GPU memory at `BATCH_SIZE=16`.

## 2. Get the data

Download FFHQ thumbnails (128×128, ~2 GB) into `data/ffhq/thumbnails128x128/`
using either of the methods documented inside `prepare_ffhq.py`:

```bash
python prepare_ffhq.py    # verifies the folder layout
```

The dataset loader scans the folder recursively, so both flat
(`thumbnails128x128/00000.png`) and bucketed
(`thumbnails128x128/00000/00000.png`) layouts work.  Any folder of PNG/JPG
faces is fine — a 10k subset is enough to start training, but quality
plateaus earlier.

To use a subset without moving files, set `MAX_IMAGES = 10_000` in
`config.py`.

## 3. Train

```bash
python train.py
```

Key knobs in `config.py`:

| name | default | meaning |
|------|---------|---------|
| `IMAGE_SIZE` | 128 | output resolution; must be a power of two |
| `CHANNEL_BASE`, `CHANNEL_MAX` | 8192, 256 | `nf(res) = min(max, base // res)` |
| `BATCH_SIZE` | 16 | per-step batch size |
| `GRAD_ACCUM_STEPS` | 1 | effective batch = `BATCH_SIZE × accum` |
| `TOTAL_STEPS` | 150 000 | total iterations (target 100k–200k) |
| `LR_G`, `LR_D` | 2.5e-3 | Adam learning rates |
| `LAMBDA_R1`, `R1_INTERVAL` | 1.0, 16 | lazy R1 penalty |
| `LAMBDA_PL`, `PL_INTERVAL` | 2.0, 4 | lazy path-length penalty |
| `SAMPLE_EVERY` | 1 000 | sample-grid save cadence |
| `CHECKPOINT_EVERY` | 5 000 | checkpoint save cadence |

While training, you'll see lines like

```
[   1000/150000] D=0.823  G=1.412  R1=0.045  PL=0.037  D(real)=+1.27  D(fake)=-1.32  sec/it=0.184
```

and a CSV at `outputs/training_log.csv` with the same fields. Sample grids
land in `outputs/samples/step_XXXXXXX.png`.

### Resume

A checkpoint is written to `outputs/checkpoints/latest.pth` every
`CHECKPOINT_EVERY` iterations and at the end of training.  Versioned
milestones go to `step_XXXXXXX.pth` alongside it.

```bash
python train.py            # auto-resumes from latest.pth if it exists
python train.py --resume   # explicit (same effect)
python train.py --no-resume  # ignore any existing checkpoint and start fresh
python train.py --total-steps 200000   # extend training
```

The checkpoint stores the architecture config it was trained with, so
`infer.py` rebuilds the correct generator regardless of the current
`config.py` values.

## 4. Inference — generate a sample face

```bash
python infer.py                                                  # one face, EMA G, truncation 0.7, random seed
python infer.py --seed 42 --truncation 0.5                       # reproducible run
python infer.py --ckpt outputs/checkpoints/step_0050000.pth --out preview.png
```

Each run prints the seed it used (e.g. `Seed: 16962057276122653174`) so a
result you like can be reproduced with `--seed <that number>`. Truncation
`< 1.0` blends `w` toward the average `w` (computed from 10k random `z`'s),
trading diversity for fidelity — `~0.5–0.7` is a good range for face GANs.

## 5. Prompt-guided inference (CLIP-steered)

The trained generator is **unconditional** — it has no text input. To bias
generation toward a prompt, `infer_prompt.py` keeps G frozen and optimizes a
single style vector `w` so the generated image's CLIP embedding matches the
prompt's CLIP embedding (a StyleCLIP-style latent optimization).

```bash
pip install open_clip_torch
python infer_prompt.py --prompt "a smiling woman with red hair"
python infer_prompt.py --prompt "an elderly man with a beard" --steps 400 --lr 0.1
```

Quality is bounded by what G already learned: prompts inside the face
distribution (hair, age, expression, lighting) work; out-of-distribution
prompts ("astronaut on Mars") will not.

Key knobs:

| flag | default | meaning |
|------|---------|---------|
| `--prompt` | (required) | text prompt to steer generation |
| `--steps` | 200 | optimization steps |
| `--lr` | 0.05 | Adam learning rate for `w` |
| `--truncation` | 0.7 | initial truncation psi for `w` |
| `--w-reg` | 0.05 | L2 pull toward `w_avg` (prevents off-manifold drift) |
| `--clip-model` | `ViT-B-16` | `open_clip` model name |
| `--clip-pretrained` | `openai` | `open_clip` pretrained tag |

If results look "averaged" or boring, drop `--w-reg` and raise
`--truncation`. If the image drifts off-manifold (weird artifacts), do the
opposite.

## Notes on quality vs. compute

At 128×128 with the default config:

- 50k iterations is enough to see clear face structure.
- 100k–150k iterations produce recognisable faces with reasonable diversity.
- Going to 200k+ gives diminishing returns at this scale; for substantially
  better fidelity, raise `CHANNEL_BASE` to 16384 / `CHANNEL_MAX` to 512 and
  retrain — that's the full StyleGAN2 channel budget.

If your GPU is smaller, drop `IMAGE_SIZE` to 64 and `BATCH_SIZE` to 8 — the
architecture scales automatically with `IMAGE_SIZE`.
