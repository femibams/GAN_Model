"""One-time setup helper for the FFHQ dataset.

The training pipeline only needs a folder of face images.  This script verifies
that ``config.FFHQ_IMG_DIR`` resolves to such a folder and reports how many
images it contains.

How to obtain FFHQ thumbnails
-----------------------------
Method 1 — official downloader (requires accepting NVIDIA's license):

    git clone https://github.com/NVlabs/ffhq-dataset.git
    cd ffhq-dataset
    python download_ffhq.py --thumbnails --json --num_threads 4
    # Move the result so it matches config.FFHQ_IMG_DIR:
    mkdir -p ../data/ffhq
    mv thumbnails128x128 ../data/ffhq/

Method 2 — Kaggle mirror (no license, ~10k subset also available):

    pip install kaggle
    # Place your kaggle.json API token in ~/.kaggle/kaggle.json
    kaggle datasets download -d greatgamedota/ffhq-face-data-set \
        -p data/ffhq --unzip

Either layout (flat or bucketed under 00000/, 01000/, ...) works because the
dataset loader scans recursively for *.png/*.jpg files.

Run me to verify
----------------
    python prepare_ffhq.py
"""
import os
import sys

import config
from dataset import _scan_images


def main() -> None:
    img_dir = config.FFHQ_IMG_DIR
    print(f"Image dir : {img_dir}")
    if not os.path.isdir(img_dir):
        sys.exit(f"  ERROR: directory not found. See docstring for download instructions.")

    paths = _scan_images(img_dir)
    print(f"Images    : {len(paths):,} ({paths[0]} ... {paths[-1]})")
    if len(paths) < 10_000:
        print("  WARN: dataset has fewer than 10k images. The model will still train")
        print("        but quality at 128x128 may suffer; see the README for guidance.")
    else:
        print("Looks good — ready to run `python train.py`.")


if __name__ == "__main__":
    main()
