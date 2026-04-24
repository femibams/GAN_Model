"""
One-time setup script for the FFHQ dataset.

Downloads the FFHQ thumbnails128x128 (2.2 GB) and metadata JSON, then
verifies the directory layout expected by FFHQDataset in dataset.py.

Usage
-----
    python prepare_ffhq.py

Requirements
------------
The FFHQ dataset requires accepting NVIDIA's license before downloading.
Follow one of the two methods below:

Method 1 — ffhq-dataset tool (official):
    pip install requests tqdm
    git clone https://github.com/NVlabs/ffhq-dataset.git
    cd ffhq-dataset
    # Download thumbnails (128×128, ~2.2 GB) + metadata JSON only:
    python download_ffhq.py --thumbnails --json --num_threads 4
    # Then move / symlink the output into this project:
    mv thumbnails128x128 ../data/ffhq/thumbnails128x128
    mv ffhq-dataset-v2.json ../data/ffhq/ffhq-dataset-v2.json
    cd ..

Method 2 — Kaggle (unofficial mirror, no license required):
    pip install kaggle
    # Place your kaggle.json API token in ~/.kaggle/kaggle.json
    kaggle datasets download -d greatgamedota/ffhq-face-data-set -p data/ffhq --unzip
    # Rename the extracted folder so the path matches config.FFHQ_IMG_DIR:
    mv data/ffhq/thumbnails128x128 data/ffhq/thumbnails128x128   # already correct

After downloading, run this script to verify the layout:
    python prepare_ffhq.py
"""

import json
import os
import sys


def verify(img_dir: str, json_file: str) -> None:
    print(f"Checking JSON  : {json_file}")
    if not os.path.exists(json_file):
        sys.exit(f"  ERROR: not found. Download ffhq-dataset-v2.json first (see docstring).")

    with open(json_file) as f:
        meta = json.load(f)
    print(f"  Entries       : {len(meta):,}")

    print(f"Checking images: {img_dir}")
    if not os.path.isdir(img_dir):
        sys.exit(f"  ERROR: directory not found. Download thumbnails128x128 first (see docstring).")

    subdirs = sorted(d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d)))
    print(f"  Subdirs found : {len(subdirs)}  (expected 70, one per 1000 images)")

    # Spot-check first and last entry in the JSON
    sample_ids = [0, len(meta) - 1]
    for iid in sample_ids:
        key = str(iid)
        if key not in meta:
            print(f"  WARN: key '{key}' missing from JSON")
            continue
        subdir = f"{(iid // 1000) * 1000:05d}"
        path = os.path.join(img_dir, subdir, f"img{iid:08d}.png")
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  Sample [{iid:05d}] {path} — {status}")

    # Quick bbox sanity-check on first entry
    first_val = meta[str(sample_ids[0])]
    rect = first_val.get("facial_components", {}).get("face_rect")
    print(f"  face_rect[0]  : {rect}  (should be [x, y, w, h] in 1024-px space)")

    print("\nVerification complete. If no ERRORs above, you're ready to train with DATASET='ffhq'.")


if __name__ == "__main__":
    import config
    verify(img_dir=config.FFHQ_IMG_DIR, json_file=config.FFHQ_JSON_FILE)
