"""FFHQ data pipeline.

Scans a directory for image files (recursively, so both flat
`thumbnails128x128/00000.png` and bucketed `thumbnails128x128/00000/00000.png`
layouts work) and yields tensors normalised to [-1, 1].

Augmentations applied to every sample:
  - random horizontal flip (50%)
  - resize to int(img_size * (1 + jitter)) then random crop to img_size

The crop jitter is small (default 0.05) so the pre-aligned faces are not
substantially decentred.
"""
import os
import glob
import random
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import config


def _pil_to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor in [-1, 1] without going through numpy.

    torchvision.ToTensor uses np.array internally and breaks in environments
    with broken numpy installs.  We convert via the raw byte buffer instead.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = bytearray(img.tobytes())
    t = torch.frombuffer(buf, dtype=torch.uint8)
    t = t.view(img.height, img.width, 3).permute(2, 0, 1).contiguous().float()
    t = t.div_(255.0).sub_(0.5).mul_(2.0)
    return t


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _scan_images(root: str) -> List[str]:
    """Recursively find image files under root, sorted for determinism."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Image directory not found: {root}")
    paths: List[str] = []
    for ext in _IMG_EXTS:
        paths.extend(glob.glob(os.path.join(root, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError(f"No images found under {root}")
    return paths


class FFHQDataset(Dataset):
    def __init__(self, img_dir: str, img_size: int,
                 hflip: bool = True, crop_jitter: float = 0.05,
                 max_images: Optional[int] = None):
        self.paths = _scan_images(img_dir)
        if max_images is not None and max_images > 0:
            self.paths = self.paths[:max_images]
        self.img_size = img_size
        self.hflip = bool(hflip)
        self.crop_jitter = float(crop_jitter)

        # Resize size used before random crop
        self._resize_size = int(round(img_size * (1.0 + max(crop_jitter, 0.0))))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        # Resize-with-jitter then random-crop, or plain resize when jitter=0
        if self._resize_size != self.img_size:
            img = TF.resize(img, [self._resize_size, self._resize_size],
                            interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            max_off = self._resize_size - self.img_size
            top = random.randint(0, max_off)
            left = random.randint(0, max_off)
            img = TF.crop(img, top, left, self.img_size, self.img_size)
        else:
            img = TF.resize(img, [self.img_size, self.img_size],
                            interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

        if self.hflip and random.random() < 0.5:
            img = TF.hflip(img)

        return _pil_to_normalized_tensor(img)


def get_dataloader(batch_size: Optional[int] = None, shuffle: bool = True,
                   max_images: Optional[int] = None):
    dataset = FFHQDataset(
        img_dir=config.FFHQ_IMG_DIR,
        img_size=config.IMAGE_SIZE,
        hflip=getattr(config, "AUG_HFLIP", True),
        crop_jitter=getattr(config, "AUG_CROP_JITTER", 0.05),
        max_images=max_images if max_images is not None
                    else getattr(config, "MAX_IMAGES", None),
    )
    print(f"Dataset: {len(dataset)} images @ {dataset.img_size}x{dataset.img_size}")

    nw = config.NUM_WORKERS
    kwargs = dict(
        dataset=dataset,
        batch_size=(batch_size if batch_size is not None else config.BATCH_SIZE),
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
    )
    # PyTorch 1.12 rejects persistent_workers / prefetch_factor when num_workers == 0
    if nw > 0:
        kwargs["persistent_workers"] = bool(getattr(config, "PERSISTENT_WORKERS", False))
        kwargs["prefetch_factor"] = int(getattr(config, "PREFETCH_FACTOR", 2))
    return DataLoader(**kwargs)


def infinite_loader(loader: DataLoader):
    """Yield batches forever, re-iterating the loader each epoch."""
    while True:
        for batch in loader:
            yield batch
