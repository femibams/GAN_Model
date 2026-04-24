import json
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import config


def attrs_to_prompt(attr_row):
    tokens = ["a portrait photo of a person"]

    if attr_row.get("Smiling", -1) > 0:
        tokens.append("smiling")
    if attr_row.get("Young", -1) > 0:
        tokens.append("young")
    if attr_row.get("Male", -1) > 0:
        tokens.append("male")

    hair_tokens = []
    if attr_row.get("Black_Hair", -1) > 0:
        hair_tokens.append("black hair")
    if attr_row.get("Blond_Hair", -1) > 0:
        hair_tokens.append("blond hair")
    if attr_row.get("Brown_Hair", -1) > 0:
        hair_tokens.append("brown hair")
    if attr_row.get("Gray_Hair", -1) > 0:
        hair_tokens.append("gray hair")
    if attr_row.get("Bald", -1) > 0:
        hair_tokens.append("bald")
    if hair_tokens:
        tokens.extend(hair_tokens[:1])

    if attr_row.get("Eyeglasses", -1) > 0:
        tokens.append("wearing eyeglasses")
    if attr_row.get("Mustache", -1) > 0:
        tokens.append("with mustache")
    if attr_row.get("Wearing_Hat", -1) > 0:
        tokens.append("wearing a hat")

    return ", ".join(tokens)


class CelebADataset(Dataset):
    def __init__(self, img_dir, bbox_file, attr_file=None, transform=None, return_prompt=False,
                 clip_embeddings=None):
        self.img_dir = img_dir
        self.return_prompt = bool(return_prompt)
        # Reading via pandas is fine, but avoid per-sample iloc overhead by
        # materializing the needed columns as arrays once.
        df = pd.read_csv(bbox_file)

        if self.return_prompt and attr_file and os.path.exists(attr_file):
            attrs_df = pd.read_csv(attr_file)
            merged = df.merge(attrs_df, on="image_id", how="left")
            self.prompts = merged.apply(attrs_to_prompt, axis=1).astype(str).to_numpy()
        else:
            self.prompts = None

        # Pre-computed CLIP embeddings [N, EMBEDDING_SIZE] on CPU — avoids CLIP
        # inference inside the training loop (which adds ~50-100ms per batch).
        self.clip_embeddings = clip_embeddings  # may be None

        self.image_ids = df["image_id"].astype(str).to_numpy()
        self.bboxes_xywh = df[["x_1", "y_1", "width", "height"]].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        # bbox format: x, y, width, height
        x, y, w, h = self.bboxes_xywh[idx]

        # Convert to normalized [x1, y1, x2, y2]
        W, H = image.size
        x1 = x / W
        y1 = y / H
        x2 = (x + w) / W
        y2 = (y + h) / H

        bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        if self.return_prompt:
            if self.clip_embeddings is not None:
                # Return the pre-computed embedding directly (a CPU tensor)
                text_input = self.clip_embeddings[idx]
            elif self.prompts is not None:
                text_input = self.prompts[idx]
            else:
                text_input = getattr(config, "DEFAULT_PROMPT", "a portrait photo of a person")
        else:
            text_input = torch.zeros(config.EMBEDDING_SIZE, dtype=torch.float32)

        return image, text_input, bbox


class FFHQDataset(Dataset):
    """FFHQ thumbnails128x128 dataset.

    Expects:
      img_dir   — path to thumbnails128x128/ (contains subdirs 00000/, 01000/, …)
      json_file — path to ffhq-dataset-v2.json

    Bounding boxes are derived from the 68 face landmarks stored under
    image.face_landmarks (already in 1024×1024 space), then normalised to [0, 1].
    Image paths follow the JSON thumbnail.file_path convention: {subdir}/{id:05d}.png
    """

    _BBOX_ORIGIN = 1024  # landmarks are always in 1024×1024 space

    def __init__(self, img_dir, json_file, transform=None, clip_embeddings=None):
        self.img_dir = img_dir
        self.transform = transform
        self.clip_embeddings = clip_embeddings

        with open(json_file) as f:
            meta = json.load(f)

        # Sort by integer key so index order is deterministic
        entries = sorted(meta.items(), key=lambda kv: int(kv[0]))

        self.image_ids = []
        self.bboxes_xyxy = []  # (x1, y1, x2, y2) in 1024-px space, derived from landmarks
        for key, val in entries:
            iid = int(key)
            landmarks = val["image"]["face_landmarks"]  # list of [x, y] pairs
            xs = [p[0] for p in landmarks]
            ys = [p[1] for p in landmarks]
            self.image_ids.append(iid)
            self.bboxes_xyxy.append((min(xs), min(ys), max(xs), max(ys)))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        iid = self.image_ids[idx]
        subdir = f"{(iid // 1000) * 1000:05d}"
        img_path = os.path.join(self.img_dir, subdir, f"{iid:05d}.png")
        image = Image.open(img_path).convert("RGB")

        x1, y1, x2, y2 = self.bboxes_xyxy[idx]
        o = self._BBOX_ORIGIN
        bbox = torch.tensor([x1 / o, y1 / o, x2 / o, y2 / o], dtype=torch.float32)
        bbox.clamp_(0.0, 1.0)

        if self.transform:
            image = self.transform(image)

        if self.clip_embeddings is not None:
            text_input = self.clip_embeddings[idx]
        else:
            text_input = getattr(config, "DEFAULT_PROMPT", "a high-quality portrait photo of a person")

        return image, text_input, bbox


def precompute_clip_embeddings(clip_model, tokenizer, device,
                               embedding_size, cache_path="outputs/clip_embeddings.pt",
                               attr_file=None, bbox_file=None, prompts=None):
    """Encode all prompts once and cache to disk. Returns a CPU float32 tensor [N, D].

    Callers may either pass a ready-made ``prompts`` list, or supply ``attr_file`` /
    ``bbox_file`` for CelebA-style attribute-derived prompts.
    """
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    if os.path.exists(cache_path):
        print(f"Loading cached CLIP embeddings from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print("Pre-computing CLIP embeddings for the full dataset (one-time cost)...")
    if prompts is None:
        df = pd.read_csv(bbox_file)
        if attr_file and os.path.exists(attr_file):
            attrs_df = pd.read_csv(attr_file)
            merged = df.merge(attrs_df, on="image_id", how="left")
            prompts = merged.apply(attrs_to_prompt, axis=1).astype(str).tolist()
        else:
            default = getattr(config, "DEFAULT_PROMPT", "a portrait photo of a person")
            prompts = [default] * len(df)

    batch_size = 512
    all_embeddings = []
    clip_model.eval()
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            tokens = tokenizer(batch).to(device)
            emb = clip_model.encode_text(tokens).float().cpu()
            all_embeddings.append(emb)
            if start % 10000 == 0:
                print(f"  {start}/{len(prompts)}")

    embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(embeddings, cache_path)
    print(f"Saved CLIP embeddings to {cache_path} — shape {tuple(embeddings.shape)}")
    return embeddings


def get_dataloader(return_prompt=False, batch_size=None, shuffle=True, clip_embeddings=None):
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset_name = getattr(config, "DATASET", "celeba").lower()
    if dataset_name == "ffhq":
        dataset = FFHQDataset(
            img_dir=config.FFHQ_IMG_DIR,
            json_file=config.FFHQ_JSON_FILE,
            transform=transform,
            clip_embeddings=clip_embeddings,
        )
    else:
        dataset = CelebADataset(
            img_dir=config.IMG_DIR,
            bbox_file=config.BBOX_FILE,
            attr_file=getattr(config, "ATTR_FILE", None),
            transform=transform,
            return_prompt=return_prompt,
            clip_embeddings=clip_embeddings,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=(batch_size if batch_size is not None else config.BATCH_SIZE),
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=bool(getattr(config, "PERSISTENT_WORKERS", False)) if config.NUM_WORKERS > 0 else False,
        prefetch_factor=int(getattr(config, "PREFETCH_FACTOR", 2)) if config.NUM_WORKERS > 0 else None,
    )

    return dataloader
