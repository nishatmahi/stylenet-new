import os
import re
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

T5_CKPT = "csebuetnlp/banglat5"
MAX_LEN = 48
DROP_P  = 0.2      # token-drop rate for the denoising task

tokenizer = T5Tokenizer.from_pretrained(T5_CKPT, use_fast=False)


def strip_ext(img_id):
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        if img_id.lower().endswith(ext.lower()):
            return img_id[: -len(ext)]
    return img_id


def encode(texts, max_len=MAX_LEN):
    texts = [t.replace("\u09f7", "\u0964") for t in texts]
    enc = tokenizer(texts, max_length=max_len, truncation=True,
                    padding=True, return_tensors="pt")
    return enc.input_ids, enc.attention_mask


def to_labels(input_ids):
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return labels


class FactualDataset(Dataset):
    """CLIP features + the paired factual caption. The caption is returned as
    text too, because the V2L loss needs to encode it on the text side."""
    def __init__(self, cache_dir, caption_file):
        self.cache_dir = cache_dir
        self.items = self._load(caption_file)

    def _load(self, caption_file):
        with open(caption_file, encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        items, imgs = [], set()
        r = re.compile(r'#\d*')
        missing = malformed = 0

        for line in lines:
            parts = [x.strip() for x in r.split(line) if x.strip()]
            if len(parts) < 2:
                malformed += 1
                continue
            img_id = strip_ext(parts[0])
            if not os.path.exists(os.path.join(self.cache_dir, f"{img_id}.pt")):
                missing += 1
                continue
            items.append((img_id, parts[1]))
            imgs.add(img_id)

        if malformed:
            print(f"[WARN] {caption_file}: {malformed} malformed lines.")
        if missing:
            print(f"[WARN] {caption_file}: {missing} lines without cached features.")
        if not items:
            raise RuntimeError(f"No usable samples in {caption_file}")
        print(f"[INFO] {os.path.basename(caption_file)}: {len(items)} captions, "
              f"{len(imgs)} images")
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, ix):
        img_id, caption = self.items[ix]
        feats = torch.load(os.path.join(self.cache_dir, f"{img_id}.pt"),
                           map_location='cpu')          # [50, 768] fp16
        return feats, caption


def collate_factual(batch):
    feats, caps = zip(*batch)
    ids, mask = encode(caps)
    return torch.stack(feats, 0), ids, mask, to_labels(ids)


class StyleTextDataset(Dataset):
    """Monolingual stylized text. Returns (corrupted, clean) for the denoising
    reconstruction task: the encoder sees a corrupted sentence and the decoder
    must restore it, so it has to rely on the style vector for what the noise
    removed. This is what forces the style vector to carry style."""
    def __init__(self, caption_file, drop_p=DROP_P, seed=0):
        with open(caption_file, encoding='utf-8') as f:
            self.lines = [x.strip() for x in f if x.strip()]
        if not self.lines:
            raise RuntimeError(f"{caption_file} is empty")
        self.drop_p = drop_p
        self.rng = random.Random(seed)
        print(f"[INFO] {os.path.basename(caption_file)}: {len(self.lines)} lines")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, ix):
        text = self.lines[ix]
        words = text.split()
        kept = [w for w in words if self.rng.random() > self.drop_p]
        if not kept:
            kept = words[:1]
        return " ".join(kept), text


def collate_style(batch):
    corrupted, clean = zip(*batch)
    c_ids, c_mask = encode(corrupted)
    t_ids, _ = encode(clean)
    return c_ids, c_mask, to_labels(t_ids)


def factual_loader(cache_dir, caption_file, bs, shuffle=False, workers=4):
    return DataLoader(FactualDataset(cache_dir, caption_file),
                      batch_size=bs, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0,
                      collate_fn=collate_factual)


def style_loader(caption_file, bs, shuffle=True, workers=2):
    return DataLoader(StyleTextDataset(caption_file),
                      batch_size=bs, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0,
                      collate_fn=collate_style)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch
