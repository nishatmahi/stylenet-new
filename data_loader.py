import os
import re
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

T5_CKPT = "csebuetnlp/banglat5"
MAX_LEN = 48

# At 0.2 the decoder reconstructs by copying the surviving words and never
# consults the style vector — txt loss collapsed to 0.55 and s_style was
# starved of gradient. Higher noise forces it to use the style vector for
# what the corruption removed.
DROP_P  = 0.5

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
    """One sample per unique image per epoch; which of that image's ~5 captions
    is used rotates with the epoch. All captions are still seen across
    training, but the same feature file isn't re-read 5x within one epoch.

    The caption is returned as text too, because V2L encodes it on the text
    side."""
    def __init__(self, cache_dir, caption_file):
        self.cache_dir = cache_dir
        self.img_ids, self.captions = self._load(caption_file)
        self.epoch = 0

    def _load(self, caption_file):
        with open(caption_file, encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        by_img, order = {}, []
        r = re.compile(r'#\d*')
        missing = malformed = 0

        for line in lines:
            parts = [x.strip() for x in r.split(line) if x.strip()]
            if len(parts) < 2:
                malformed += 1
                continue
            img_id = strip_ext(parts[0])
            if img_id not in by_img:
                if not os.path.exists(os.path.join(self.cache_dir, f"{img_id}.pt")):
                    missing += 1
                    by_img[img_id] = None
                    continue
                by_img[img_id] = []
                order.append(img_id)
            if by_img[img_id] is not None:
                by_img[img_id].append(parts[1])

        if malformed:
            print(f"[WARN] {caption_file}: {malformed} malformed lines.")
        if missing:
            print(f"[WARN] {caption_file}: {missing} images without cached features.")
        if not order:
            raise RuntimeError(f"No usable samples in {caption_file}")

        caps = [by_img[i] for i in order]
        total = sum(len(c) for c in caps)
        print(f"[INFO] {os.path.basename(caption_file)}: {len(order)} images, "
              f"{total} captions ({total/len(order):.1f} per image). "
              f"One caption/image per epoch.")
        return order, caps

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, ix):
        img_id = self.img_ids[ix]
        caps = self.captions[ix]
        caption = caps[self.epoch % len(caps)]
        feats = torch.load(os.path.join(self.cache_dir, f"{img_id}.pt"),
                           map_location='cpu')          # [50, 768] fp16
        return feats, caption


def collate_factual(batch):
    feats, caps = zip(*batch)
    ids, mask = encode(caps)
    return torch.stack(feats, 0), ids, mask, to_labels(ids)


class StyleTextDataset(Dataset):
    """Monolingual stylized text. Returns (corrupted, clean): the encoder sees
    a corrupted sentence and the decoder must restore the original, so it has
    to rely on the style vector for what the noise removed."""
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
