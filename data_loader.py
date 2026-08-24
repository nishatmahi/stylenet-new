"""
data_loader.py — one style per run, no style lexicon anywhere.

The style pass corrupts its input with UNIFORM word dropout: every word is
dropped with the same probability, whether it carries style or content. That
is deliberate. The earlier design deleted words judged to be "style words",
which required somebody to decide which Bangla words are romantic — a
judgement that was measured to be wrong on 4.2% of the corpus and to leak
style into the input on 9.4% of lines.

Uniform dropout makes no such claim. Its one knob, `dropout`, trades content
preservation against style leakage, and it is tuned from validation loss and
sample output rather than from anyone's opinion about Bangla.
"""

import os
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

try:
    from normalizer import normalize
except ImportError:
    raise ImportError(
        "BanglaT5 needs the csebuetnlp normalization pipeline it was pretrained "
        "with:\n  pip install git+https://github.com/csebuetnlp/normalizer")

T5_CKPT = "csebuetnlp/banglat5"
MAX_LEN = 48

# Every word dropped with this probability during the style pass.
#   low  -> content survives, but style words survive too and can be copied
#   high -> no copying, but the input is gutted and content must be invented
# One number, tuned from val loss. Not a claim about which words are stylish.
DROPOUT = 0.4

tokenizer = T5Tokenizer.from_pretrained(T5_CKPT, use_fast=False)


def strip_ext(img_id):
    for e in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        if img_id.lower().endswith(e.lower()):
            return img_id[:-len(e)]
    return img_id


def encode(texts, max_len=MAX_LEN):
    texts = [normalize(t) for t in texts]
    enc = tokenizer(texts, max_length=max_len, truncation=True,
                    padding=True, return_tensors="pt")
    return enc.input_ids, enc.attention_mask


def to_labels(input_ids):
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return labels


def _split_line(line):
    """`1000092795.jpg#0<TAB>caption` -> ('1000092795', 'caption')."""
    if '\t' in line:
        key, cap = line.split('\t', 1)
    elif '#' in line:
        key, cap = line.split('#', 1)
        parts = cap.split(None, 1)
        cap = parts[1] if len(parts) > 1 else ''
    else:
        return None
    return strip_ext(key.split('#')[0].strip()), cap.strip()


def read_lines(path):
    return [l.strip() for l in open(path, encoding='utf-8') if l.strip()]


# --------------------------------------------------------------------------
class FactualDataset(Dataset):
    """One row per (image, caption) pair. No epoch state, no rotation."""

    def __init__(self, cache_dir, caption_file):
        self.cache_dir = cache_dir
        self.rows = self._load(caption_file)

    def _load(self, caption_file):
        rows, seen, malformed = [], {}, 0
        for line in read_lines(caption_file):
            parsed = _split_line(line)
            if parsed is None or not parsed[1]:
                malformed += 1
                continue
            img, cap = parsed
            if img not in seen:
                seen[img] = os.path.exists(
                    os.path.join(self.cache_dir, f"{img}.pt"))
            if seen[img]:
                rows.append((img, cap))

        miss = sum(1 for v in seen.values() if not v)
        nimg = sum(1 for v in seen.values() if v)
        if malformed:
            print(f"[WARN] {caption_file}: {malformed} malformed lines.")
        if miss:
            print(f"[WARN] {caption_file}: {miss} images without cached features.")
        if not rows:
            raise RuntimeError(f"No usable samples in {caption_file}")
        print(f"[INFO] {os.path.basename(caption_file)}: {nimg} images, "
              f"{len(rows)} captions ({len(rows)/max(nimg,1):.1f} per image).")
        return rows

    def captions(self):
        return [c for _, c in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, ix):
        img, cap = self.rows[ix]
        feats = torch.load(os.path.join(self.cache_dir, f"{img}.pt"),
                           map_location='cpu')        # [50, 768] fp16
        return feats, cap


def collate_factual(batch):
    feats, caps = zip(*batch)
    ids, mask = encode(caps)
    return torch.stack(feats, 0), ids, mask, to_labels(ids)


class StyleTextDataset(Dataset):
    """Encoder sees the styled sentence with words randomly deleted; the
    decoder must restore it in full.

    No word list. Every word is treated identically. Whatever the adapters
    end up learning about style, they learn because the decoder had to put
    the missing words back — not because anything was labelled 'style'.
    """

    def __init__(self, caption_file, dropout=DROPOUT, seed=0, min_keep=2):
        self.lines = read_lines(caption_file)
        if not self.lines:
            raise RuntimeError(f"{caption_file} is empty")
        self.p = float(dropout)
        self.seed = seed
        self.min_keep = min_keep
        print(f"[INFO] {os.path.basename(caption_file)}: {len(self.lines)} lines, "
              f"uniform dropout p={self.p}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, ix):
        # Seeded per item: workers fork with the same RNG state, so a shared
        # generator would hand every worker identical corruptions.
        rng = random.Random(self.seed * 1_000_003 + ix)
        words = self.lines[ix].split()
        kept = [w for w in words if rng.random() > self.p]
        if len(kept) < self.min_keep:                  # never hand over nothing
            kept = rng.sample(words, min(self.min_keep, len(words)))
            kept.sort(key=words.index)
        return " ".join(kept), self.lines[ix]


def collate_style(batch):
    corrupt, clean = zip(*batch)
    c_ids, c_mask = encode(corrupt)
    t_ids, _ = encode(clean)
    return c_ids, c_mask, to_labels(t_ids)


class SubsetEpochSampler(torch.utils.data.Sampler):
    """Optional, off unless --captions_per_epoch > 0. Shortens epochs only."""

    def __init__(self, n, num_samples=None, seed=0):
        self.n = n
        self.num_samples = min(num_samples or n, n)
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(self.n, generator=g).tolist()[:self.num_samples]

    def __len__(self):
        return self.num_samples


def factual_loader(cache_dir, caption_file, bs, shuffle=False, workers=4,
                   captions_per_epoch=0, return_dataset=False):
    ds = FactualDataset(cache_dir, caption_file)
    sampler = None
    if shuffle and captions_per_epoch and captions_per_epoch < len(ds):
        sampler = SubsetEpochSampler(len(ds), captions_per_epoch)
        shuffle = False
        print(f"[INFO] sampling {len(sampler)}/{len(ds)} captions per epoch")
    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle, sampler=sampler,
                    num_workers=workers, pin_memory=True,
                    persistent_workers=workers > 0, collate_fn=collate_factual)
    return (dl, ds) if return_dataset else dl


def style_loader(caption_file, bs, dropout=DROPOUT, shuffle=True, workers=2):
    return DataLoader(
        StyleTextDataset(caption_file, dropout=dropout),
        batch_size=bs, shuffle=shuffle, num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0, collate_fn=collate_style)
