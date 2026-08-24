"""
data_loader.py — one style per run.


"""

import os
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
    """One row per (image, caption) pair. No epoch state, no rotation.

    NOTE: rows for the same image are CONSECUTIVE, ~4.8 of them. Slicing
    [:3] off a batch gives you one picture three times — see the peek block
    in train.py.
    """

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
    """The style corpus, uncorrupted. One sentence per line.

    StyleNet Task 2 is a plain language model. The sentence appears only as
    the decoder's target; the encoder gets nothing. There is therefore no
    corruption scheme, no word list, and nothing for the decoder to copy.
    """

    def __init__(self, caption_file):
        self.lines = read_lines(caption_file)
        if not self.lines:
            raise RuntimeError(f"{caption_file} is empty")
        print(f"[INFO] {os.path.basename(caption_file)}: "
              f"{len(self.lines)} lines (language model)")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, ix):
        return self.lines[ix]


def collate_style(batch):
    """Labels only. There is no encoder side."""
    ids, _ = encode(list(batch))
    return to_labels(ids)


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


def style_loader(caption_file, bs, shuffle=True, workers=2):
    return DataLoader(StyleTextDataset(caption_file), batch_size=bs,
                      shuffle=shuffle, num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0, collate_fn=collate_style)
