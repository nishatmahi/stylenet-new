import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

try:
    from normalizer import normalize
except ImportError:
    raise ImportError(
        "BanglaT5 requires the csebuetnlp normalization pipeline it was "
        "pretrained with. Install it:\n"
        "  pip install git+https://github.com/csebuetnlp/normalizer"
    )

T5_CKPT = "csebuetnlp/banglat5"
MAX_LEN = 48

# 0.3, not 0.5. At 0.5 half of a ~10-word Bangla caption is gone and the decoder
# is being asked to invent content words. A single global style vector cannot
# supply missing CONTENT, so extra noise does not strengthen the style signal —
# it teaches hallucination. If the decoder is ignoring the style vector,
# diagnose with style_geometry(): cos(fac,style) near 1.0 means the vectors
# collapsed onto each other and no amount of input noise helps.
DROP_P = 0.3

tokenizer = T5Tokenizer.from_pretrained(T5_CKPT, use_fast=False)


def strip_ext(img_id):
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        if img_id.lower().endswith(ext.lower()):
            return img_id[: -len(ext)]
    return img_id


def encode(texts, max_len=MAX_LEN):
    # The real normalization pipeline, not a single danda replace. Must be
    # applied to captions, style corpora, AND the reference side of any metric
    # you compute later, or your BLEU is measuring the wrong thing.
    texts = [normalize(t) for t in texts]
    enc = tokenizer(texts, max_length=max_len, truncation=True,
                    padding=True, return_tensors="pt")
    return enc.input_ids, enc.attention_mask


def to_labels(input_ids):
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return labels


def _split_line(line):
    """`1000092795.jpg#0<TAB>caption` -> ('1000092795', 'caption').

    Split on the tab, not on a `#\\d*` regex. The regex also fired on a bare
    '#' and on any '#' inside the caption itself, silently truncating it.
    """
    if '\t' in line:
        key, cap = line.split('\t', 1)
    elif '#' in line:
        key, cap = line.split('#', 1)
        cap = cap.split(None, 1)[1] if len(cap.split(None, 1)) > 1 else ''
    else:
        return None
    return strip_ext(key.split('#')[0].strip()), cap.strip()


class FactualDataset(Dataset):
    """One row per (image, caption) pair. No epoch state, no rotation.

    The previous version returned one row per IMAGE and rotated which caption
    was used via set_epoch(). Under persistent_workers=True the workers fork
    once and keep their own copy of the dataset, so the parent's set_epoch()
    never reached them — self.epoch stayed 0 forever and only caption #0 of
    each image was ever trained on. Four fifths of the caption data was loaded,
    counted in the INFO line, and discarded.

    The caption is returned as text as well, because V2L encodes it on the
    text side.
    """

    def __init__(self, cache_dir, caption_file):
        self.cache_dir = cache_dir
        self.rows = self._load(caption_file)

    def _load(self, caption_file):
        with open(caption_file, encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        rows, seen, malformed = [], {}, 0
        for line in lines:
            parsed = _split_line(line)
            if parsed is None or not parsed[1]:
                malformed += 1
                continue
            img_id, caption = parsed
            if img_id not in seen:
                seen[img_id] = os.path.exists(
                    os.path.join(self.cache_dir, f"{img_id}.pt"))
            if seen[img_id]:
                rows.append((img_id, caption))

        missing = sum(1 for v in seen.values() if not v)
        n_img = sum(1 for v in seen.values() if v)
        if malformed:
            print(f"[WARN] {caption_file}: {malformed} malformed lines.")
        if missing:
            print(f"[WARN] {caption_file}: {missing} images without cached features.")
        if not rows:
            raise RuntimeError(f"No usable samples in {caption_file}")

        print(f"[INFO] {os.path.basename(caption_file)}: {n_img} images, "
              f"{len(rows)} captions ({len(rows)/max(n_img,1):.1f} per image).")
        return rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, ix):
        img_id, caption = self.rows[ix]
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
        self.seed = seed
        print(f"[INFO] {os.path.basename(caption_file)}: {len(self.lines)} lines")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, ix):
        # Seed per item instead of holding one random.Random on the dataset.
        # Every worker forks with an identical RNG state, so a shared generator
        # produced duplicated corruption across workers and was not reproducible
        # run to run. Keying on ix makes the corruption a deterministic function
        # of the item, identical in every worker.
        rng = random.Random(self.seed * 1_000_003 + ix)
        text = self.lines[ix]
        words = text.split()
        kept = [w for w in words if rng.random() > self.drop_p]
        if not kept:
            kept = words[:1]
        return " ".join(kept), text


def collate_style(batch):
    corrupted, clean = zip(*batch)
    c_ids, c_mask = encode(corrupted)
    t_ids, _ = encode(clean)
    return c_ids, c_mask, to_labels(t_ids)


class SubsetEpochSampler(torch.utils.data.Sampler):
    """OPTIONAL. Off unless --captions_per_epoch > 0.

    Draws `num_samples` rows per epoch from a fresh permutation, purely to make
    epochs shorter. Every caption is still reachable because the permutation is
    redrawn each epoch.

    NOTE this set_epoch DOES work under persistent_workers, unlike the one that
    used to live on the dataset. The sampler runs in the MAIN process and ships
    indices to the workers; only the dataset object is copied into them.
    """

    def __init__(self, n, num_samples=None, seed=0):
        self.n = n
        self.num_samples = min(num_samples or n, n)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(self.n, generator=g).tolist()[:self.num_samples]

    def __len__(self):
        return self.num_samples


def factual_loader(cache_dir, caption_file, bs, shuffle=False, workers=4,
                   captions_per_epoch=0):
    ds = FactualDataset(cache_dir, caption_file)
    sampler = None
    if shuffle and captions_per_epoch and captions_per_epoch < len(ds):
        sampler = SubsetEpochSampler(len(ds), captions_per_epoch)
        shuffle = False                       # mutually exclusive with sampler
        print(f"[INFO] sampling {len(sampler)}/{len(ds)} captions per epoch")
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, sampler=sampler,
                      num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0,
                      collate_fn=collate_factual)


def style_loader(caption_file, bs, shuffle=True, workers=2):
    return DataLoader(StyleTextDataset(caption_file),
                      batch_size=bs, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0,
                      collate_fn=collate_style)
