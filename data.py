"""
data.py — datasets for both models.

FactualDataset   (image feature, caption) for the factual captioner.
StyleDataset     (CLIP text embedding, style, sentence) for the style model.
                 No images. No pairing with factual captions. Nothing is
                 corrupted, masked, or deleted.
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


def build_tokenizer(ckpt=T5_CKPT):
    from models import add_style_tokens
    tok = T5Tokenizer.from_pretrained(ckpt, use_fast=False)
    ids = add_style_tokens(tok)
    return tok, ids


def read_lines(path):
    return [l.strip() for l in open(path, encoding='utf-8') if l.strip()]


def strip_ext(img_id):
    for e in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        if img_id.lower().endswith(e.lower()):
            return img_id[:-len(e)]
    return img_id


def split_line(line):
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


class Encoder:
    """Tokenisation, kept in one place so the factual and style models and the
    evaluation references all go through the identical pipeline."""

    def __init__(self, tokenizer, max_len=MAX_LEN):
        self.tok, self.max_len = tokenizer, max_len

    def __call__(self, texts):
        texts = [normalize(t) for t in texts]
        enc = self.tok(texts, max_length=self.max_len, truncation=True,
                       padding=True, return_tensors="pt")
        return enc.input_ids, enc.attention_mask

    def labels(self, ids):
        lab = ids.clone()
        lab[lab == self.tok.pad_token_id] = -100
        return lab


# ------------------------------------------------------------------ factual
class FactualDataset(Dataset):
    """One row per (image, caption) pair.

    Rows for the same image are CONSECUTIVE — about 4.8 of them. Anything
    that slices [:3] off a batch gets one picture three times; see the
    `distinct_images` helper below.
    """

    def __init__(self, cache_dir, caption_file):
        self.cache_dir = cache_dir
        self.rows = self._load(caption_file)

    def _load(self, caption_file):
        rows, seen, malformed = [], {}, 0
        for line in read_lines(caption_file):
            parsed = split_line(line)
            if parsed is None or not parsed[1]:
                malformed += 1
                continue
            img, cap = parsed
            if img not in seen:
                seen[img] = os.path.exists(os.path.join(self.cache_dir, f"{img}.pt"))
            if seen[img]:
                rows.append((img, cap))
        miss = sum(1 for v in seen.values() if not v)
        nimg = sum(1 for v in seen.values() if v)
        if malformed:
            print(f"[WARN] {caption_file}: {malformed} malformed lines")
        if miss:
            print(f"[WARN] {caption_file}: {miss} images without cached features")
        if not rows:
            raise RuntimeError(f"No usable samples in {caption_file}")
        print(f"[INFO] {os.path.basename(caption_file)}: {nimg} images, "
              f"{len(rows)} captions ({len(rows)/max(nimg,1):.1f} per image)")
        return rows

    def distinct_images(self, k):
        """Feature tensors for the first k DISTINCT images."""
        seen, out = set(), []
        for i, (img, _) in enumerate(self.rows):
            if img in seen:
                continue
            seen.add(img)
            out.append(self[i][0])
            if len(out) == k:
                break
        return torch.stack(out)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, ix):
        img, cap = self.rows[ix]
        feats = torch.load(os.path.join(self.cache_dir, f"{img}.pt"),
                           map_location='cpu')            # [50, 768] fp16
        return feats, cap


def factual_loader(cache_dir, caption_file, encoder, bs, shuffle=False,
                   workers=4, return_dataset=False):
    ds = FactualDataset(cache_dir, caption_file)

    def collate(batch):
        feats, caps = zip(*batch)
        ids, mask = encoder(list(caps))
        return torch.stack(feats, 0), ids, mask, encoder.labels(ids)

    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
                    pin_memory=True, persistent_workers=workers > 0,
                    collate_fn=collate)
    return (dl, ds) if return_dataset else dl


# ------------------------------------------------------------------ style
class StyleDataset(Dataset):
    """Sentences from the style corpus and from the factual text corpus, each
    with its CLIP TEXT embedding and its style label.

    Two corpora, not one. The style corpus alone gives the model nothing to
    distinguish — the discriminative half of Eq. 8 needs a contrasting class,
    and on FlickrStyle10k the paper uses factual as the undesired style for
    both romantic and humorous.
    """

    def __init__(self, style_pt, factual_pt, style_name, style_ids):
        s = torch.load(style_pt, map_location='cpu')
        f = torch.load(factual_pt, map_location='cpu')
        for d, p in ((s, style_pt), (f, factual_pt)):
            if d.get('kind') != 'text':
                raise RuntimeError(f"{p} is not a text-embedding file")
        if s['emb'].shape[1] != f['emb'].shape[1]:
            raise RuntimeError("style and factual embeddings have different dims")

        self.dim = s['emb'].shape[1]
        self.rows = (
            [(s['emb'][i], s['lines'][i], style_name) for i in range(len(s['lines']))] +
            [(f['emb'][i], f['lines'][i], 'factual') for i in range(len(f['lines']))]
        )
        self.style_ids = style_ids
        self.style_name = style_name
        print(f"[INFO] style set: {len(s['lines'])} {style_name} + "
              f"{len(f['lines'])} factual = {len(self.rows)} rows, dim {self.dim}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, ix):
        emb, text, style = self.rows[ix]
        other = 'factual' if style != 'factual' else self.style_name
        return emb, text, self.style_ids[style], self.style_ids[other]


def style_loader(style_pt, factual_pt, style_name, style_ids, encoder, bs,
                 shuffle=True, workers=2):
    ds = StyleDataset(style_pt, factual_pt, style_name, style_ids)

    def collate(batch):
        emb, text, true_id, other_id = zip(*batch)
        ids, _ = encoder(list(text))
        return (torch.stack(emb, 0),
                torch.tensor(true_id, dtype=torch.long),
                torch.tensor(other_id, dtype=torch.long),
                encoder.labels(ids))

    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
                    pin_memory=True, persistent_workers=workers > 0,
                    collate_fn=collate)
    return dl, ds.dim
