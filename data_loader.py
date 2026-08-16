import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

T5_CKPT = "csebuetnlp/banglat5"
MAX_LEN = 48

# No bos_token. T5 shifts labels right with decoder_start_token_id =
# pad_token_id, which is what BanglaT5 was pretrained with.
tokenizer = T5Tokenizer.from_pretrained(T5_CKPT, use_fast=False)


def strip_ext(img_id):
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        if img_id.lower().endswith(ext.lower()):
            return img_id[: -len(ext)]
    return img_id


def encode_captions(captions):
    """Raw strings -> labels with -100 on padding. T5Tokenizer appends </s>.
    Danda U+09F7 is mapped to U+0964, which the tokenizer round-trips."""
    texts = [c.replace("\u09f7", "\u0964") for c in captions]
    enc = tokenizer(texts, max_length=MAX_LEN, truncation=True,
                    padding=True, return_tensors="pt")
    labels = enc.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return labels


class FactualDataset(Dataset):
    """Image-caption pairs from precomputed fp16 ViT features."""
    def __init__(self, cache_dir, caption_file):
        self.cache_dir = cache_dir
        self.items = self._load(caption_file)

    def _load(self, caption_file):
        with open(caption_file, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        items, imgs = [], set()
        r = re.compile(r'#\d*')
        missing, malformed = 0, 0

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
            print(f"[WARN] {caption_file}: {malformed} malformed lines skipped.")
        if missing:
            print(f"[WARN] {caption_file}: {missing} lines dropped, no cached features.")
        if not items:
            raise RuntimeError(f"[FactualDataset] No valid samples in {caption_file}.")

        print(f"[INFO] {os.path.basename(caption_file)}: {len(items)} captions, "
              f"{len(imgs)} images.")
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, ix):
        img_id, caption = self.items[ix]
        raw = torch.load(os.path.join(self.cache_dir, f"{img_id}.pt"),
                         map_location='cpu')
        return raw, caption


class StyledTextDataset(Dataset):
    """Monolingual stylized text. No images — the StyleNet premise."""
    def __init__(self, caption_file):
        with open(caption_file, 'r', encoding='utf-8') as f:
            self.captions = [x.strip() for x in f if x.strip()]
        if not self.captions:
            raise RuntimeError(f"[StyledTextDataset] {caption_file} is empty.")
        print(f"[INFO] {os.path.basename(caption_file)}: {len(self.captions)} lines.")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, ix):
        return self.captions[ix]


def collate_factual(batch):
    raw_feats, captions = zip(*batch)
    return torch.stack(raw_feats, 0), encode_captions(captions)


def collate_styled(captions):
    return encode_captions(captions)


def get_data_loader(cache_dir, caption_file, batch_size, shuffle=False, num_workers=4):
    return DataLoader(FactualDataset(cache_dir, caption_file),
                      batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0,
                      prefetch_factor=4 if num_workers > 0 else None,
                      collate_fn=collate_factual)


def get_styled_data_loader(caption_file, batch_size, shuffle=False, num_workers=2):
    return DataLoader(StyledTextDataset(caption_file),
                      batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0,
                      collate_fn=collate_styled)
