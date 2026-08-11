import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

T5_CKPT = "csebuetnlp/banglat5"
tokenizer = T5Tokenizer.from_pretrained(T5_CKPT, use_fast=False)

if tokenizer.bos_token is None:
    tokenizer.add_special_tokens({"bos_token": "<s>"})


def strip_ext(img_id):
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        if img_id.lower().endswith(ext.lower()):
            return img_id[: -len(ext)]
    return img_id


def find_cache_file(cache_dir, img_id):
    candidate = os.path.join(cache_dir, f"{img_id}.pt")
    return candidate if os.path.exists(candidate) else None


class Flickr7kBanglaDataset(Dataset):
    """
    Loads PRECOMPUTED raw ViT features (cached as float16 to fit disk
    quota) from cache_dir instead of raw images. Run the caching script
    once before training.
    """
    def __init__(self, cache_dir, caption_file):
        self.cache_dir = cache_dir
        self.imgname_caption_list = self._get_imgname_and_caption(caption_file)

    def _get_imgname_and_caption(self, caption_file):
        with open(caption_file, 'r', encoding='utf-8') as f:
            res = [ln.strip() for ln in f if ln.strip()]

        imgname_caption_list = []
        r = re.compile(r'#\d*')

        missing, malformed = 0, 0
        for line in res:
            parts = [x.strip() for x in r.split(line) if x.strip()]
            if len(parts) < 2:
                malformed += 1
                continue

            img_name, caption = parts[0], parts[1]
            img_id = strip_ext(img_name)
            cache_path = find_cache_file(self.cache_dir, img_id)

            if cache_path is None:
                missing += 1
                continue

            imgname_caption_list.append((cache_path, caption))

        if malformed > 0:
            print(f"[WARN] Skipped {malformed} malformed caption lines (no/invalid caption).")
        if missing > 0:
            print(f"[WARN] Dropped {missing} samples — no cached ViT features found. "
                  f"Run the caching script first if this number is unexpectedly high.")

        if len(imgname_caption_list) == 0:
            raise RuntimeError("[Flickr7kBanglaDataset] No valid samples found after filtering.")

        return imgname_caption_list

    def __len__(self):
        return len(self.imgname_caption_list)

    def __getitem__(self, ix):
        cache_path, caption = self.imgname_caption_list[ix]
        raw_features = torch.load(cache_path).float()   # cast fp16 cache back to fp32
        return raw_features, caption


class FlickrStyle7kBanglaDataset(Dataset):
    '''Styled caption dataset — unaffected by caching, no images involved.'''
    def __init__(self, caption_file):
        self.caption_list = self._get_caption(caption_file)

    def _get_caption(self, caption_file):
        with open(caption_file, 'r', encoding='utf-8') as f:
            caption_list = f.readlines()
        caption_list = [x.strip() for x in caption_list]
        return caption_list

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, ix):
        return self.caption_list[ix]


def collate_fn(batch):
    raw_feats, captions = zip(*batch)
    raw_feats = torch.stack(raw_feats, 0)   # [B, N_patches+1, vit_hidden]

    ids_list = []
    for cap in captions:
        ids = tokenizer.encode(cap, add_special_tokens=False)
        ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
        ids_list.append(torch.tensor(ids, dtype=torch.long))
    max_len = max(len(ids) for ids in ids_list)
    padded = [torch.cat([ids, torch.full((max_len - len(ids),), tokenizer.pad_token_id, dtype=torch.long)]) for ids in ids_list]
    input_ids = torch.stack(padded, 0)
    lengths = torch.tensor([len(ids) for ids in ids_list], dtype=torch.long)
    return raw_feats, input_ids, lengths


def collate_fn_styled(captions):
    ids_list = []
    for cap in captions:
        ids = tokenizer.encode(cap, add_special_tokens=False)
        ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
        ids_list.append(torch.tensor(ids, dtype=torch.long))
    max_len = max(len(ids) for ids in ids_list)
    padded = [torch.cat([ids, torch.full((max_len - len(ids),), tokenizer.pad_token_id, dtype=torch.long)]) for ids in ids_list]
    input_ids = torch.stack(padded, 0)
    lengths = torch.tensor([len(ids) for ids in ids_list], dtype=torch.long)
    return input_ids, lengths


def get_data_loader(cache_dir, caption_file, batch_size, shuffle=False, num_workers=4):
    dataset = Flickr7kBanglaDataset(cache_dir, caption_file)
    data_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              pin_memory=True,
                              collate_fn=collate_fn)
    return data_loader


def get_styled_data_loader(caption_file, batch_size, shuffle=False, num_workers=4):
    dataset = FlickrStyle7kBanglaDataset(caption_file)
    data_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              pin_memory=True,
                              collate_fn=collate_fn_styled)
    return data_loader
