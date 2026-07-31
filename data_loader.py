"""
data_loader.py — DataLoaders for StyleNet Transformer training.

Two dataset types:
  1) Flickr7kPrecomputedDataset  — factual captions with precomputed ViT features
  2) FlickrStyle7kBanglaDataset  — styled captions (text-only, no images)

Tokenizer: MT5Tokenizer (SentencePiece, covers Bengali natively).
"""

import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

# ---- Pretrained mT5 tokenizer (SentencePiece, covers Bengali) ----
tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')

# NOTE: mT5's tokenizer has no bos_token_id (it's None by design for T5-family
# models). T5 conventionally uses pad_token_id as the decoder "start" token
# instead of BOS. We use pad_token_id here in place of your old bos_token_id.
DECODER_START_ID = tokenizer.pad_token_id

# ---- Image transform (used only by extract_features.py) ----
class Rescale:
    '''Rescale the image to a given size'''
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        w, h = image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = int(self.output_size * h / w), self.output_size
            else:
                new_h, new_w = self.output_size, int(self.output_size * w / h)
        else:
            new_h, new_w = self.output_size
        image = image.resize((new_w, new_h))
        return image

image_transform = transforms.Compose([
    Rescale((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ---- Helper functions ----
def strip_ext(img_id):
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        if img_id.lower().endswith(ext.lower()):
            return img_id[: -len(ext)]
    return img_id


# ========================================================================
#  Dataset 1:  Factual captions with PRECOMPUTED ViT features
# ========================================================================
class Flickr7kPrecomputedDataset(Dataset):
    """Pairs precomputed ViT features with captions.

    Instead of loading images from disk and running ViT forward on every batch,
    this dataset looks up pre-extracted CLS features by image ID.
    Saves ~2.85 GB VRAM and speeds up training ~30-40%.
    """
    def __init__(self, features_dict, caption_file):
        """
        Args:
            features_dict: dict  {img_id_without_ext: tensor[vit_dim]}
            caption_file:  path to caption file (format: img_name#N caption_text)
        """
        self.features_dict = features_dict
        self.samples = self._parse_captions(caption_file)

    def _parse_captions(self, caption_file):
        with open(caption_file, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        r = re.compile(r'#\d*')
        samples = []
        missing, malformed = 0, 0

        for line in lines:
            parts = [x.strip() for x in r.split(line) if x.strip()]
            if len(parts) < 2:
                malformed += 1
                continue

            img_name, caption = parts[0], parts[1]
            img_id = strip_ext(img_name)

            if img_id not in self.features_dict:
                missing += 1
                continue

            samples.append((img_id, caption))

        if malformed > 0:
            print(f"[WARN] Skipped {malformed} malformed caption lines.")
        if missing > 0:
            print(f"[WARN] Dropped {missing} samples — image ID not in features dict.")
        if len(samples) == 0:
            raise RuntimeError("[Flickr7kPrecomputedDataset] No valid samples after filtering.")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, ix):
        img_id, caption = self.samples[ix]
        feature = self.features_dict[img_id]  # tensor[vit_dim], already float32
        return feature, caption


# ========================================================================
#  Dataset 2:  Styled captions (text only — no images)
# ========================================================================
class FlickrStyle7kBanglaDataset(Dataset):
    '''Styled caption dataset (romantic / humorous — text only, no images)'''
    def __init__(self, caption_file):
        self.caption_list = self._get_caption(caption_file)

    def _get_caption(self, caption_file):
        with open(caption_file, 'r', encoding='utf-8') as f:
            caption_list = f.readlines()
        caption_list = [x.strip() for x in caption_list if x.strip()]
        return caption_list

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, ix):
        return self.caption_list[ix]


# ========================================================================
#  Collate functions
# ========================================================================
# NOTE: decoder_input_ids start with DECODER_START_ID (pad token, per mT5
# convention) instead of bos_token_id. labels use -100 for padded positions
# so mT5's internal loss computation ignores them (standard HF convention).

def _encode_batch(captions):
    ids_list = []
    for cap in captions:
        ids = tokenizer.encode(cap, add_special_tokens=False)
        ids = ids + [tokenizer.eos_token_id]
        ids_list.append(torch.tensor(ids, dtype=torch.long))

    max_len = max(len(ids) for ids in ids_list)

    # labels: pad with -100 (ignored in loss)
    labels = [torch.cat([ids, torch.full((max_len - len(ids),), -100, dtype=torch.long)]) for ids in ids_list]
    labels = torch.stack(labels, 0)

    # decoder_input_ids: shifted right, start token = pad_token_id, pad with pad_token_id
    decoder_input_ids = torch.full((len(ids_list), max_len), tokenizer.pad_token_id, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        decoder_input_ids[i, 0] = DECODER_START_ID
        if len(ids) > 1:
            decoder_input_ids[i, 1:len(ids)] = ids[:-1]

    lengths = torch.tensor([len(ids) for ids in ids_list], dtype=torch.long)
    return decoder_input_ids, labels, lengths


def collate_fn_precomputed(batch):
    """Collate for factual data with precomputed ViT features."""
    features, captions = zip(*batch)
    features = torch.stack(features, 0)       # [batch, vit_dim]
    decoder_input_ids, labels, lengths = _encode_batch(captions)
    return features, decoder_input_ids, labels, lengths


def collate_fn_styled(captions):
    """Collate for styled data (text-only, no images)."""
    decoder_input_ids, labels, lengths = _encode_batch(captions)
    return decoder_input_ids, labels, lengths


# ========================================================================
#  DataLoader factories
# ========================================================================
def get_precomputed_data_loader(features_dict, caption_file, batch_size,
                                 shuffle=False, num_workers=2):
    """DataLoader for factual training with precomputed ViT features."""
    dataset = Flickr7kPrecomputedDataset(features_dict, caption_file)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_fn_precomputed,
                      pin_memory=True, persistent_workers=True)


def get_styled_data_loader(caption_file, batch_size, shuffle=False, num_workers=2):
    """DataLoader for styled training (text-only captions)."""
    dataset = FlickrStyle7kBanglaDataset(caption_file)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_fn_styled,
                      pin_memory=True, persistent_workers=True)
