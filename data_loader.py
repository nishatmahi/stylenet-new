import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

# Load HuggingFace Bangla tokenizer (Stylenet adaptation)
tokenizer = AutoTokenizer.from_pretrained("hishab/titulm-mpt-1b-v2.0", trust_remote_code=True)

# Rescale utility
class Rescale:
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

# Standard image transform
image_transform = transforms.Compose([
    Rescale((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Multi-extension image finder (jpg/jpeg/png)
def find_image_with_any_ext(img_folder, img_id):
    for ext in ['jpg', 'jpeg', 'png']:
        candidate = os.path.join(img_folder, f"{img_id}.{ext}")
        if os.path.exists(candidate):
            return candidate
    return None

# Extension stripper
def strip_ext(img_id):
    for ext in ['.jpg', '.jpeg', '.png']:
        if img_id.lower().endswith(ext):
            return img_id[: -len(ext)]
    return img_id

# Caption dataset (image+caption)
class BanglaCaptionDataset(Dataset):
    def __init__(self, img_paths, captions, transform=None):
        self.img_paths = img_paths
        self.captions = captions
        self.transform = transform if transform else image_transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, ix):
        img_path = self.img_paths[ix]
        caption = self.captions[ix]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        encoding = tokenizer(
            caption,
            truncation=True,
            padding='max_length',
            max_length=32,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attn_mask = encoding["attention_mask"].squeeze(0)
        return image, input_ids, attn_mask

# Robust caption+image loader (handles tab/multi-space/single space, multi-ext image, ext stripping)
def load_img_caption_lists(data_txt_file, image_folder):
    img_paths = []
    captions = []
    with open(data_txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r'\s{2,}|\t', line, maxsplit=1)
            if len(parts) < 2:
                parts = line.split(' ', 1)
                if len(parts) < 2:
                    continue
            id_and_idx, caption = parts
            img_id = id_and_idx.split('#')[0].strip()
            img_id = strip_ext(img_id)  # Always strip ext if present
            img_path = find_image_with_any_ext(image_folder, img_id)
            if img_path is None:
                print(f"Warning: {img_id} not found in {image_folder}, skipping")
                continue
            img_paths.append(img_path)
            captions.append(caption.strip())
    print(f"Loaded {len(img_paths)} images and {len(captions)} captions.")
    return img_paths, captions

# Collate: batch, pad, sort
def collate_fn(data):
    data.sort(key=lambda x: (x[1] != tokenizer.pad_token_id).sum(), reverse=True)
    images, input_ids, attn_masks = zip(*data)
    images = torch.stack(images, 0)
    input_ids = torch.stack(input_ids, 0)
    attn_masks = torch.stack(attn_masks, 0)
    lengths = torch.LongTensor([(ids != tokenizer.pad_token_id).sum().item() for ids in input_ids])
    return images, input_ids, attn_masks, lengths

# Loader
def get_loader(img_paths, captions, batch_size=32, shuffle=True, num_workers=2):
    dataset = BanglaCaptionDataset(img_paths, captions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

# Styled caption dataset (no image)
class BanglaStyledCaptionDataset(Dataset):
    def __init__(self, captions):
        self.captions = captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, ix):
        caption = self.captions[ix]
        encoding = tokenizer(
            caption,
            truncation=True,
            padding='max_length',
            max_length=32,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attn_mask = encoding["attention_mask"].squeeze(0)
        return input_ids, attn_mask

def get_styled_loader(captions, batch_size=64, shuffle=True, num_workers=2):
    dataset = BanglaStyledCaptionDataset(captions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Manual test block (clean, no unnecessary args)
if __name__ == "__main__":
    img_paths, captions = load_img_caption_lists(
        data_txt_file="your_factual_captions.txt",
        image_folder="your_image_folder"
    )
    loader = get_loader(img_paths, captions, batch_size=3)
    for i, (images, input_ids, attn_mask, lengths) in enumerate(loader):
        print("Batch:", i)
        print("Image batch:", images.shape)
        print("Input_ids:", input_ids.shape)
        print("Attention mask:", attn_mask.shape)
        print("Lengths:", lengths)
        if i == 2: break
