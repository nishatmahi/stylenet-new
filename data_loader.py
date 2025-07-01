import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

# HuggingFace Bangla tokenizer (change here if you want another model)
tokenizer = AutoTokenizer.from_pretrained("hishab/titulm-mpt-1b-v2.0")

# Rescale utility (as in original code)
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

# Standard transform
image_transform = transforms.Compose([
    Rescale((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom Bangla Dataset (original style, factual or styled captions)
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
        # Image open and preprocess
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        # Tokenize caption (no <s>, </s> manual)
        encoding = tokenizer(
            caption,
            truncation=True,
            padding='max_length',
            max_length=32,  # Or your desired
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)    # [seq_len]
        attn_mask = encoding["attention_mask"].squeeze(0)
        return image, input_ids, attn_mask

# Loader for flexible tab/space separated files (factual or stylized)
def load_img_caption_lists(data_txt_file, image_folder, img_ext="jpg"):
    img_paths = []
    captions = []
    with open(data_txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # Robust: split by tab or 2+ spaces, fallback single space
            parts = re.split(r'\s{2,}|\t', line, maxsplit=1)
            if len(parts) < 2:
                parts = line.split(' ', 1)
                if len(parts) < 2:
                    continue
            id_and_idx, caption = parts
            img_id = id_and_idx.split('#')[0]
            if not (img_id.endswith('.jpg') or img_id.endswith('.jpeg') or img_id.endswith('.png')):
                img_id = img_id + '.' + img_ext
            img_path = os.path.join(image_folder, img_id)
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} not found, skipping")
                continue
            img_paths.append(img_path)
            captions.append(caption.strip())
    print(f"Loaded {len(img_paths)} images and {len(captions)} captions.")
    return img_paths, captions

# Collate function (original style) — batch padding
def collate_fn(data):
    data.sort(key=lambda x: (x[1] != tokenizer.pad_token_id).sum(), reverse=True)
    images, input_ids, attn_masks = zip(*data)
    images = torch.stack(images, 0)
    input_ids = torch.stack(input_ids, 0)
    attn_masks = torch.stack(attn_masks, 0)
    lengths = torch.LongTensor([(ids != tokenizer.pad_token_id).sum().item() for ids in input_ids])
    return images, input_ids, attn_masks, lengths

# Loader utility for train/test loop
def get_loader(img_paths, captions, batch_size=32, shuffle=True, num_workers=2):
    dataset = BanglaCaptionDataset(img_paths, captions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

# Styled captions (only caption, no image needed)
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

# For manual testing
if __name__ == "__main__":
    # Example usage (set your file paths here)
    img_paths, captions = load_img_caption_lists(
        data_txt_file="your_factual_captions.txt",
        image_folder="your_image_folder",
        img_ext="jpg"
    )
    loader = get_loader(img_paths, captions, batch_size=3)
    for i, (images, input_ids, attn_mask, lengths) in enumerate(loader):
        print("Batch:", i)
        print("Image batch:", images.shape)
        print("Input_ids:", input_ids.shape)
        print("Attention mask:", attn_mask.shape)
        print("Lengths:", lengths)
        if i == 2: break
