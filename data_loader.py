import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hishab/titulm-mpt-1b-v2.0", trust_remote_code=True)

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

image_transform = transforms.Compose([
    Rescale((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def find_image_with_any_ext(img_folder, img_id):
    for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
        candidate = os.path.join(img_folder, f"{img_id}.{ext}")
        if os.path.exists(candidate):
            return candidate
    return None

def strip_ext(img_id):
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        if img_id.lower().endswith(ext.lower()):
            return img_id[: -len(ext)]
    return img_id

# FACTUAL: image+caption (will check for image)
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
            caption, truncation=True, padding='max_length',
            max_length=32, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attn_mask = encoding["attention_mask"].squeeze(0)
        return image, input_ids, attn_mask

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
            img_id = strip_ext(img_id)
            img_path = find_image_with_any_ext(image_folder, img_id)
            if img_path is None:
                continue  # warning remove: silently skip if image not found
            img_paths.append(img_path)
            captions.append(caption.strip())
    print(f"Loaded {len(img_paths)} images and {len(captions)} factual captions.")
    return img_paths, captions

def collate_fn(data):
    data.sort(key=lambda x: (x[1] != tokenizer.pad_token_id).sum(), reverse=True)
    images, input_ids, attn_masks = zip(*data)
    images = torch.stack(images, 0)
    input_ids = torch.stack(input_ids, 0)
    attn_masks = torch.stack(attn_masks, 0)
    lengths = torch.LongTensor([(ids != tokenizer.pad_token_id).sum().item() for ids in input_ids])
    return images, input_ids, attn_masks, lengths

def get_loader(img_paths, captions, batch_size=32, shuffle=True, num_workers=1):
    dataset = BanglaCaptionDataset(img_paths, captions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

# STYLED: caption only, NO image check, for humor/romantic style
class BanglaStyledCaptionDataset(Dataset):
    def __init__(self, captions):
        self.captions = captions
    def __len__(self):
        return len(self.captions)
    def __getitem__(self, ix):
        caption = self.captions[ix]
        encoding = tokenizer(
            caption, truncation=True, padding='max_length',
            max_length=32, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attn_mask = encoding["attention_mask"].squeeze(0)
        return input_ids, attn_mask

def load_styled_caption_list(data_txt_file):
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
            # Always take only caption part
            _, caption = parts
            captions.append(caption.strip())
    print(f"Loaded {len(captions)} styled captions.")
    return captions

def get_styled_loader(captions, batch_size=64, shuffle=True, num_workers=2):
    dataset = BanglaStyledCaptionDataset(captions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Manual test (optional)
if __name__ == "__main__":
    # Factual test
    img_paths, captions = load_img_caption_lists(
        data_txt_file="your_factual_captions.txt",
        image_folder="your_image_folder"
    )
    loader = get_loader(img_paths, captions, batch_size=3)
    for i, (images, input_ids, attn_mask, lengths) in enumerate(loader):
        print("Batch:", i)
        if i == 2: break

    # Styled test
    styled_captions = load_styled_caption_list("your_humorous_captions.txt")
    styled_loader = get_styled_loader(styled_captions, batch_size=3)
    for i, (input_ids, attn_mask) in enumerate(styled_loader):
        print("Styled batch:", i)
        if i == 2: break
