import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

# ---- Bangla HuggingFace Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained("hishab/titulm-mpt-1b-v2.0", trust_remote_code=True)

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

# ---- Image transforms ----
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

class Flickr7kBanglaDataset(Dataset):
    '''Flickr7k-style dataset for Bangla image-caption'''
    def __init__(self, img_dir, caption_file, tokenizer, transform=None):
        self.img_dir = img_dir
        self.imgname_caption_list = self._get_imgname_and_caption(caption_file)
        self.tokenizer = tokenizer
        self.transform = transform if transform else image_transform

    def _get_imgname_and_caption(self, caption_file):
        with open(caption_file, 'r', encoding='utf-8') as f:
            res = f.readlines()
        imgname_caption_list = []
        r = re.compile(r'#\d*')
        for line in res:
            img_and_cap = r.split(line)
            img_and_cap = [x.strip() for x in img_and_cap]
            imgname_caption_list.append(img_and_cap)
        return imgname_caption_list

    def __len__(self):
        return len(self.imgname_caption_list)

    def __getitem__(self, ix):
        img_name = self.imgname_caption_list[ix][0]
        img_id = strip_ext(img_name)
        img_path = find_image_with_any_ext(self.img_dir, img_id)
        caption = self.imgname_caption_list[ix][1]

        # Robust image loading (RGB, RGBA, Grayscale)
        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"[ERROR] Could not open image: {img_path}, {e}")
            image = Image.new("RGB", (224, 224))
        if image.mode == "RGBA" or image.mode == "LA":
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        # HuggingFace tokenizer, pad/truncate, BOS/EOS auto (add_special_tokens=True)
        tokens = self.tokenizer(
            caption,
            truncation=True,
            padding='max_length',
            max_length=32,
            return_tensors="pt",
            add_special_tokens=True
        )
        input_ids = tokens["input_ids"].squeeze(0)
        return image, input_ids

class FlickrStyle7kBanglaDataset(Dataset):
    '''Styled caption dataset'''
    def __init__(self, caption_file, tokenizer):
        self.caption_list = self._get_caption(caption_file)
        self.tokenizer = tokenizer

    def _get_caption(self, caption_file):
        with open(caption_file, 'r', encoding='utf-8') as f:
            caption_list = f.readlines()
        caption_list = [x.strip() for x in caption_list]
        return caption_list

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, ix):
        caption = self.caption_list[ix]
        tokens = self.tokenizer(
            caption,
            truncation=True,
            padding='max_length',
            max_length=32,
            return_tensors="pt",
            add_special_tokens=True
        )
        input_ids = tokens["input_ids"].squeeze(0)
        return input_ids

def collate_fn(data):
    data.sort(key=lambda x: (x[1] != tokenizer.pad_token_id).sum(), reverse=True)
    images, input_ids = zip(*data)
    images = torch.stack(images, 0)
    input_ids = torch.stack(input_ids, 0)
    lengths = torch.LongTensor([(ids != tokenizer.pad_token_id).sum().item() for ids in input_ids])
    return images, input_ids, lengths

def collate_fn_styled(captions):
    captions = list(captions)
    captions.sort(key=lambda x: (x != tokenizer.pad_token_id).sum(), reverse=True)
    lengths = torch.LongTensor([(cap != tokenizer.pad_token_id).sum().item() for cap in captions])
    captions = torch.stack(captions, 0)
    return captions, lengths

def get_data_loader(img_dir, caption_file, batch_size, shuffle=False, num_workers=0):
    dataset = Flickr7kBanglaDataset(img_dir, caption_file, tokenizer, transform=image_transform)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return data_loader

def get_styled_data_loader(caption_file, batch_size, shuffle=False, num_workers=0):
    dataset = FlickrStyle7kBanglaDataset(caption_file, tokenizer)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=collate_fn_styled)
    return data_loader

# ==== Test/debug block ====
if __name__ == "__main__":
    img_dir = "./your_image_folder"
    factual_file = "./your_factual_captions.txt"
    humorous_file = "./your_humorous_captions.txt"
    romantic_file = "./your_romantic_captions.txt"

    data_loader = get_data_loader(img_dir, factual_file, batch_size=3)
    for i, (images, input_ids, lengths) in enumerate(data_loader):
        print(f"Batch: {i}", images.shape, input_ids.shape, lengths)
        if i == 2: break

    styled_loader_humorous = get_styled_data_loader(humorous_file, batch_size=3)
    for i, (captions, lengths) in enumerate(styled_loader_humorous):
        print(f"Humorous batch: {i}", captions.shape, lengths)
        if i == 2: break

    styled_loader_romantic = get_styled_data_loader(romantic_file, batch_size=3)
    for i, (captions, lengths) in enumerate(styled_loader_romantic):
        print(f"Romantic batch: {i}", captions.shape, lengths)
        if i == 2: break
