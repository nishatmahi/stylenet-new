import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import MT5Tokenizer

# ---- Pretrained mT5 tokenizer (SentencePiece, covers Bengali) ----
tokenizer = MT5Tokenizer.from_pretrained('google/mt5-base')

# NOTE: mT5's tokenizer has no bos_token_id (it's None by design for T5-family
# models). T5 conventionally uses pad_token_id as the decoder "start" token
# instead of BOS. We use pad_token_id here in place of your old bos_token_id.
DECODER_START_ID = tokenizer.pad_token_id

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
    def __init__(self, img_dir, caption_file, transform=None):
        self.img_dir = img_dir
        self.imgname_caption_list = self._get_imgname_and_caption(caption_file)
        self.transform = transform if transform else image_transform

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
            img_path = find_image_with_any_ext(self.img_dir, img_id)

            if img_path is None or not os.path.exists(img_path):
                missing += 1
                continue

            imgname_caption_list.append((img_path, caption))

        if malformed > 0:
            print(f"[WARN] Skipped {malformed} malformed caption lines (no/invalid caption).")
        if missing > 0:
            print(f"[WARN] Dropped {missing} samples due to missing image files.")

        if len(imgname_caption_list) == 0:
            raise RuntimeError("[Flickr7kBanglaDataset] No valid samples found after filtering.")

        return imgname_caption_list

    def __len__(self):
        return len(self.imgname_caption_list)

    def __getitem__(self, ix):
        img_path, caption = self.imgname_caption_list[ix]
        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"[ERROR] Could not open image (corrupt?): {img_path}, {e}")
            image = Image.new("RGB", (224, 224))

        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, caption

class FlickrStyle7kBanglaDataset(Dataset):
    '''Styled caption dataset (romantic / humorous — text only, no images)'''
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

# --------- Collate functions ---------
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

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    decoder_input_ids, labels, lengths = _encode_batch(captions)
    return images, decoder_input_ids, labels, lengths

def collate_fn_styled(captions):
    decoder_input_ids, labels, lengths = _encode_batch(captions)
    return decoder_input_ids, labels, lengths

# --------- Loader functions ---------
def get_data_loader(img_dir, caption_file, batch_size, shuffle=False, num_workers=2):
    dataset = Flickr7kBanglaDataset(img_dir, caption_file, transform=image_transform)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, collate_fn=collate_fn)

def get_styled_data_loader(caption_file, batch_size, shuffle=False, num_workers=2):
    dataset = FlickrStyle7kBanglaDataset(caption_file)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, collate_fn=collate_fn_styled)
