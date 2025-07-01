import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image

class FlickerDataset(Dataset):
    '''Flickr8K dataset'''
    def __init__(self, img_dir, caption_file, tokenizer, transform=None):
        '''
        Args:
            img_dir: Directory with all the images
            caption_file: Path to the factual caption file
            tokenizer: Tokenizer instance
            transform: Optional transform to be applied
        '''
        self.img_dir = img_dir
        self.imgname_caption_list = self._get_imgname_and_caption(caption_file)
        self.transform = transform
        self.tokenizer = tokenizer

    def _get_imgname_and_caption(self, caption_file):
        '''extract image name and caption from factual caption file'''
        with open(caption_file, 'r') as f:
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
        '''return one data pair (image and caption)'''
        img_name = self.imgname_caption_list[ix][0]
        img_name = os.path.join(self.img_dir, img_name)
        caption = self.imgname_caption_list[ix][1]

        # Add valid extension
        valid_extensions = ['.jpg', '.png', '.jpeg','.JPG']
        valid_name = ""
        for ext in valid_extensions:
            valid_name = img_name + ext
            if os.path.exists(valid_name):
                img_name = valid_name
                break

        # --- ADD THIS PART ---
        try:
            image = read_image(img_name)  # Shape: [C, H, W]
        except Exception as e:
            print(f"[ERROR] Failed to load image: {img_name}")
            print(f"[ERROR] Reason: {e}")
            raise e
    # --- END ADD ---
        
        # Convert to RGB if needed (only modification)
        if image.shape[0] == 1:  # Grayscale -> RGB
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:  # RGBA -> RGB
            image = image[:3]
        
        image = image.float() / 255.0  # Keep original scaling

        if self.transform is not None:
            image = self.transform(image)

        # Original caption processing
        caption_list = self.tokenizer.encode(caption, return_tensors="pt").tolist()[0]
        caption_list = torch.Tensor(caption_list)
        return image, caption_list


class FlickrStyleDataset(Dataset):
    '''Styled caption dataset'''
    def __init__(self, caption_file, tokenizer):
        '''
        Args:
            caption_file: Path to styled caption file
            tokenizer: Tokenizer instance
        '''
        self.caption_list = self._get_caption(caption_file)
        self.tokenizer = tokenizer

    def _get_caption(self, caption_file):
        '''extract caption list from styled caption file'''
        with open(caption_file, 'r') as f:
            caption_list = f.readlines()

        caption_list = [x.strip() for x in caption_list]
        return caption_list

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, ix):
        caption_text = self.caption_list[ix]
        caption_text = caption_text.split("#")[-1][1:]  # Remove image name
        caption_text = caption_text.strip()

        caption_list = self.tokenizer.encode(caption_text, return_tensors="pt").tolist()[0]
        caption_list = torch.Tensor(caption_list)
        return caption_list


def get_data_loader(img_dir, caption_file, tokenizer, batch_size,
                   transform=None, shuffle=False, num_workers=0):
    if transform is None:
        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    flickr7k = FlickerDataset(img_dir, caption_file, tokenizer, transform)
    data_loader = DataLoader(dataset=flickr7k,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           collate_fn=lambda data: collate_fn(data, tokenizer.pad_token_id))
    return data_loader


def get_styled_data_loader(caption_file, tokenizer, batch_size,
                         shuffle=False, num_workers=1):
    flickr_styled_7k = FlickrStyleDataset(caption_file, tokenizer)
    data_loader = DataLoader(dataset=flickr_styled_7k,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           collate_fn=lambda data: collate_fn_styled(data, tokenizer.pad_token_id))
    return data_loader


def collate_fn(data, pad_token_id):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    lengths = torch.LongTensor([len(cap) for cap in captions])
    captions = [pad_sequence(cap, max(lengths), pad_token_id) for cap in captions]
    captions = torch.stack(captions, 0)
    return images, captions, lengths


def collate_fn_styled(captions, pad_token_id):
    captions.sort(key=lambda x: len(x), reverse=True)
    lengths = torch.LongTensor([len(cap) for cap in captions])
    captions = [pad_sequence(cap, max(lengths), pad_token_id) for cap in captions]
    captions = torch.stack(captions, 0)
    return captions, lengths


def pad_sequence(seq, max_len, pad_token_id):
    return torch.cat((seq, torch.full((max_len - len(seq),), fill_value=pad_token_id, dtype=torch.long)))
