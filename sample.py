import os
import torch
from torchvision.io import read_image
from torchvision import transforms
from transformers import AutoTokenizer
import re
from config import config
from models import EncoderViT, FactoredLSTM

def load_sample_images(img_dir, transform):
    img_names = sorted(os.listdir(img_dir))
    img_list = []
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        image = read_image(img_path).float() / 255.0
        if transform:
            image = transform(image)
        img_list.append(image)
    return img_names, img_list

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("hishab/titulm-mpt-1b-v2.0", trust_remote_code=True)

# Initialize models
encoder = EncoderViT(config.emb_dim).to("cuda")
decoder = FactoredLSTM(config.emb_dim, config.hidden_dim, config.factored_dim, vocab_size=tokenizer.vocab_size).to("cuda")

encoder.load_state_dict(torch.load(os.path.join(config.model_path, "encoder-last.pkl")))
decoder.load_state_dict(torch.load(os.path.join(config.model_path, "decoder-last.pkl")))

# Image transform 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load and preprocess images
img_names, img_list = load_sample_images(config.simg_path, transform)

# Use only the first image
idx = 0
selected_image = img_list[idx].unsqueeze(0).to("cuda")

encoder.eval()
decoder.eval()
with torch.no_grad():
    features = encoder(selected_image)
    style = "factual"  # Choose your style here!
    caption_ids = decoder.sample(features, beam_size=5, max_len=30, mode=style)
    caption = tokenizer.decode(caption_ids, skip_special_tokens=True).strip()
    caption = re.sub(r"(।){2,}", "।", caption)

print(f"Image: {img_names[idx]}")
print(f"Caption: {caption}")
