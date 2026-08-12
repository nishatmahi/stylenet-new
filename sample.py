import os
import torch
from PIL import Image
from torchvision import transforms
from models import EncoderViT, BanglaT5StyleCaptioner
from data_loader import tokenizer  # SAME tokenizer instance used in training

# ============================================================
# FILL THESE IN — must match your actual training run
# ============================================================
STYLE_RANK = 8  # must match --style_rank used in train.py (default 8)
T5_CKPT = "csebuetnlp/banglat5"  # must match --t5_ckpt used in train.py
SAMPLE_IMG_DIR = "/kaggle/input/datasets/kaggleperfect/dataset/data/Images"
CHECKPOINT_PATH = "/kaggle/working/stylenet_new_again_models/best_model.pth"
# ============================================================

def load_sample_images(img_dir, transform):
    img_names = sorted(os.listdir(img_dir))
    img_list = []
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if transform:
            image = transform(image)
        img_list.append(image)
    return img_names, img_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = EncoderViT(emb_dim=768).to(device)
decoder = BanglaT5StyleCaptioner(
    t5_ckpt=T5_CKPT,
    tokenizer_len=len(tokenizer),
    style_rank=STYLE_RANK,
    styles=("factual", "romantic"),
    pad_token_id=tokenizer.pad_token_id,
).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
print(f"[DEBUG] Loaded checkpoint from epoch {checkpoint['epoch'] + 1}, "
      f"best_val_loss={checkpoint.get('best_val_loss', 'N/A')}")

encoder.eval()
decoder.eval()

class Rescale:
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, image):
        w, h = image.size
        if h > w:
            new_h, new_w = int(self.output_size * h / w), self.output_size
        else:
            new_h, new_w = self.output_size, int(self.output_size * w / h)
        return image.resize((new_w, new_h))

transform = transforms.Compose([
    Rescale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

img_names, img_list = load_sample_images(SAMPLE_IMG_DIR, transform)

with torch.no_grad():
    idx = 4
    image = img_list[idx].unsqueeze(0).to(device)
    features = encoder(image)   # (1, N_patches+1, 768) — full patch sequence
    print("Image features shape:", features.shape)

    output = decoder.sample(
        features,
        tokenizer=tokenizer,
        beam_size=5,
        max_len=30,
        mode="romantic",
    )
    caption = tokenizer.decode(output, skip_special_tokens=True)
    print(img_names[idx], "| Predicted Caption:", caption)
