import os
import torch
from PIL import Image
from torchvision import transforms
from models import EncoderViT, TransformerFactoredDecoder
from data_loader import Rescale, tokenizer  # SAME tokenizer instance used in training —
                                              # do not reconstruct a separate one, token
                                              # ids must match exactly what the model's
                                              # embeddings/lm_head were trained against.

# ============================================================
# FILL THESE IN — must match your actual training run exactly
# ============================================================
FACTORED_DIM = 512  # must match --factored_dim used in train.py (default is 512;
                     # only change this if you passed a different value at train time)
SAMPLE_IMG_DIR = "/kaggle/input/dataset/data/Images"  # folder of images you want to caption
CHECKPOINT_PATH = "/kaggle/working/stylenet_transformer_models/best_model.pth"
# ============================================================


def load_sample_images(img_dir, transform):
    img_names = sorted(os.listdir(img_dir))
    img_list = []
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        print("Loading image from:", img_path)
        image = Image.open(img_path).convert("RGB")
        if transform:
            image = transform(image)
        img_list.append(image)
    return img_names, img_list


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Build models exactly as in train.py ---
encoder = EncoderViT(decoder_hidden_size=768).to(device)
decoder = TransformerFactoredDecoder(
    tokenizer=tokenizer,
    gpt2_name="flax-community/gpt2-bengali",
    factored_dim=FACTORED_DIM,
).to(device)

# --- Load from the checkpoint dict train.py actually produces ---
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# strict=False: the checkpoint intentionally excludes frozen vit.*/gpt2.*
# weights (see save_checkpoint_safely in train.py) since those are
# re-populated by from_pretrained() automatically above. This is expected,
# not a sign of a corrupted or partial checkpoint.
encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=False)

print(f"[DEBUG] Loaded checkpoint from epoch {checkpoint['epoch'] + 1}, "
      f"best_val_loss={checkpoint.get('best_val_loss', 'N/A')}")
print(f"[DEBUG] cross_attn_gate at load time: {decoder.cross_attn_gate.item():.6f}")

encoder.eval()
decoder.eval()

# --- Same image transform pipeline as data_loader.py ---
transform = transforms.Compose([
    Rescale((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

img_names, img_list = load_sample_images(SAMPLE_IMG_DIR, transform)

with torch.no_grad():
    idx = 4  # whichever image you want
    image = img_list[idx].unsqueeze(0).to(device)

    features = encoder(image)  # (1, 197, 768) — full patch sequence, not a pooled vector
    print("Image features shape:", features.shape)
    print("First 10 values of CLS patch embedding:", features[0, 0, :10].cpu().numpy())

    output = decoder.sample(
        features,
        tokenizer=tokenizer,
        beam_size=5,
        max_len=30,
        mode="romantic",
    )
    caption = tokenizer.decode(output, skip_special_tokens=True)
    print(img_names[idx], "| Predicted Caption:", caption)
