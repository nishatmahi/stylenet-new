import os
import torch
from PIL import Image
from torchvision import transforms

from config import config
from models import BanglaT5StyleCaptioner, load_vit_for_inference
from data_loader import tokenizer, strip_ext

# ============================================================
T5_CKPT        = "csebuetnlp/banglat5"
FACTORED_DIM   = 256                     # must match --factored_dim in train.py
CHECKPOINT     = "/kaggle/working/stylenet_t5_models/best_model.pth"
IMG_DIR        = config.simg_path        # sample folder, same as LSTM sample.py
VIT_CACHE_DIR  = "/kaggle/working/vit_feature_cache"

MODES       = ["factual", "romantic"]
BEAM_SIZE   = 5
MAX_NEW     = 40
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Rescale:
    """Matches the caching script exactly: int output_size, aspect-preserving,
    short side scaled to 224, then CenterCrop."""
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


def get_features(img_name, vit_extract):
    """Cache hit -> load fp16 tensor from disk. Miss -> run ViT once."""
    cache_path = os.path.join(VIT_CACHE_DIR, f"{strip_ext(img_name)}.pt")

    if os.path.exists(cache_path):
        feats = torch.load(cache_path, map_location='cpu')
        return feats.unsqueeze(0).to(device), "cache"

    if vit_extract is None:
        raise RuntimeError(f"No cached features for {img_name} and ViT not loaded.")

    image = Image.open(os.path.join(IMG_DIR, img_name)).convert("RGB")
    image = transform(image).unsqueeze(0)
    return vit_extract(image), "vit"


def main():
    names = sorted(
        n for n in os.listdir(IMG_DIR)
        if n.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    print(f"Captioning {len(names)} image(s) from {IMG_DIR}: {names}")

    need_vit = any(
        not os.path.exists(os.path.join(VIT_CACHE_DIR, f"{strip_ext(n)}.pt"))
        for n in names
    )
    vit_extract = load_vit_for_inference(device=device) if need_vit else None
    print("[INFO] ViT loaded for uncached images." if need_vit
          else "[INFO] All features cached — ViT not loaded.")

    model = BanglaT5StyleCaptioner(
        t5_ckpt=T5_CKPT,
        vit_hidden=768,
        factored_dim=FACTORED_DIM,
        styles=("factual", "romantic"),
        drop_text_encoder=True,
        gradient_checkpointing=False,     # keep use_cache=True for generation
    ).to(device)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)

    real_missing = [k for k in missing if not k.startswith("t5.encoder.")]
    if real_missing:
        print(f"[WARN] {len(real_missing)} missing keys, e.g. {real_missing[:5]}")
    if unexpected:
        print(f"[WARN] {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")

    print(f"[DEBUG] Loaded epoch {ckpt.get('epoch', -1) + 1}, "
          f"best_val_loss={ckpt.get('best_val_loss', 'N/A')}")

    model.eval()
    model.t5.config.use_cache = True

    with torch.no_grad():
        for name in names:
            feats, src = get_features(name, vit_extract)
            print(f"\n{name}  (features from {src}, shape {tuple(feats.shape)})")
            for mode in MODES:
                ids = model.generate_caption(
                    raw_features=feats,
                    mode=mode,
                    num_beams=BEAM_SIZE,
                    max_new_tokens=MAX_NEW,
                )
                caption = tokenizer.decode(ids[0], skip_special_tokens=True)
                print(f"  [{mode}] {caption}")


if __name__ == "__main__":
    main()
