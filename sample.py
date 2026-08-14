import os
import torch
from PIL import Image
from torchvision import transforms

from models import BanglaT5StyleCaptioner, load_compatible, load_vit_for_inference
from data_loader import tokenizer, strip_ext

# ============================================================
T5_CKPT       = "csebuetnlp/banglat5"
CHECKPOINT    = "/kaggle/working/stylenet_t5_models/best_model.pth"
IMG_DIR       = "/kaggle/input/datasets/kaggleperfect/sample-data/sample/sample_images"
VIT_CACHE_DIR = "/kaggle/working/vit_feature_cache"

MAX_IMAGES = 20
MODES      = ["factual", "romantic"]
BEAM_SIZE  = 5
MAX_NEW    = 40
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def get_features(img_name, vit_extract):
    cache_path = os.path.join(VIT_CACHE_DIR, f"{strip_ext(img_name)}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location='cpu').unsqueeze(0).to(device), "cache"
    image = Image.open(os.path.join(IMG_DIR, img_name)).convert("RGB")
    return vit_extract(transform(image).unsqueeze(0)), "vit"


def main():
    names = sorted(n for n in os.listdir(IMG_DIR)
                   if n.lower().endswith(('.jpg', '.jpeg', '.png')))[:MAX_IMAGES]
    print(f"Captioning {len(names)} image(s) from {IMG_DIR}")

    need_vit = any(not os.path.exists(os.path.join(VIT_CACHE_DIR, f"{strip_ext(n)}.pt"))
                   for n in names)
    vit_extract = load_vit_for_inference(device=device) if need_vit else None

    ckpt = torch.load(CHECKPOINT, map_location=device)
    model = BanglaT5StyleCaptioner(
        t5_ckpt=T5_CKPT, vit_hidden=768,
        styles=tuple(ckpt.get('styles', ["factual", "romantic"])),
        gradient_checkpointing=False,
    ).to(device)
    load_compatible(model, ckpt['model_state_dict'])
    print(f"[DEBUG] epoch {ckpt.get('epoch', -1)+1}, "
          f"best_val_loss={ckpt.get('best_val_loss', 'N/A')}")

    model.eval()
    model.t5.config.use_cache = True

    with torch.no_grad():
        for name in names:
            feats, src = get_features(name, vit_extract)
            print(f"\n{name}  ({src})")
            for mode in MODES:
                ids = model.generate_caption(raw_features=feats, mode=mode,
                                             num_beams=BEAM_SIZE,
                                             max_new_tokens=MAX_NEW)
                print(f"  [{mode}] {tokenizer.decode(ids[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
