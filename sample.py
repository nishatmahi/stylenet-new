import os
import torch
from PIL import Image
from torchvision import transforms

from config import config
from models import EncoderViT, BanglaT5StyleCaptioner
from data_loader import tokenizer

# ============================================================
T5_CKPT       = "csebuetnlp/banglat5"
STYLE_RANK    = 8                       # must match --style_rank in train.py
CHECKPOINT    = "/kaggle/working/stylenet_new_again_models/best_model.pth"
IMG_DIR       = "/kaggle/input/datasets/kaggleperfect/sample-data/sample/sample_images"
VIT_CACHE_DIR = "/kaggle/working/vit_feature_cache"

MODES     = ["factual", "romantic"]
BEAM_SIZE = 5
MAX_NEW   = 40
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Rescale:
    """Identical to the caching script: int output_size, aspect-preserving."""
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


def get_raw_features(img_name, encoder):
    """Cache hit -> load fp16 from disk. Miss -> run ViT on that one image."""
    img_id = os.path.splitext(img_name)[0]
    cache_path = os.path.join(VIT_CACHE_DIR, f"{img_id}.pt")

    if os.path.exists(cache_path):
        raw = torch.load(cache_path, map_location='cpu').float().unsqueeze(0)
        return raw.to(device), "cache"

    image = Image.open(os.path.join(IMG_DIR, img_name)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return encoder.extract_raw(image), "vit"


@torch.no_grad()
def generate(decoder, features, mode):
    """Same memory construction as decoder.forward, then HF generate with
    KV cache. decoder_start_token_id is bos because training fed
    decoder_input_ids = captions[:, :-1], which begins with <s>."""
    decoder._active_style = mode
    memory, attn_mask = decoder._build_encoder_memory(features, mode, 1, device)

    return decoder.t5.generate(
        inputs_embeds=memory,
        attention_mask=attn_mask,
        decoder_start_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=BEAM_SIZE,
        max_new_tokens=MAX_NEW,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
        use_cache=True,
    )


def main():
    names = sorted(
        n for n in os.listdir(IMG_DIR)
        if n.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    print(f"Captioning {len(names)} image(s) from {IMG_DIR}: {names}")

    encoder = EncoderViT(emb_dim=768).to(device)
    decoder = BanglaT5StyleCaptioner(
        t5_ckpt=T5_CKPT,
        tokenizer_len=len(tokenizer),
        style_rank=STYLE_RANK,
        styles=("factual", "romantic"),
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    print(f"[DEBUG] Loaded epoch {ckpt['epoch'] + 1}, "
          f"best_val_loss={ckpt.get('best_val_loss', 'N/A')}")

    encoder.eval()
    decoder.eval()
    decoder.t5.config.use_cache = True

    with torch.no_grad():
        for name in names:
            raw, src = get_raw_features(name, encoder)
            features = encoder.forward_from_cache(raw)
            print(f"\n{name}  (features from {src}, shape {tuple(features.shape)})")
            for mode in MODES:
                ids = generate(decoder, features, mode)
                caption = tokenizer.decode(ids[0], skip_special_tokens=True)
                print(f"  [{mode}] {caption}")


if __name__ == "__main__":
    main()
