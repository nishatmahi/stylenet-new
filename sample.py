import os
import re
import torch

from models import BanglaT5StyleCaptioner, load_compatible
from data_loader import tokenizer, strip_ext

# ============================================================
T5_CKPT       = "csebuetnlp/banglat5"
CHECKPOINT    = "/kaggle/working/stylenet_t5_models/best_model.pth"
TEST_FILE     = "/kaggle/working/splits/factual_test.txt"   # 1000 held-out images
VIT_CACHE_DIR = "/kaggle/working/vit_feature_cache"

N_IMAGES  = 20
MODES     = ["factual", "romantic"]
BEAM_SIZE = 5
MAX_NEW   = 40
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_images_and_refs(path, limit):
    """Unique image ids from the TEST split with their factual references.
    These images appear in neither training nor validation."""
    r = re.compile(r'#\d*')
    order, refs = [], {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in r.split(line) if x.strip()]
            if len(parts) < 2:
                continue
            img_id = strip_ext(parts[0])
            if img_id not in refs:
                if len(order) >= limit:
                    continue
                order.append(img_id)
                refs[img_id] = []
            refs[img_id].append(parts[1])
    return order, refs


def main():
    if not os.path.exists(TEST_FILE):
        raise RuntimeError(f"{TEST_FILE} missing. Run prepare_splits.py first.")

    img_ids, refs = test_images_and_refs(TEST_FILE, N_IMAGES)
    print(f"Captioning {len(img_ids)} held-out TEST images\n")

    ckpt = torch.load(CHECKPOINT, map_location=device)
    model = BanglaT5StyleCaptioner(
        t5_ckpt=T5_CKPT, vit_hidden=768,
        styles=tuple(ckpt.get('styles', ["factual", "romantic"])),
        gradient_checkpointing=False,
    ).to(device)
    load_compatible(model, ckpt['model_state_dict'])
    print(f"[DEBUG] epoch {ckpt.get('epoch', -1)+1}, "
          f"best_val_loss={ckpt.get('best_val_loss', 'N/A')}\n")

    model.eval()
    model.t5.config.use_cache = True

    with torch.no_grad():
        for img_id in img_ids:
            cache_path = os.path.join(VIT_CACHE_DIR, f"{img_id}.pt")
            if not os.path.exists(cache_path):
                print(f"{img_id}: no cached features, skipping\n")
                continue
            feats = torch.load(cache_path, map_location='cpu').unsqueeze(0).to(device)

            print(f"{img_id}.jpg")
            print(f"  [reference] {refs[img_id][0]}")
            for mode in MODES:
                ids = model.generate_caption(raw_features=feats, mode=mode,
                                             num_beams=BEAM_SIZE,
                                             max_new_tokens=MAX_NEW)
                print(f"  [{mode}]    {tokenizer.decode(ids[0], skip_special_tokens=True)}")
            print()


if __name__ == "__main__":
    main()
