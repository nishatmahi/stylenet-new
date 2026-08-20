import os
import re
import torch

from models import StyleNetT5, load_compatible
from data_loader import tokenizer, strip_ext

# ============================================================
CHECKPOINT = "/kaggle/working/stylenet_clip/best_model.pth"
TEST_FILE  = "/kaggle/working/splits/factual_test.txt"
CACHE_DIR  = "/kaggle/working/clip_feature_cache"

N_IMAGES  = 20
LAMS      = [1.5, 2.0, 2.5, 3.0]   # must match --lams in train.py
BEAM_SIZE = 5
MAX_NEW   = 40
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_images(path, limit):
    """Unique image ids from the test split with their FACTUAL references.

    No stylized references are loaded. The style corpus is monolingual — no
    romantic caption belongs to any particular image — so any image-to-style
    pairing would be fabricated. Stylized output is evaluated by human or LLM
    judgement on relevance and style appropriateness, as in FS-StyleCap and
    Detach-and-Attach.
    """
    r = re.compile(r'#\d*')
    order, refs = [], {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = [x.strip() for x in r.split(line.strip()) if x.strip()]
            if len(parts) < 2:
                continue
            img = strip_ext(parts[0])
            if img not in refs:
                if len(order) >= limit:
                    continue
                order.append(img)
                refs[img] = []
            refs[img].append(parts[1])
    return order, refs


def main():
    ckpt = torch.load(CHECKPOINT, map_location=device)
    styles = ckpt.get('styles', ["factual", "romantic"])

    model = StyleNetT5(styles=tuple(styles),
                       gradient_checkpointing=False).to(device)
    load_compatible(model, ckpt['model'])
    model.eval()
    model.t5.config.use_cache = True

    print(f"[ckpt] epoch {ckpt.get('epoch', -1)+1}, "
          f"best val {ckpt.get('best', 'N/A')}, styles {styles}")
    for s in styles:
        print(f"  ||s_{s}|| = {model.style[s].norm().item():.4f}")

    img_ids, fac_refs = test_images(TEST_FILE, N_IMAGES)
    print(f"\ncaptioning {len(img_ids)} held-out TEST images\n")

    with torch.no_grad():
        for img in img_ids:
            p = os.path.join(CACHE_DIR, f"{img}.pt")
            if not os.path.exists(p):
                print(f"{img}: no cached features\n")
                continue
            feats = torch.load(p, map_location='cpu').unsqueeze(0).to(device)

            print(f"{img}.jpg")
            print(f"  [ref]      {fac_refs[img][0]}")

            ids = model.generate(feats, target="factual", lam=0.0,
                                 num_beams=BEAM_SIZE, max_new_tokens=MAX_NEW)
            print(f"  [factual]  {tokenizer.decode(ids[0], skip_special_tokens=True)}")

            for s in styles:
                if s == "factual":
                    continue
                for lam in LAMS:
                    ids = model.generate(feats, target=s, lam=lam,
                                         num_beams=BEAM_SIZE,
                                         max_new_tokens=MAX_NEW)
                    txt = tokenizer.decode(ids[0], skip_special_tokens=True)
                    print(f"  [{s} λ={lam}]  {txt}")
            print()


if __name__ == "__main__":
    main()
