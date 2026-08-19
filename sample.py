%%writefile sample.py
import os
import re
import torch

from models import StyleNetT5, load_compatible
from data_loader import tokenizer, strip_ext

# ============================================================
CHECKPOINT = "/kaggle/working/stylenet_clip/best_model.pth"
TEST_FILE  = "/kaggle/working/splits/factual_test.txt"
CACHE_DIR  = "/kaggle/working/clip_feature_cache"
SPLIT_DIR  = "/kaggle/working/splits"

N_IMAGES  = 20
LAMS      = [1.0, 3.0]     # style strength; 0 would be identical to factual
BEAM_SIZE = 5
MAX_NEW   = 40
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_images(path, limit):
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


def style_refs(split_dir, styles):
    """imgname<TAB>caption from prepare_splits.py"""
    out = {}
    for s in styles:
        p = os.path.join(split_dir, f"{s}_test_refs.txt")
        if not os.path.exists(p):
            continue
        d = {}
        with open(p, encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    k, v = line.rstrip('\n').split('\t', 1)
                    d[k] = v
        out[s] = d
    return out


def main():
    ckpt = torch.load(CHECKPOINT, map_location=device)
    styles = ckpt.get('styles', ["factual", "romantic", "humorous"])

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
    sty_refs = style_refs(SPLIT_DIR, [s for s in styles if s != "factual"])
    print(f"\ncaptioning {len(img_ids)} held-out TEST images\n")

    with torch.no_grad():
        for img in img_ids:
            p = os.path.join(CACHE_DIR, f"{img}.pt")
            if not os.path.exists(p):
                print(f"{img}: no cached features\n")
                continue
            feats = torch.load(p, map_location='cpu').unsqueeze(0).to(device)

            print(f"{img}.jpg")
            print(f"  [ref factual] {fac_refs[img][0]}")
            for s in styles:
                if s != "factual" and img in sty_refs.get(s, {}):
                    print(f"  [ref {s}] {sty_refs[s][img]}")

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
