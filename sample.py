import os
import re
import json
import torch

from models import StyleNetT5, load_compatible
from data_loader import tokenizer, strip_ext

# ============================================================
CHECKPOINT = "/kaggle/working/stylenet_clip/best_model.pth"
TEST_FILE  = "/kaggle/working/splits/factual_test.txt"
CACHE_DIR  = "/kaggle/working/clip_feature_cache"
OUT_JSON   = "/kaggle/working/test_generations.json"

N_IMAGES  = 20        # raise to 1000 for the final evaluation run
LAMS      = [1.0, 1.5, 2.0]
BEAM_SIZE = 5
MAX_NEW   = 40
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_images(path, limit):
    """Unique image ids from the test split with their FACTUAL references.

    No stylized references: the style corpus is monolingual, so no romantic
    caption belongs to any particular image and any pairing would be
    fabricated. Stylized output is judged on relevance and style
    appropriateness, as in FS-StyleCap and Detach-and-Attach.
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

    model = StyleNetT5(styles=tuple(styles)).to(device)
    load_compatible(model, ckpt['model'])
    model.eval()
    model.t5.config.use_cache = True

    print(f"[ckpt] epoch {ckpt.get('epoch', -1)+1}, "
          f"best val {ckpt.get('best', 'N/A')}, styles {styles}")

    img_ids, fac_refs = test_images(TEST_FILE, N_IMAGES)
    print(f"\ncaptioning {len(img_ids)} held-out TEST images\n")

    records = []
    with torch.no_grad():
        for img in img_ids:
            p = os.path.join(CACHE_DIR, f"{img}.pt")
            if not os.path.exists(p):
                continue
            feats = torch.load(p, map_location='cpu').unsqueeze(0).to(device)

            rec = {"image_id": img, "references": fac_refs[img]}
            print(f"{img}.jpg")
            print(f"  [ref]      {fac_refs[img][0]}")

            ids = model.generate(feats, target="factual", lam=0.0,
                                 num_beams=BEAM_SIZE, max_new_tokens=MAX_NEW)
            fac = tokenizer.decode(ids[0], skip_special_tokens=True)
            rec["factual"] = fac
            print(f"  [factual]  {fac}")

            for s in styles:
                if s == "factual":
                    continue
                for lam in LAMS:
                    ids = model.generate(feats, target=s, lam=lam,
                                         num_beams=BEAM_SIZE,
                                         max_new_tokens=MAX_NEW)
                    txt = tokenizer.decode(ids[0], skip_special_tokens=True)
                    rec[f"{s}_lam{lam}"] = txt
                    print(f"  [{s} λ={lam}]  {txt}")
            print()
            records.append(rec)

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"wrote {OUT_JSON} ({len(records)} images) — "
          f"use this for BLEU/CIDEr on 'factual' and LLM judging on the "
          f"stylized fields")


if __name__ == "__main__":
    main()
