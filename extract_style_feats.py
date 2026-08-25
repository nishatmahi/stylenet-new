"""
extract_style_feats.py — NLLB-CLIP embeddings for the STYLE model only.

PPCap trains the stylized model on text and runs it on images, using CLIP's
shared embedding space as the bridge (paper Sec. 3.3, Fig. 3). OpenAI CLIP's
text tower is English-only, so for Bangla we use NLLB-CLIP: an NLLB-200 text
tower trained onto a FROZEN CLIP image tower, covering all 201 Flores-200
languages. Bengali is one of the five low-resource languages the NLLB-CLIP
paper reports numbers on.

Your FACTUAL model is untouched by this file. It only ever sees images, so it
keeps its existing CLIP ViT-B/32 patch-token cache. The two models share no
weights, so they do not have to share a CLIP.

    pip install open_clip_torch

    # style corpora -> text embeddings (training side)
    python extract_style_feats.py --mode text \
        --in_file  /kaggle/working/splits/romantic_train.txt \
        --out      /kaggle/working/style_feats/romantic_train.pt

    # images -> image embeddings (inference side)
    python extract_style_feats.py --mode image \
        --image_dir /kaggle/input/flickr30k/images \
        --id_file   /kaggle/working/splits/factual_test.txt \
        --out       /kaggle/working/style_feats/test_images.pt
"""

import os
import argparse

import torch
from PIL import Image

MODEL = "nllb-clip-base-siglip"
PRETRAINED = "v1"
LANG = "ben_Beng"          # Flores-200 code for Bengali


def load_model(device):
    from open_clip import create_model_from_pretrained, get_tokenizer
    model, transform = create_model_from_pretrained(MODEL, PRETRAINED, device=device)
    tokenizer = get_tokenizer(MODEL)
    tokenizer.set_language(LANG)
    model.eval()
    return model, transform, tokenizer


def read_lines(path):
    return [l.strip() for l in open(path, encoding='utf-8') if l.strip()]


def strip_ext(name):
    for e in ('.jpg', '.jpeg', '.png'):
        if name.lower().endswith(e):
            return name[:-len(e)]
    return name


def image_ids(path):
    """Distinct image ids, in order, from a `<id>.jpg#n<TAB>caption` file."""
    seen, order = set(), []
    for line in read_lines(path):
        key = line.split('\t', 1)[0] if '\t' in line else line.split('#', 1)[0]
        img = strip_ext(key.split('#')[0].strip())
        if img not in seen:
            seen.add(img)
            order.append(img)
    return order


@torch.no_grad()
def encode_text(model, tokenizer, lines, device, bs):
    out = []
    for i in range(0, len(lines), bs):
        toks = tokenizer(lines[i:i + bs]).to(device)
        e = model.encode_text(toks).float().cpu()
        out.append(e)
        print(f"  {min(i+bs, len(lines))}/{len(lines)}", flush=True)
    return torch.cat(out, 0)


@torch.no_grad()
def encode_images(model, transform, ids, image_dir, device, bs):
    feats, kept = [], []
    batch, batch_ids = [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        feats.append(model.encode_image(x).float().cpu())
        kept.extend(batch_ids)
        batch.clear()
        batch_ids.clear()

    missing = 0
    for n, img_id in enumerate(ids):
        path = None
        for e in ('.jpg', '.jpeg', '.png'):
            p = os.path.join(image_dir, img_id + e)
            if os.path.exists(p):
                path = p
                break
        if path is None:
            missing += 1
            continue
        batch.append(transform(Image.open(path).convert('RGB')))
        batch_ids.append(img_id)
        if len(batch) == bs:
            flush()
            print(f"  {n+1}/{len(ids)}", flush=True)
    flush()
    if missing:
        print(f"[WARN] {missing} images not found under {image_dir}")
    return torch.cat(feats, 0), kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['text', 'image'], required=True)
    p.add_argument('--in_file', help='text mode: one sentence per line')
    p.add_argument('--id_file', help='image mode: caption file to take image ids from')
    p.add_argument('--image_dir', help='image mode: directory of images')
    p.add_argument('--out', required=True)
    p.add_argument('--batch_size', type=int, default=64)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, transform, tokenizer = load_model(device)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    if args.mode == 'text':
        lines = read_lines(args.in_file)
        print(f"[text] {len(lines)} lines from {os.path.basename(args.in_file)}")
        emb = encode_text(model, tokenizer, lines, device, args.batch_size)
        torch.save({'kind': 'text', 'lines': lines, 'emb': emb,
                    'model': MODEL, 'lang': LANG}, args.out)
    else:
        ids = image_ids(args.id_file)
        print(f"[image] {len(ids)} distinct images from {os.path.basename(args.id_file)}")
        emb, kept = encode_images(model, transform, ids, args.image_dir,
                                  device, args.batch_size)
        torch.save({'kind': 'image', 'ids': kept, 'emb': emb,
                    'model': MODEL}, args.out)

    print(f"\nwrote {args.out}   shape {tuple(emb.shape)}")
    print("dim is read back automatically by the trainer — nothing to hardcode.")


if __name__ == '__main__':
    main()
