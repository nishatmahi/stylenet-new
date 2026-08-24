"""
sample.py — two-stage decoding.

    stage 1   image  --(lam=0)-->  factual caption
    stage 2   that text --(lam)-->  styled caption

Both stages run in a configuration the model was trained in: stage 1 is the
CAP pass, stage 2 is the STY pass (text in the encoder, knobs on). The
image-plus-knobs combination is never used, because it never occurs during
training.

    python sample.py                                   # 20 images
    python sample.py --n_images 1000 --out_json preds.json
    python sample.py --one_step 0                      # drop the comparison
"""

import os
import json
import argparse
import contextlib
import torch

from models import StyleNetT5, load_compatible
from data_loader import tokenizer, encode, _split_line, normalize


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default="/kaggle/working/stylenet_clip/best_model.pth")
    p.add_argument('--test_file', default="/kaggle/working/splits/factual_test.txt")
    p.add_argument('--cache_dir', default="/kaggle/working/clip_feature_cache")
    p.add_argument('--out_json', default="/kaggle/working/test_generations.json")
    p.add_argument('--n_images', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--beam_size', type=int, default=5)
    p.add_argument('--max_new', type=int, default=48)
    p.add_argument('--lams', default='0.5,1.0,1.5')
    p.add_argument('--one_step', type=int, default=1,
                   help='also decode image->styled in a single pass, for '
                        'comparison. This is the untrained configuration; it '
                        'is here as an ablation, not as the system.')
    p.add_argument('--amp', type=int, default=1)
    return p.parse_args()


def test_images(path, limit):
    """Image ids with their factual references.

    No stylized references: the style corpus is monolingual, so no romantic
    sentence belongs to any particular image and any pairing would be invented.
    Stylized output is judged on relevance and style appropriateness, as in
    FS-StyleCap and MemCap.

    References are normalized with the same pipeline the model trained
    through — the model emits normalized Bangla, so scoring raw references
    against it costs BLEU for a text-processing reason, not a model reason.
    """
    order, refs, bad = [], {}, 0
    for line in open(path, encoding='utf-8'):
        parsed = _split_line(line.strip())
        if parsed is None or not parsed[1]:
            bad += 1
            continue
        img, cap = parsed
        if img not in refs:
            if len(order) >= limit:
                continue
            order.append(img)
            refs[img] = []
        refs[img].append(normalize(cap))
    if bad:
        print(f"[WARN] {bad} malformed lines in {path}")
    return order, refs


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lams = [float(x) for x in args.lams.split(',')]

    ck = torch.load(args.checkpoint, map_location=device)
    style = ck.get('style', 'romantic')
    model = StyleNetT5(clip_dim=ck.get('clip_dim', 768), style=style,
                       bottleneck=ck.get('bottleneck', 64)).to(device)
    load_compatible(model, ck['model'])
    model.eval()
    print(f"[ckpt] epoch {ck.get('epoch', -1)+1}, best val {ck.get('best')}, "
          f"style {style}")

    cuda = device.type == 'cuda'
    if args.amp and cuda and torch.cuda.is_bf16_supported():
        amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    elif args.amp and cuda:
        amp = lambda: torch.autocast('cuda', dtype=torch.float16)
    else:
        amp = contextlib.nullcontext

    ids_all, refs = test_images(args.test_file, args.n_images)
    keep = [i for i in ids_all
            if os.path.exists(os.path.join(args.cache_dir, f"{i}.pt"))]
    if len(keep) < len(ids_all):
        print(f"[WARN] {len(ids_all)-len(keep)} test images have no cached feature")
    print(f"\ndecoding {len(keep)} held-out TEST images, two-stage\n")

    records = {i: {"image_id": i, "references": refs[i]} for i in keep}

    with torch.no_grad():
        for s in range(0, len(keep), args.batch_size):
            chunk = keep[s:s + args.batch_size]
            feats = torch.stack([
                torch.load(os.path.join(args.cache_dir, f"{i}.pt"),
                           map_location='cpu') for i in chunk]).to(device)

            # ---- stage 1: image -> factual caption --------------------
            with amp():
                fac_ids = model.generate(feats, lam=0.0,
                                         num_beams=args.beam_size,
                                         max_new_tokens=args.max_new,
                                         repetition_penalty=1.15,
                                         no_repeat_ngram_size=3)
            factual = tokenizer.batch_decode(fac_ids, skip_special_tokens=True)
            for img, t in zip(chunk, factual):
                records[img]["factual"] = t

            # ---- stage 2: that text -> styled caption ------------------
            tids, tmask = encode(factual)
            tids, tmask = tids.to(device), tmask.to(device)
            for lam in lams:
                with amp():
                    out = model.generate_text(tids, tmask, lam=lam,
                                              num_beams=args.beam_size,
                                              max_new_tokens=args.max_new)
                txts = tokenizer.batch_decode(out, skip_special_tokens=True)
                for img, t in zip(chunk, txts):
                    records[img][f"{style}_lam{lam}"] = t

            # ---- ablation: single pass, image + knobs ------------------
            if args.one_step:
                for lam in lams:
                    with amp():
                        out = model.generate(feats, lam=lam,
                                             num_beams=args.beam_size,
                                             max_new_tokens=args.max_new,
                                             repetition_penalty=1.15,
                                             no_repeat_ngram_size=3)
                    txts = tokenizer.batch_decode(out, skip_special_tokens=True)
                    for img, t in zip(chunk, txts):
                        records[img][f"onestep_lam{lam}"] = t

            print(f"  {min(s+args.batch_size, len(keep))}/{len(keep)}", flush=True)

    out = [records[i] for i in keep]
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    for rec in out[:5]:
        print(f"\n{rec['image_id']}.jpg")
        print(f"  [ref]        {rec['references'][0]}")
        print(f"  [1 factual]  {rec['factual']}")
        for lam in lams:
            print(f"  [2 {style} λ={lam}]  {rec[f'{style}_lam{lam}']}")
        if args.one_step:
            for lam in lams:
                print(f"      (one-step λ={lam})  {rec[f'onestep_lam{lam}']}")

    print(f"\nwrote {args.out_json} ({len(out)} images)")
    print("BLEU/CIDEr go on 'factual' — the stylized fields have no paired "
          "reference by construction, so they are judged on whether the "
          "content of stage 1 survives into stage 2 and whether style appears.")


if __name__ == "__main__":
    main()
