import os
import json
import argparse
import torch

from models import StyleNetT5, load_compatible
from data_loader import tokenizer, _split_line, normalize


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default="/kaggle/working/stylenet_clip/best_model.pth")
    p.add_argument('--test_file', default="/kaggle/working/splits/factual_test.txt")
    p.add_argument('--cache_dir', default="/kaggle/working/clip_feature_cache")
    p.add_argument('--out_json', default="/kaggle/working/test_generations.json")
    p.add_argument('--n_images', type=int, default=20)   # raise for the real run
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--beam_size', type=int, default=5)
    p.add_argument('--max_new', type=int, default=40)
    p.add_argument('--lams', default='1.0,1.5,2.0')
    p.add_argument('--amp', type=int, default=1)
    return p.parse_args()


def test_images(path, limit):
    """Unique image ids from the test split with their FACTUAL references.

    No stylized references: the style corpus is monolingual, so no romantic
    caption belongs to any particular image and any pairing would be
    fabricated. Stylized output is judged on relevance and style
    appropriateness, as in FS-StyleCap and Detach-and-Attach.

    References are normalized with the same csebuetnlp pipeline the model was
    trained through. The model's output is already in normalized form, so
    scoring raw references against normalized hypotheses deflates BLEU for a
    reason that has nothing to do with the model.
    """
    order, refs, malformed = [], {}, 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            parsed = _split_line(line.strip())
            if parsed is None or not parsed[1]:
                malformed += 1
                continue
            img, cap = parsed
            if img not in refs:
                if len(order) >= limit:
                    continue
                order.append(img)
                refs[img] = []
            refs[img].append(normalize(cap))
    if malformed:
        print(f"[WARN] {malformed} malformed lines in {path}")
    return order, refs


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lams = [float(x) for x in args.lams.split(',')]

    ckpt = torch.load(args.checkpoint, map_location=device)
    styles = ckpt.get('styles', ["factual", "romantic"])
    # fall back to the training defaults if the checkpoint predates these keys
    alpha = ckpt.get('alpha', 0.5)
    clip_dim = ckpt.get('clip_dim', 768)

    model = StyleNetT5(clip_dim=clip_dim, styles=tuple(styles),
                       alpha=alpha).to(device)
    load_compatible(model, ckpt['model'])
    model.eval()
    model.t5.config.use_cache = True

    print(f"[ckpt] epoch {ckpt.get('epoch', -1)+1}, "
          f"best val {ckpt.get('best', 'N/A')}, styles {styles}, "
          f"alpha {alpha}, clip_dim {clip_dim}")

    # fp16 on a T4 roughly halves generation time; bf16 where available
    cuda = device.type == 'cuda'
    if args.amp and cuda and torch.cuda.is_bf16_supported():
        amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    elif args.amp and cuda:
        amp = lambda: torch.autocast('cuda', dtype=torch.float16)
    else:
        import contextlib
        amp = contextlib.nullcontext

    img_ids, fac_refs = test_images(args.test_file, args.n_images)

    # keep only images we actually have features for
    keep, skipped = [], 0
    for img in img_ids:
        if os.path.exists(os.path.join(args.cache_dir, f"{img}.pt")):
            keep.append(img)
        else:
            skipped += 1
    if skipped:
        print(f"[WARN] {skipped} test images have no cached feature; skipped")
    print(f"\ncaptioning {len(keep)} held-out TEST images\n")

    # (style, lam) combinations to decode for every image
    combos = [("factual", 0.0)]
    for s in styles:
        if s != "factual":
            combos += [(s, lam) for lam in lams]

    records = {img: {"image_id": img, "references": fac_refs[img]} for img in keep}

    with torch.no_grad():
        for start in range(0, len(keep), args.batch_size):
            chunk = keep[start:start + args.batch_size]
            feats = torch.stack([
                torch.load(os.path.join(args.cache_dir, f"{i}.pt"),
                           map_location='cpu') for i in chunk
            ]).to(device)

            # batch over images, loop over style/lam — one beam search per
            # combo instead of one per (image, combo) pair
            for style, lam in combos:
                with amp():
                    ids = model.generate(feats, target=style, lam=lam,
                                         num_beams=args.beam_size,
                                         max_new_tokens=args.max_new)
                texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
                key = "factual" if style == "factual" else f"{style}_lam{lam}"
                for img, txt in zip(chunk, texts):
                    records[img][key] = txt

            print(f"  {min(start + args.batch_size, len(keep))}/{len(keep)}",
                  flush=True)

    out = [records[i] for i in keep]
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # show the first few so you can eyeball them without opening the file
    for rec in out[:5]:
        print(f"\n{rec['image_id']}.jpg")
        print(f"  [ref]      {rec['references'][0]}")
        print(f"  [factual]  {rec['factual']}")
        for style, lam in combos:
            if style == "factual":
                continue
            print(f"  [{style} λ={lam}]  {rec[f'{style}_lam{lam}']}")

    print(f"\nwrote {args.out_json} ({len(out)} images) — "
          f"BLEU/CIDEr on 'factual', LLM or human judging on the stylized fields")


if __name__ == "__main__":
    main()
