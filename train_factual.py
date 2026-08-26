"""
train_factual.py — step 1. Image -> factual Bangla caption.

Only the decoder (+ enc_to_dec_proj if dims differ) trains. Cached CLIP
features go straight into cross-attention. No projector, no auxiliary loss,
no staging, no freezing.

Run smoke_kaggle.py first. If it does not print PASS, do not run this.
"""
import os, argparse, torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput
from data import CaptionData
from models import build_tokenizer, build_factual, factual_params


def encode(model, f):
    return BaseModelOutput(last_hidden_state=f)


def main(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.save_dir, exist_ok=True)
    tok = build_tokenizer()
    print(f"[tok] len {len(tok)}  bos {tok.bos_token_id}  eos {tok.eos_token_id}"
          f"  pad {tok.pad_token_id}")

    tr_ds = CaptionData(a.cache_dir, os.path.join(a.split_dir, 'factual_train.txt'), tok)
    va_ds = CaptionData(a.cache_dir, os.path.join(a.split_dir, 'factual_val.txt'), tok)
    tr = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True,
                    num_workers=2, pin_memory=True)
    va = DataLoader(va_ds, batch_size=a.batch_size, shuffle=False, num_workers=2)

    feat_dim = tr_ds[0][0].shape[-1]
    model = build_factual(tok, feat_dim).to(dev)
    params, has_proj = factual_params(model)
    print(f"[model] feat_dim {feat_dim}  trainable "
          f"{sum(p.numel() for p in params)/1e6:.1f}M  "
          f"enc_to_dec_proj: {'yes' if has_proj else 'no (dims match)'}")

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, a.warmup, a.epochs * len(tr))
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')

    peek, peek_ids = va_ds.distinct_images(3)
    peek = peek.to(dev)
    gold = va_ds.first_caption()

    best = float('inf')
    for ep in range(a.epochs):
        model.train()
        for i, (f, ids, m, lab) in enumerate(tr):
            f, lab = f.to(dev, non_blocking=True), lab.to(dev, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                loss = model(encoder_outputs=encode(model, f), labels=lab).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad()
            if i % 200 == 0:
                print(f"ep{ep+1} [{i}/{len(tr)}] loss {loss.item():.3f} "
                      f"lr {sch.get_last_lr()[0]:.2e}", flush=True)

        model.eval(); s = n = 0
        with torch.no_grad():
            for f, ids, m, lab in va:
                s += model(encoder_outputs=encode(model, f.to(dev)),
                           labels=lab.to(dev)).loss.item()
                n += 1
        val = s / max(n, 1)
        print(f"[EPOCH {ep+1}] val {val:.4f}")

        with torch.no_grad():
            g = model.generate(encoder_outputs=encode(model, peek),
                               max_new_tokens=a.max_new, num_beams=4,
                               no_repeat_ngram_size=3, early_stopping=True)
        outs = tok.batch_decode(g, skip_special_tokens=True)
        for j, (iid, t) in enumerate(zip(peek_ids, outs)):
            print(f"  [{j+1}] {iid}\n      gen  {t}\n      gold {gold[iid]}")
        if len(set(outs)) < len(outs):
            print("  [WARN] identical captions for different images -- "
                  "the model is ignoring the encoder features")

        if val < best:
            best = val
            model.save_pretrained(os.path.join(a.save_dir, 'best'))
            tok.save_pretrained(os.path.join(a.save_dir, 'best'))
            print(f"[EPOCH {ep+1}] saved  (best {val:.4f})")
        else:
            print(f"[EPOCH {ep+1}] val rose ({val:.4f} > {best:.4f}) -- not saved")
    print(f"done. best val {best:.4f}   checkpoint {a.save_dir}/best")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--split_dir', default='/kaggle/working/splits')
    p.add_argument('--cache_dir', default='/kaggle/working/clip_feature_cache')
    p.add_argument('--save_dir',  default='/kaggle/working/p2_factual')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--max_new', type=int, default=48)
    main(p.parse_args())
