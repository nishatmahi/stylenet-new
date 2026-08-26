"""
train_factual.py — step 1. Image -> factual Bangla caption.

Ordinary captioning. Only the decoder trains; cached CLIP features go straight
into cross-attention. No projector, no auxiliary loss, no staging.
"""
import os, argparse, torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput
from data import CaptionData
from models import build_tokenizer, build_factual


def main(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.save_dir, exist_ok=True)
    tok = build_tokenizer()

    tr_ds = CaptionData(a.cache_dir, os.path.join(a.split_dir, 'factual_train.txt'), tok)
    va_ds = CaptionData(a.cache_dir, os.path.join(a.split_dir, 'factual_val.txt'), tok)
    tr = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True,
                    num_workers=2, pin_memory=True)
    va = DataLoader(va_ds, batch_size=a.batch_size, shuffle=False, num_workers=2)

    model = build_factual(tok).to(dev)
    print(f"  decoder {sum(p.numel() for p in model.decoder.parameters())/1e6:.1f}M trainable")

    opt = torch.optim.AdamW(model.decoder.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, a.warmup, a.epochs * len(tr))
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
    peek = va_ds.distinct_images(3).to(dev)

    best = float('inf')
    for ep in range(a.epochs):
        model.train()
        for i, (f, ids, m, lab) in enumerate(tr):
            f, lab = f.to(dev), lab.to(dev)
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                loss = model(encoder_outputs=BaseModelOutput(last_hidden_state=f),
                             labels=lab).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad()
            if i % 200 == 0:
                print(f"ep{ep+1} [{i}/{len(tr)}] loss {loss.item():.3f} "
                      f"lr {sch.get_last_lr()[0]:.2e}", flush=True)

        model.eval(); s = n = 0
        with torch.no_grad():
            for f, ids, m, lab in va:
                s += model(encoder_outputs=BaseModelOutput(
                    last_hidden_state=f.to(dev)), labels=lab.to(dev)).loss.item()
                n += 1
        val = s / max(n, 1)
        print(f"[EPOCH {ep+1}] val {val:.4f}")

        with torch.no_grad():
            ids = model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=peek),
                                 max_new_tokens=40, num_beams=4, no_repeat_ngram_size=3)
        for j, t in enumerate(tok.batch_decode(ids, skip_special_tokens=True)):
            print(f"  sample {j+1}: {t}")

        if val < best:
            best = val
            model.save_pretrained(os.path.join(a.save_dir, 'best'))
            tok.save_pretrained(os.path.join(a.save_dir, 'best'))
            print(f"[EPOCH {ep+1}] saved  (best {val:.4f})")
    print(f"done. best val {best:.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--split_dir', default='/kaggle/working/splits')
    p.add_argument('--cache_dir', default='/kaggle/working/clip_feature_cache')
    p.add_argument('--save_dir',  default='/kaggle/working/p2_factual')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--epochs', type=int, default=6)
    main(p.parse_args())
