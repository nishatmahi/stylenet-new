"""
train_style.py — the text-only GeDi style discriminator.

No CLIP, no image, no noise. Reads the style corpus and the factual text corpus
as plain text. Validation reports the loss AND the discrimination accuracy: on
held-out sentences, how often the TRUE control code gives a higher likelihood
than the flipped one. That number, not free generation, is what tells you the
discriminator can tell the styles apart.
"""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse, torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from data import StyleData, add_style_tokens
from models import build_tokenizer, StyleModel, ppcap_loss, seq_logprob


def main(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.save_dir, exist_ok=True)
    tok = build_tokenizer()
    sid = add_style_tokens(tok)
    print("style control tokens:", sid)

    tr_ds = StyleData(a.style_train, a.factual_train, a.style, tok, sid, ratio=a.ratio)
    va_ds = StyleData(a.style_val, a.factual_val, a.style, tok, sid, ratio=a.ratio)
    tr = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True, num_workers=2)
    va = DataLoader(va_ds, batch_size=a.batch_size, shuffle=False, num_workers=2)

    model = StyleModel(tok).to(dev)
    print(f"  style model {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    steps_per_ep = (len(tr) + a.accum - 1) // a.accum
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, a.warmup, a.epochs * steps_per_ep)
    print(f"  batch {a.batch_size} x accum {a.accum} = effective {a.batch_size*a.accum}"
          f"   {steps_per_ep} optimiser steps/epoch")
    desired, undesired = sid[a.style], sid['factual']

    def flip_of(s_id):
        return torch.where(s_id == desired,
                           torch.full_like(s_id, undesired),
                           torch.full_like(s_id, desired))

    best = float('inf')
    for ep in range(a.epochs):
        model.train()
        for i, (ids, m, s_id, is_sty) in enumerate(tr):
            ids, m, s_id = ids.to(dev), m.to(dev), s_id.to(dev)
            loss, lg, ld = ppcap_loss(model, s_id, flip_of(s_id), ids, m, lam=a.lam)
            (loss / a.accum).backward()
            if (i + 1) % a.accum == 0 or (i + 1) == len(tr):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
            if i % 100 == 0:
                print(f"ep{ep+1} [{i}/{len(tr)}] loss {loss.item():.3f} "
                      f"gen {lg.item():.3f} disc {ld.item():.3f}", flush=True)

        model.eval(); s = n = 0; correct = tot = 0
        with torch.no_grad():
            for ids, m, s_id, is_sty in va:
                ids, m, s_id = ids.to(dev), m.to(dev), s_id.to(dev)
                fl = flip_of(s_id)
                l, _, _ = ppcap_loss(model, s_id, fl, ids, m, lam=a.lam)
                s += l.item(); n += 1
                lp_true = seq_logprob(model, s_id, ids, m)
                lp_flip = seq_logprob(model, fl, ids, m)
                correct += (lp_true > lp_flip).sum().item(); tot += ids.size(0)
        val = s / max(n, 1); acc = correct / max(tot, 1)
        print(f"[EPOCH {ep+1}] val {val:.4f}   disc_acc {acc:.3f}")

        # unconditional style sample (no image) — a sanity peek only
        with torch.no_grad():
            ys = torch.full((3, 1), tok.bos_token_id or tok.eos_token_id,
                            dtype=torch.long, device=dev)
            code = torch.full((3,), desired, device=dev)
            for _ in range(25):
                nxt = model.step_logits(code, ys).argmax(-1, keepdim=True)
                ys = torch.cat([ys, nxt], 1)
            for j, t in enumerate(tok.batch_decode(ys, skip_special_tokens=True)):
                print(f"    [{a.style} sample {j+1}] {t}")

        if val < best:
            best = val
            torch.save({'model': model.state_dict(), 'style': a.style},
                       os.path.join(a.save_dir, 'best_model.pth'))
            print(f"[EPOCH {ep+1}] saved  (best {val:.4f})")
        else:
            print(f"[EPOCH {ep+1}] val rose ({val:.4f} > {best:.4f}) -- not saved")
    print(f"done. best val {best:.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--style', default='romantic')
    p.add_argument('--style_train', required=True)
    p.add_argument('--factual_train', required=True)
    p.add_argument('--style_val', required=True)
    p.add_argument('--factual_val', required=True)
    p.add_argument('--save_dir', default='/kaggle/working/p2_style_romantic')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--ratio', type=float, default=1.0)
    p.add_argument('--lam', type=float, default=0.8)
    main(p.parse_args())
