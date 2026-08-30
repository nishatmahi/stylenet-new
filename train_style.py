"""
train_style.py — step 2. The style discriminator.


"""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse, torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from data import StyleData, add_style_tokens
from models import build_tokenizer, StyleModel, noise_injection, ppcap_loss


def main(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.save_dir, exist_ok=True)
    tok = build_tokenizer()
    sid = add_style_tokens(tok)
    print("style control tokens:", sid)

    tr_ds = StyleData(a.style_pt, a.factual_pt, a.style, tok, sid, ratio=a.ratio)
    va_ds = StyleData(a.style_val_pt, a.factual_val_pt, a.style, tok, sid, ratio=a.ratio)
    tr = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True, num_workers=2)
    va = DataLoader(va_ds, batch_size=a.batch_size, shuffle=False, num_workers=2)

    clip_dim = tr_ds.rows[0][0].numel()
    model = StyleModel(clip_dim, tok, prefix_len=a.prefix_len).to(dev)
    print(f"  style model {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    steps_per_ep = (len(tr) + a.accum - 1) // a.accum
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, a.warmup, a.epochs * steps_per_ep)
    print(f"  batch {a.batch_size} x accum {a.accum} = effective {a.batch_size*a.accum}"
          f"   {steps_per_ep} optimiser steps/epoch")
    desired, undesired = sid[a.style], sid['factual']

    best = float('inf')
    for ep in range(a.epochs):
        model.train()
        for i, (emb, ids, m, s_id, is_sty) in enumerate(tr):
            emb = noise_injection(emb.to(dev), a.variance, a.normalize_prefix)
            ids, m, s_id = ids.to(dev), m.to(dev), s_id.to(dev)
            flip = torch.where(s_id == desired,
                               torch.full_like(s_id, undesired),
                               torch.full_like(s_id, desired))
            loss, lg, ld = ppcap_loss(model, emb, s_id, flip, ids, m, lam=a.lam)
            (loss / a.accum).backward()
            if (i + 1) % a.accum == 0 or (i + 1) == len(tr):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
            if i % 100 == 0:
                print(f"ep{ep+1} [{i}/{len(tr)}] loss {loss.item():.3f} "
                      f"gen {lg.item():.3f} disc {ld.item():.3f}", flush=True)

        model.eval(); s = n = 0
        with torch.no_grad():
            for emb, ids, m, s_id, is_sty in va:
                emb = noise_injection(emb.to(dev), 0.0, a.normalize_prefix)
                ids, m, s_id = ids.to(dev), m.to(dev), s_id.to(dev)
                flip = torch.where(s_id == desired,
                                   torch.full_like(s_id, undesired),
                                   torch.full_like(s_id, desired))
                l, _, _ = ppcap_loss(model, emb, s_id, flip, ids, m, lam=a.lam)
                s += l.item(); n += 1
        val = s / max(n, 1)
        print(f"[EPOCH {ep+1}] val {val:.4f}")

        # THE check: same embedding, both control codes. They must differ.
        with torch.no_grad():
            e = noise_injection(torch.stack([tr_ds.rows[j][0] for j in (0, 1, 2)]).to(dev),
                                0.0, a.normalize_prefix)
            for name, code in ((a.style, desired), ('factual', undesired)):
                ids0 = torch.full((3, 1), tok.bos_token_id or tok.eos_token_id,
                                  dtype=torch.long, device=dev)
                out = _greedy(model, e, torch.full((3,), code, device=dev), ids0, 30)
                for j, t in enumerate(tok.batch_decode(out, skip_special_tokens=True)):
                    print(f"    [{name:8s} {j+1}] {t}")

        if val < best:
            best = val
            torch.save({'model': model.state_dict(), 'clip_dim': clip_dim,
                        'prefix_len': a.prefix_len, 'style': a.style},
                       os.path.join(a.save_dir, 'best_model.pth'))
            print(f"[EPOCH {ep+1}] saved  (best {val:.4f})")
        else:
            print(f"[EPOCH {ep+1}] val rose ({val:.4f} > {best:.4f}) -- not saved")
    print(f"done. best val {best:.4f}")


@torch.no_grad()
def _greedy(model, emb, style_ids, ys, steps):
    for _ in range(steps):
        nxt = model.step_logits(emb, style_ids, ys).argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], 1)
    return ys


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--style', default='romantic')
    p.add_argument('--style_pt', required=True)
    p.add_argument('--factual_pt', required=True)
    p.add_argument('--style_val_pt', required=True)
    p.add_argument('--factual_val_pt', required=True)
    p.add_argument('--save_dir', default='/kaggle/working/p2_style_romantic')
    p.add_argument('--batch_size', type=int, default=16)     # micro-batch
    p.add_argument('--accum', type=int, default=4)           # 16x4 = paper's 64
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--epochs', type=int, default=6)          # paper: 20
    p.add_argument('--prefix_len', type=int, default=10)
    p.add_argument('--ratio', type=float, default=1.0)       # factual : style
    p.add_argument('--lam', type=float, default=0.8)         # paper: 0.8
    p.add_argument('--variance', type=float, default=0.016)  # paper: 0.016
    p.add_argument('--normalize_prefix', type=int, default=1)
    main(p.parse_args())
