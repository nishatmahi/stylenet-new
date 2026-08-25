"""
train_factual.py — the factual captioner. No style anything.

This is PPCap's "off-the-shelf factual image captioning model". In the paper
it is ClipCap, already trained, and never touched again. Here it is BanglaT5
plus your CLIP projector, trained once and then frozen for the rest of the
project.

It matters more than it looks. Table 4 of the paper: the factual model scores
CIDEr 86.0 and the full framework scores 68.1 — the factual model is the
ceiling everything else works under. The generative style discriminator can
only re-rank what this model proposes; it cannot add content this model never
saw.

    python train_factual.py \
        --split_dir /kaggle/working/splits \
        --cache_dir /kaggle/working/clip_feature_cache \
        --save_dir  /kaggle/working/factual
"""

import os
import argparse
import contextlib

import torch
from torch.optim.lr_scheduler import LambdaLR

from data import build_tokenizer, Encoder, factual_loader
from models import FactualCaptioner, load_compatible


def amp_context(args, cuda):
    """bf16 has fp32's exponent range so it cannot overflow; fp16 can, and T5
    is well known to be fragile in it. is_bf16_supported() returns True on
    Turing (T4) where bf16 is emulated rather than native — slower, but
    numerically safe, which is the trade we want. --force_bf16 keeps it on
    there; set 0 to A/B against fp16 and watch the GradScaler's scale."""
    major = torch.cuda.get_device_capability()[0] if cuda else 0
    use_bf16 = args.amp and cuda and (major >= 8 or args.force_bf16)
    use_fp16 = args.amp and cuda and not use_bf16
    if cuda:
        print(f"gpu: {torch.cuda.get_device_name(0)}  (SM {major}.x)")
    if use_bf16:
        ctx = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    elif use_fp16:
        ctx = lambda: torch.autocast('cuda', dtype=torch.float16)
    else:
        ctx = contextlib.nullcontext
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    print("autocast: " + ("bf16" if use_bf16 else
                          "fp16+GradScaler" if use_fp16 else "off (fp32)"))
    return ctx, scaler


def run_epoch(model, loader, opt, sched, device, args, epoch, amp, scaler):
    model.train()
    tot_cap = tot_v2l = 0.0
    n = 0
    opt.zero_grad(set_to_none=True)
    params = [p for g in opt.param_groups for p in g['params']]

    for i, (feats, ids, mask, labels) in enumerate(loader):
        feats = feats.to(device, non_blocking=True)
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with amp():
            l_cap, img_content = model.caption_loss(feats, labels)
            l_v2l = model.v2l_loss(img_content, ids, mask)
            loss = l_cap + args.w_v2l * l_v2l
        scaler.scale(loss / args.accum_steps).backward()

        if (i + 1) % args.accum_steps == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)

        tot_cap += l_cap.item()
        tot_v2l += l_v2l.item()
        n += 1
        if i % args.log_step == 0 or i == len(loader) - 1:
            print(f"Epoch [{epoch+1}] [{i}/{len(loader)}] "
                  f"cap {l_cap.item():.3f}  v2l {l_v2l.item():.3f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)

    opt.zero_grad(set_to_none=True)
    return tot_cap / max(n, 1), tot_v2l / max(n, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    s, n = 0.0, 0
    for feats, ids, mask, labels in loader:
        loss, _ = model.caption_loss(feats.to(device), labels.to(device))
        s += loss.item()
        n += 1
    model.train()
    return s / max(n, 1)


@torch.no_grad()
def show(model, feats, tokenizer):
    model.eval()
    ids = model.generate(feats)
    for i, t in enumerate(tokenizer.batch_decode(ids, skip_special_tokens=True)):
        print(f"  sample {i+1}: {t}")
    model.train()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    amp, scaler = amp_context(args, device.type == 'cuda')

    tokenizer, _ = build_tokenizer(args.t5_ckpt)
    enc = Encoder(tokenizer)

    train_dl, train_ds = factual_loader(
        args.cache_dir, os.path.join(args.split_dir, 'factual_train.txt'),
        enc, args.batch_size, shuffle=True, return_dataset=True)
    val_dl, val_ds = factual_loader(
        args.cache_dir, os.path.join(args.split_dir, 'factual_val.txt'),
        enc, args.batch_size, shuffle=False, return_dataset=True)

    model = FactualCaptioner(args.t5_ckpt, args.clip_dim).to(device)
    # The style tokens live in the style model's vocabulary, not this one, but
    # the tokenizer is shared, so keep the embedding table the same size.
    model.t5.resize_token_embeddings(len(tokenizer))

    proj = [p for n_, p in model.named_parameters() if n_.startswith('projector.')]
    rest = [p for n_, p in model.named_parameters() if not n_.startswith('projector.')]
    opt = torch.optim.AdamW([{'params': rest, 'lr': args.lr},
                             {'params': proj, 'lr': args.lr_proj}],
                            weight_decay=0.01)
    sched = LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))
    print(f"  language model {sum(p.numel() for p in rest)/1e6:.1f}M")
    print(f"  projector      {sum(p.numel() for p in proj)/1e6:.1f}M")

    start, best, patience = 0, float('inf'), 0
    ck_path = os.path.join(args.save_dir, 'checkpoint-latest.pth')
    best_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device)
        load_compatible(model, ck['model'])
        try:
            opt.load_state_dict(ck['opt'])
            sched.load_state_dict(ck['sched'])
            scaler.load_state_dict(ck['scaler'])
        except Exception as e:
            print(f"[WARN] optimizer state unusable ({e}); starting fresh")
        start, best, patience = ck['epoch'] + 1, ck['best'], ck['patience']
        print(f"[resume] epoch {start+1}, best {best:.4f}")

    peek = val_ds.distinct_images(3).to(device)

    for epoch in range(start, args.epochs):
        print(f"\n=== epoch {epoch+1}/{args.epochs} ===")
        cap, v2l = run_epoch(model, train_dl, opt, sched, device, args,
                             epoch, amp, scaler)
        print(f"[EPOCH {epoch+1}] cap {cap:.4f}  v2l {v2l:.4f}")
        print(f"[EPOCH {epoch+1}] samples:")
        show(model, peek, tokenizer)

        val = validate(model, val_dl, device)
        print(f"[EPOCH {epoch+1}] val caption loss: {val:.4f}")

        state = {'epoch': epoch, 'model': model.state_dict(),
                 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                 'scaler': scaler.state_dict(), 'best': best,
                 'patience': patience, 'clip_dim': args.clip_dim,
                 't5_ckpt': args.t5_ckpt}
        if val < best:
            best, patience = val, 0
            state['best'], state['patience'] = best, patience
            torch.save(state, best_path)
            print(f"[EPOCH {epoch+1}] new best {val:.4f}")
        else:
            patience += 1
            state['patience'] = patience
            print(f"[EPOCH {epoch+1}] no improvement {patience}/{args.patience}")
        torch.save(state, ck_path)
        if patience >= args.patience:
            print(f"[early stop] best {best:.4f}")
            break

    print(f"\nfinished. best val caption loss {best:.4f}")
    print("This model is now frozen for the rest of the project.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--split_dir', default='/kaggle/working/splits')
    p.add_argument('--cache_dir', default='/kaggle/working/clip_feature_cache')
    p.add_argument('--save_dir', default='/kaggle/working/factual')
    p.add_argument('--t5_ckpt', default='csebuetnlp/banglat5')
    p.add_argument('--clip_dim', type=int, default=768)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--lr_proj', type=float, default=1e-3)
    p.add_argument('--w_v2l', type=float, default=1.0)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--amp', type=int, default=1)
    p.add_argument('--force_bf16', type=int, default=1)
    p.add_argument('--log_step', type=int, default=200)
    main(p.parse_args())
