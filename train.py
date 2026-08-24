"""
train.py — one style per run.

    python train.py --style romantic
    python train.py --style humorous --save_dir /kaggle/working/stylenet_hum

.

--epochs only decides when the loop stops. The learning rate warms up and then
stays flat, so a 3-epoch trial and a 30-epoch run see identical optimization
for the steps they share.
"""

import os
import contextlib
import argparse
import torch
from torch.optim.lr_scheduler import LambdaLR

from data_loader import factual_loader, style_loader, tokenizer
from models import StyleNetT5, load_compatible


def warmup_only(warmup):
    return lambda step: min(1.0, (step + 1) / max(1, warmup))


def build_optimizers(model, args):
    proj, rest = [], []
    for n, p in model.named_parameters():
        if ".adapter." in n:
            continue
        (proj if n.startswith("projector.") else rest).append(p)

    Opt = torch.optim.AdamW
    if args.adam8bit:
        import bitsandbytes as bnb
        Opt = bnb.optim.AdamW8bit

    opt_cap = Opt([{'params': rest, 'lr': args.lr},
                   {'params': proj, 'lr': args.lr_proj}], weight_decay=0.01)
    opt_sty = Opt(model.adapter_parameters(), lr=args.lr_style, weight_decay=0.0)

    print(f"  language model {sum(p.numel() for p in rest)/1e6:.1f}M")
    print(f"  projector      {sum(p.numel() for p in proj)/1e6:.1f}M")
    print(f"  style knobs    {sum(p.numel() for p in model.adapter_parameters())/1e6:.2f}M")
    return opt_cap, opt_sty


def run_caption_pass(model, loader, opt, sched, device, args, epoch, amp, scaler):
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    for p in model.adapter_parameters():
        p.requires_grad = False          # knobs are not trained on images

    tot = {'cap': 0.0, 'v2l': 0.0}
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
            scaler.unscale_(opt)                    # unscale before clipping
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)

        tot['cap'] += l_cap.item()
        tot['v2l'] += l_v2l.item()
        n += 1
        if i % args.log_step == 0 or i == len(loader) - 1:
            print(f"Epoch [{epoch+1}] CAP [{i}/{len(loader)}] "
                  f"cap {l_cap.item():.3f}  v2l {l_v2l.item():.3f}  "
                  f"scale {scaler.get_scale():.0f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)

    opt.zero_grad(set_to_none=True)
    return {k: v / max(n, 1) for k, v in tot.items()}


def run_style_pass(model, loader, opt, sched, device, args, epoch,
                   max_steps, amp, scaler):
    """Task 2. lam is FIXED at 1.0 here. Sampling it while demanding the same
    target at every value trains the adapter to be lam-invariant, i.e. to
    shrink. The dial belongs at inference."""
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.adapter_parameters():
        p.requires_grad = True

    params = model.adapter_parameters()
    tot, n = 0.0, 0
    for i, labels in enumerate(loader):
        if i >= max_steps:
            break
        opt.zero_grad(set_to_none=True)
        with amp():
            loss = model.lm_style_loss(labels.to(device), lam=1.0)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        tot += loss.item()
        n += 1
        if i % args.log_step == 0 or i == max_steps - 1:
            print(f"Epoch [{epoch+1}] STY [{i}/{max_steps}] "
                  f"loss {loss.item():.3f}", flush=True)

    for p in model.parameters():
        p.requires_grad = True
    return tot / max(n, 1)


@torch.no_grad()
def validate(model, fac_loader, sty_loader, device):
    model.eval()
    s, n = 0.0, 0
    for feats, ids, mask, labels in fac_loader:
        loss, _ = model.caption_loss(feats.to(device), labels.to(device))
        s += loss.item(); n += 1
    cap = s / max(n, 1)
    print(f"  val caption loss: {cap:.4f}")

    s, n = 0.0, 0
    for labels in sty_loader:
        s += model.lm_style_loss(labels.to(device), lam=1.0).item()
        n += 1
    print(f"  val style loss:   {s/max(n,1):.4f}")
    model.train()
    return cap


@torch.no_grad()
def show_samples(model, feats, style, lams):
    """One pass per lam, the same way sample.py runs it."""
    model.eval()
    print(f"    knob strength |delta|/|h| = "
          f"{model.adapter_report(feats[:1], lam=1.0):.4f}   "
          f"(~0 = knobs dead, raise --lr_style; 0.05-0.4 healthy)")

    def gen(lam):
        ids = model.generate(feats, lam=lam, num_beams=5, max_new_tokens=48,
                             repetition_penalty=1.15, no_repeat_ngram_size=3)
        return tokenizer.batch_decode(ids, skip_special_tokens=True)

    fac = gen(0.0)
    outs = {lam: gen(lam) for lam in lams}
    for i in range(feats.size(0)):
        print(f"  --- sample {i+1} ---")
        print(f"    [factual] {fac[i]}")
        for lam in lams:
            print(f"    [{style} λ={lam}] {outs[lam][i]}")
    model.train()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"device: {device}   style: {args.style}")

    cuda = device.type == 'cuda'
    # torch.cuda.is_bf16_supported() returns True on Turing (T4, SM 7.5) where
    # bf16 is emulated. Real bf16 starts at Ampere, SM 8.0. But the run that
    # produced good factual captions used bf16 on a T4, and T5 is unstable in
    # fp16, so --force_bf16 defaults to on. Set 0 to A/B it and watch `scale`.
    major = torch.cuda.get_device_capability()[0] if cuda else 0
    use_bf16 = args.amp and cuda and (major >= 8 or args.force_bf16)
    use_fp16 = args.amp and cuda and not use_bf16
    if cuda:
        print(f"gpu: {torch.cuda.get_device_name(0)}  (SM {major}.x)")
    if use_bf16:
        amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    elif use_fp16:
        amp = lambda: torch.autocast('cuda', dtype=torch.float16)
    else:
        amp = contextlib.nullcontext
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    print("autocast: " + ("bf16" if use_bf16 else
                          "fp16+GradScaler" if use_fp16 else "off (fp32)"))

    sd = args.split_dir
    train_fac = factual_loader(
        args.cache_dir, os.path.join(sd, 'factual_train.txt'), args.batch_size,
        shuffle=True, captions_per_epoch=args.captions_per_epoch)
    val_fac = factual_loader(args.cache_dir,
                             os.path.join(sd, 'factual_val.txt'),
                             args.batch_size, shuffle=False)

    train_sty = style_loader(os.path.join(sd, f'{args.style}_train.txt'),
                             args.style_batch_size)
    val_sty = style_loader(os.path.join(sd, f'{args.style}_val.txt'),
                           args.style_batch_size, shuffle=False)

    cap_opt_steps = max(1, len(train_fac) // args.accum_steps)
    sty_steps = args.max_style_steps if args.max_style_steps > 0 else cap_opt_steps
    print(f"steps/epoch: cap {len(train_fac)} batches ({cap_opt_steps} opt steps), "
          f"style {sty_steps}")

    model = StyleNetT5(t5_ckpt=args.t5_ckpt, clip_dim=args.clip_dim,
                       style=args.style, bottleneck=args.bottleneck).to(device)
    opt_cap, opt_sty = build_optimizers(model, args)
    sched_cap = LambdaLR(opt_cap, warmup_only(args.warmup))
    sched_sty = LambdaLR(opt_sty, warmup_only(args.warmup))

    start, best, patience = 0, float('inf'), 0
    ck_path = os.path.join(args.save_dir, 'checkpoint-latest.pth')
    best_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device)
        load_compatible(model, ck['model'])
        try:
            opt_cap.load_state_dict(ck['opt_cap'])
            opt_sty.load_state_dict(ck['opt_sty'])
            sched_cap.load_state_dict(ck['sched_cap'])
            sched_sty.load_state_dict(ck['sched_sty'])
            if 'scaler' in ck:
                scaler.load_state_dict(ck['scaler'])
        except Exception as e:
            print(f"[WARN] optimizer state unusable ({e}); starting fresh")
        start = ck['epoch'] + 1
        best = ck.get('best', float('inf'))
        patience = ck.get('patience', 0)
        print(f"[resume] from epoch {start+1}, best {best:.4f}")

    lams = [float(x) for x in args.lams.split(',')]
    peek = None

    for epoch in range(start, args.epochs):
        print(f"\n=== epoch {epoch+1}/{args.epochs} ===")
        if train_fac.sampler is not None and hasattr(train_fac.sampler, 'set_epoch'):
            train_fac.sampler.set_epoch(epoch)

        cap = run_caption_pass(model, train_fac, opt_cap, sched_cap,
                               device, args, epoch, amp, scaler)
        sty = run_style_pass(model, train_sty, opt_sty, sched_sty,
                             device, args, epoch, sty_steps, amp, scaler)
        print(f"[EPOCH {epoch+1}] cap {cap['cap']:.4f}  v2l {cap['v2l']:.4f}  "
              f"style {sty:.4f}")

        if peek is None:
            # FactualDataset stores ~4.8 consecutive rows per image, so
            # [:3] off a batch is the SAME picture three times. Walk the
            # rows and take three DISTINCT images.
            seen, f3 = set(), []
            for i, (img, _) in enumerate(val_fac.dataset.rows):
                if img in seen:
                    continue
                seen.add(img)
                f3.append(val_fac.dataset[i][0])
                if len(f3) == 3:
                    break
            peek = torch.stack(f3).to(device)
        print(f"[EPOCH {epoch+1}] samples:")
        show_samples(model, peek, args.style, lams)

        print(f"[EPOCH {epoch+1}] validation:")
        val = validate(model, val_fac, val_sty, device)

        state = {'epoch': epoch, 'model': model.state_dict(),
                 'opt_cap': opt_cap.state_dict(), 'opt_sty': opt_sty.state_dict(),
                 'sched_cap': sched_cap.state_dict(),
                 'sched_sty': sched_sty.state_dict(),
                 'scaler': scaler.state_dict(),
                 'best': best, 'patience': patience,
                 'style': args.style, 'clip_dim': args.clip_dim,
                 'bottleneck': args.bottleneck}
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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--style', default='romantic',
                   help='ONE style per run. reads <style>_train.txt / _val.txt')
    p.add_argument('--split_dir', default='/kaggle/working/splits')
    p.add_argument('--cache_dir', default='/kaggle/working/clip_feature_cache')
    p.add_argument('--save_dir', default='/kaggle/working/stylenet_clip')
    p.add_argument('--t5_ckpt', default='csebuetnlp/banglat5')
    p.add_argument('--clip_dim', type=int, default=768)
    p.add_argument('--bottleneck', type=int, default=64)

    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--style_batch_size', type=int, default=32)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--captions_per_epoch', type=int, default=0)

    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--lr_proj', type=float, default=1e-3)
    p.add_argument('--lr_style', type=float, default=1e-3)
    p.add_argument('--w_v2l', type=float, default=1.0)

    p.add_argument('--max_style_steps', type=int, default=0)   # 0 = match cap

    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--adam8bit', type=int, default=0)
    p.add_argument('--amp', type=int, default=1)
    p.add_argument('--force_bf16', type=int, default=1)
    p.add_argument('--log_step', type=int, default=200)
    p.add_argument('--lams', default='1.0,1.5,2.0')
    main(p.parse_args())
