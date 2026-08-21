import os
import contextlib
import argparse
import torch
from torch.optim.lr_scheduler import LambdaLR

from data_loader import factual_loader, style_loader, tokenizer
from models import StyleNetT5, load_compatible


def warmup_only(warmup):
    """Linear warmup, then flat. Deliberately NOT tied to --epochs: the LR
    schedule must not change depending on how long you plan to run, so a
    3-epoch trial and a 30-epoch run see identical optimization for their
    shared steps."""
    return lambda step: min(1.0, (step + 1) / max(1, warmup))


def build_optimizers(model, args):
    proj, rest = [], []
    for n, p in model.named_parameters():
        if n.startswith("style.") or n.startswith("modulator."):
            continue
        (proj if n.startswith("projector.") else rest).append(p)

    Opt = torch.optim.AdamW
    if args.adam8bit:
        import bitsandbytes as bnb
        Opt = bnb.optim.AdamW8bit

    opt_cap = Opt([
        {'params': rest, 'lr': args.lr},
        {'params': proj, 'lr': args.lr_proj},
    ], weight_decay=0.01)

    opt_style = Opt([model.style[s] for s in model.styles],
                    lr=args.lr_style, weight_decay=0.0)

    # The modulator gets its OWN optimizer and is stepped in BOTH passes.
    # It must learn where style applies, which needs stylized data (style
    # pass), but it sits on the image path, so it also needs image-grounded
    # gradient (caption pass) or it drifts into text-only behaviour. One
    # optimizer keeps a single consistent moment estimate across both.
    opt_mod = Opt(model.modulator.parameters(),
                  lr=args.lr_style, weight_decay=0.0)

    print(f"  language model: {sum(p.numel() for p in rest)/1e6:.1f}M")
    print(f"  projector: {sum(p.numel() for p in proj)/1e6:.1f}M")
    print(f"  style vectors: {sum(model.style[s].numel() for s in model.styles)}")
    print(f"  modulator: {sum(p.numel() for p in model.modulator.parameters())/1e6:.2f}M")
    return opt_cap, opt_style, opt_mod


def run_caption_pass(model, loader, opt, opt_mod, sched, sched_mod,
                     device, args, epoch, amp, scaler):
    """Task 1 — image captioning. Modulator trains here too, on image content
    with s_factual, which is what keeps it image-grounded. Style VECTORS stay
    frozen: opt_cap excludes them, so their .grad would never be cleared and
    would inflate clip_grad_norm_ across the epoch."""
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    for s in model.styles:
        model.style[s].requires_grad = False

    tot = {'cap': 0.0, 'v2l': 0.0}
    n = 0
    opt.zero_grad(set_to_none=True)
    opt_mod.zero_grad(set_to_none=True)
    opt_params = [p for g in opt.param_groups for p in g['params']]
    mod_params = list(model.modulator.parameters())

    for i, (feats, cap_ids, cap_mask, cap_labels) in enumerate(loader):
        feats = feats.to(device, non_blocking=True)
        cap_ids = cap_ids.to(device, non_blocking=True)
        cap_mask = cap_mask.to(device, non_blocking=True)
        cap_labels = cap_labels.to(device, non_blocking=True)

        with amp():
            l_cap, img_content = model.caption_loss(feats, cap_labels)
            l_v2l = model.v2l_loss(img_content, cap_ids, cap_mask)
            loss = l_cap + args.w_v2l * l_v2l
        scaler.scale(loss / args.accum_steps).backward()

        if (i + 1) % args.accum_steps == 0:
            # unscale before clipping, or the clip threshold is applied to
            # gradients still multiplied by the loss scale
            scaler.unscale_(opt)
            scaler.unscale_(opt_mod)
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            torch.nn.utils.clip_grad_norm_(mod_params, 1.0)
            scaler.step(opt)
            scaler.step(opt_mod)
            scaler.update()
            sched.step()
            sched_mod.step()
            opt.zero_grad(set_to_none=True)
            opt_mod.zero_grad(set_to_none=True)

        tot['cap'] += l_cap.item()
        tot['v2l'] += l_v2l.item()
        n += 1

        if i % args.log_step == 0 or i == len(loader) - 1:
            print(f"Epoch [{epoch+1}/{args.epochs}] CAP [{i}/{len(loader)}] "
                  f"cap {l_cap.item():.3f}  v2l {l_v2l.item():.3f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)

    # drop any gradient left over from a trailing partial accumulation window
    opt.zero_grad(set_to_none=True)
    opt_mod.zero_grad(set_to_none=True)
    return {k: v / max(n, 1) for k, v in tot.items()}


def run_style_pass(model, loader, opt_style, opt_mod, sched_style, sched_mod,
                   device, args, epoch, style, max_steps, amp, scaler):
    """Task 2 — text style injection on the monolingual corpus.

    Only s_style and the modulator are unfrozen. The encoder, decoder and
    projector stay frozen: this pass trains on text with no images, so any
    gradient into those weights teaches the decoder to generate without
    consulting memory — the counterpart of StyleNet excluding encoder.A.

    The modulator is on the image path, but three things contain that: its
    delta is hard-bounded to alpha * ||content|| per position, it also receives
    image-grounded gradient in the caption pass, and max_steps is chosen so the
    two gradient sources are within ~1x of each other (see main()). The old
    default of 1024 steps per style, stepping every iteration with no
    accumulation, gave the modulator roughly 9x more text-only gradient than
    image gradient — the bound limited the damage but the module was still
    overwhelmingly shaped by text.

    Step counts are also equalized across styles: factualtext_train.txt has
    156k lines against romantic's 32k, and since romantic is an offset FROM
    factual, an uncapped loop moved the origin 4.8x faster than the target.
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    model.style[style].requires_grad = True
    for p in model.modulator.parameters():
        p.requires_grad = True

    params = [model.style[style]] + list(model.modulator.parameters())
    is_factual = (style == "factual")

    tot, n = 0.0, 0
    for i, (c_ids, c_mask, labels) in enumerate(loader):
        if i >= max_steps:
            break
        # factual IS the origin of the interpolation, so _style_at ignores lam
        # for it. Don't sample or print a value that has no effect.
        lam = 1.0 if is_factual else float(
            torch.empty(1).uniform_(args.lam_min, args.lam_max).item())

        opt_style.zero_grad(set_to_none=True)
        opt_mod.zero_grad(set_to_none=True)
        with amp():
            loss = model.text_style_loss(c_ids.to(device), c_mask.to(device),
                                         labels.to(device), style, lam=lam)
        scaler.scale(loss).backward()
        scaler.unscale_(opt_style)
        scaler.unscale_(opt_mod)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt_style)
        scaler.step(opt_mod)
        scaler.update()
        sched_style.step()
        sched_mod.step()

        tot += loss.item()
        n += 1

        if i % args.log_step == 0 or i == max_steps - 1:
            lam_s = "" if is_factual else f" lam {lam:.2f}"
            print(f"Epoch [{epoch+1}/{args.epochs}] {style.upper()[:3]} "
                  f"[{i}/{max_steps}] loss {loss.item():.3f}{lam_s}", flush=True)

    for p in model.parameters():
        p.requires_grad = True
    return tot / max(n, 1)


@torch.no_grad()
def validate(model, fac_loader, style_loaders, device):
    model.eval()
    cap_sum, n = 0.0, 0
    for feats, ids, mask, labels in fac_loader:
        loss, _ = model.caption_loss(feats.to(device), labels.to(device))
        cap_sum += loss.item()
        n += 1
    cap = cap_sum / max(n, 1)
    print(f"  val caption loss: {cap:.4f}")

    for style, loader in style_loaders.items():
        s, m = 0.0, 0
        for c_ids, c_mask, labels in loader:
            s += model.text_style_loss(c_ids.to(device), c_mask.to(device),
                                       labels.to(device), style, lam=1.0).item()
            m += 1
        print(f"  val {style} text loss: {s/max(m,1):.4f}")

    model.train()
    return cap


@torch.no_grad()
def show_samples(model, feats, styles, lams):
    model.eval()
    print("    geometry: " + "  ".join(f"{k}={v:.3f}"
                                       for k, v in model.style_geometry().items()))
    for s in styles:
        if s == "factual":
            continue
        gm, gs, ratio = model.gate_report(feats, target=s, lam=1.0)
        print(f"    gate[{s}]: mean={gm:.3f} std={gs:.3f} "
              f"delta/content={ratio:.3f}  "
              f"(std ~0 = degenerated to a global vector)")

    for i in range(feats.size(0)):
        print(f"  --- sample {i+1} ---")
        for style in styles:
            for lam in ([0.0] if style == "factual" else lams):
                ids = model.generate(feats[i:i+1], target=style, lam=lam,
                                     num_beams=5)
                txt = tokenizer.decode(ids[0], skip_special_tokens=True)
                tag = style if style == "factual" else f"{style} λ={lam}"
                print(f"    [{tag}] {txt}")
    model.train()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # bf16 where the card supports it (A100/L4/H100) — no GradScaler needed.
    # A Kaggle T4 has no bf16, so it gets fp16 + GradScaler rather than falling
    # back to fp32 and leaving ~2x speed on the table.
    cuda = device.type == 'cuda'
    use_bf16 = args.amp and cuda and torch.cuda.is_bf16_supported()
    use_fp16 = args.amp and cuda and not use_bf16
    if use_bf16:
        amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    elif use_fp16:
        amp = lambda: torch.autocast('cuda', dtype=torch.float16)
    else:
        amp = contextlib.nullcontext
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    except (AttributeError, TypeError):          # torch < 2.4
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    print(f"autocast: {'bf16' if use_bf16 else 'fp16+GradScaler' if use_fp16 else 'off (fp32)'}")

    sd = args.split_dir
    styles = [s.strip() for s in args.styles.split(',') if s.strip()]
    all_styles = ["factual"] + styles
    print(f"styles: {all_styles}")

    train_fac = factual_loader(args.cache_dir,
                               os.path.join(sd, 'factual_train.txt'),
                               args.batch_size, shuffle=True,
                               captions_per_epoch=args.captions_per_epoch)
    val_fac = factual_loader(args.cache_dir,
                             os.path.join(sd, 'factual_val.txt'),
                             args.batch_size, shuffle=False)

    text_stem = {"factual": "factualtext"}
    train_style_loaders, val_style_loaders = {}, {}
    for s in all_styles:
        stem = text_stem.get(s, s)
        train_style_loaders[s] = style_loader(
            os.path.join(sd, f'{stem}_train.txt'), args.style_batch_size)
        val_style_loaders[s] = style_loader(
            os.path.join(sd, f'{stem}_val.txt'), args.style_batch_size,
            shuffle=False)

    # ---- balance the modulator's two gradient sources --------------------
    # The caption pass takes one optimizer step per accum_steps batches; the
    # style pass takes one per batch. Match total style steps to total caption
    # steps, split evenly across styles, so the modulator is not dominated by
    # text-only gradient.
    cap_opt_steps = max(1, len(train_fac) // args.accum_steps)
    if args.max_style_steps > 0:
        style_steps = args.max_style_steps
    else:
        style_steps = max(1, cap_opt_steps // len(all_styles))
    print(f"steps/epoch: cap={len(train_fac)} batches "
          f"({cap_opt_steps} opt steps), style={style_steps} per style "
          f"x {len(all_styles)} styles = {style_steps*len(all_styles)}")

    model = StyleNetT5(t5_ckpt=args.t5_ckpt, clip_dim=args.clip_dim,
                       styles=tuple(all_styles), alpha=args.alpha).to(device)

    opt_cap, opt_style, opt_mod = build_optimizers(model, args)

    sched = LambdaLR(opt_cap, warmup_only(args.warmup))
    sched_style = LambdaLR(opt_style, warmup_only(args.warmup))
    sched_mod = LambdaLR(opt_mod, warmup_only(args.warmup))

    start, best, patience = 0, float('inf'), 0
    ckpt_path = os.path.join(args.save_dir, 'checkpoint-latest.pth')
    best_path = os.path.join(args.save_dir, 'best_model.pth')

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        load_compatible(model, ck['model'])
        try:
            opt_cap.load_state_dict(ck['opt_cap'])
            opt_style.load_state_dict(ck['opt_style'])
            opt_mod.load_state_dict(ck['opt_mod'])
            sched.load_state_dict(ck['sched'])
            sched_style.load_state_dict(ck['sched_style'])
            sched_mod.load_state_dict(ck['sched_mod'])
            if 'scaler' in ck:
                scaler.load_state_dict(ck['scaler'])
        except Exception as e:
            print(f"[WARN] optimizer state unusable ({e}); starting fresh")
        start = ck['epoch'] + 1
        best = ck.get('best', float('inf'))
        patience = ck.get('patience', 0)
        print(f"[resume] from epoch {ck['epoch']+1}, best {best:.4f}")

    lams = [float(x) for x in args.lams.split(',')]
    last_feats = None

    for epoch in range(start, args.epochs):
        print(f"\n=== epoch {epoch+1}/{args.epochs} ===")
        # Only relevant when --captions_per_epoch > 0. Works under
        # persistent_workers: the sampler lives in this process.
        if train_fac.sampler is not None and hasattr(train_fac.sampler, 'set_epoch'):
            train_fac.sampler.set_epoch(epoch)

        cap_avg = run_caption_pass(model, train_fac, opt_cap, opt_mod, sched,
                                   sched_mod, device, args, epoch, amp, scaler)
        style_avg = {}
        for s in all_styles:
            style_avg[s] = run_style_pass(model, train_style_loaders[s],
                                          opt_style, opt_mod, sched_style,
                                          sched_mod, device, args, epoch, s,
                                          style_steps, amp, scaler)

        print(f"[EPOCH {epoch+1}] cap {cap_avg['cap']:.4f}  "
              f"v2l {cap_avg['v2l']:.4f}  " +
              "  ".join(f"{s} {v:.4f}" for s, v in style_avg.items()))

        if last_feats is None:
            last_feats = next(iter(val_fac))[0][:3].to(device)
        print(f"[EPOCH {epoch+1}] samples:")
        show_samples(model, last_feats, all_styles, lams)

        print(f"[EPOCH {epoch+1}] validation:")
        val = validate(model, val_fac, val_style_loaders, device)

        state = {'epoch': epoch, 'model': model.state_dict(),
                 'opt_cap': opt_cap.state_dict(),
                 'opt_style': opt_style.state_dict(),
                 'opt_mod': opt_mod.state_dict(),
                 'sched': sched.state_dict(),
                 'sched_style': sched_style.state_dict(),
                 'sched_mod': sched_mod.state_dict(),
                 'scaler': scaler.state_dict(),
                 'best': best, 'patience': patience, 'styles': all_styles}

        if val < best:
            best, patience = val, 0
            state['best'], state['patience'] = best, patience
            torch.save(state, best_path)
            print(f"[EPOCH {epoch+1}] new best {val:.4f}")
        else:
            patience += 1
            state['patience'] = patience
            print(f"[EPOCH {epoch+1}] no improvement {patience}/{args.patience}")

        torch.save(state, ckpt_path)
        if patience >= args.patience:
            print(f"[early stop] best {best:.4f}")
            break

    print(f"\nfinished. best val caption loss {best:.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--split_dir', default='/kaggle/working/splits')
    p.add_argument('--cache_dir', default='/kaggle/working/clip_feature_cache')
    p.add_argument('--save_dir', default='/kaggle/working/stylenet_clip')
    p.add_argument('--t5_ckpt', default='csebuetnlp/banglat5')
    p.add_argument('--clip_dim', type=int, default=768)
    # train both styles by default; the old default trained romantic only
    p.add_argument('--styles', default='romantic,humorous')
    # bs 32 / accum 4 keeps the effective caption batch at 128. Try 48/3 on a
    # T4; drop to 16/8 if you OOM. Keep accum_steps ~= 128 / batch_size.
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--style_batch_size', type=int, default=32)
    p.add_argument('--accum_steps', type=int, default=4)
    # 0 (default) = every caption every epoch. Set to e.g. 35000 only if you
    # want shorter epochs; it changes nothing else.
    p.add_argument('--captions_per_epoch', type=int, default=0)

    # 5e-4 on the full 247M-param backbone erodes the pretrained Bangla that
    # BanglaT5 was chosen for. 5e-5 with warmup, then flat.
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--lr_proj', type=float, default=1e-3)
    p.add_argument('--lr_style', type=float, default=1e-3)
    p.add_argument('--w_v2l', type=float, default=1.0)

    # hard cap on style perturbation: ||delta|| <= alpha * ||content||
    # per position, so content cannot be overwritten
    p.add_argument('--alpha', type=float, default=0.5)

    p.add_argument('--lam_min', type=float, default=0.5)
    p.add_argument('--lam_max', type=float, default=2.5)
    # 0 = auto: match total style steps to total caption optimizer steps
    p.add_argument('--max_style_steps', type=int, default=0)

    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--adam8bit', type=int, default=0)
    p.add_argument('--amp', type=int, default=1)
    p.add_argument('--log_step', type=int, default=200)
    p.add_argument('--lams', default='1.0,1.5,2.0')
    args = p.parse_args()
    main(args)
