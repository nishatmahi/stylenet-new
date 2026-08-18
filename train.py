import os
import argparse
import torch
from torch.optim.lr_scheduler import LambdaLR

from data_loader import (factual_loader, style_loader, infinite,
                         tokenizer)
from models import StyleNetT5, load_compatible


def build_optimizer(model, args):
    proj, rest = [], []
    for n, p in model.named_parameters():
        (proj if n.startswith("projector.") or n.startswith("style.")
         else rest).append(p)

    Opt = torch.optim.AdamW
    if args.adam8bit:
        import bitsandbytes as bnb
        Opt = bnb.optim.AdamW8bit

    opt = Opt([
        {'params': rest, 'lr': args.lr},
        {'params': proj, 'lr': args.lr_proj},
    ], weight_decay=0.01)

    print(f"  language model: {sum(p.numel() for p in rest)/1e6:.1f}M")
    print(f"  projector + style vectors: {sum(p.numel() for p in proj)/1e6:.1f}M")
    return opt


def run_epoch(model, fac_loader, style_iters, opt, sched, device, args, epoch):
    """One backward pass per step covering captioning, V2L alignment, and text
    style injection together. FS-StyleCap's w/o MultiTask ablation — training
    these sequentially — gives CIDEr 2.45 against 66.26 for simultaneous. The
    alternating two-stage loop is exactly what produced identical captions on
    every image before."""
    model.train()
    tot = {'cap': 0.0, 'v2l': 0.0, 'txt': 0.0}
    n = 0

    for i, (feats, cap_ids, cap_mask, cap_labels) in enumerate(fac_loader):
        feats = feats.to(device, non_blocking=True)
        cap_ids = cap_ids.to(device, non_blocking=True)
        cap_mask = cap_mask.to(device, non_blocking=True)
        cap_labels = cap_labels.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        l_cap, img_content = model.caption_loss(feats, cap_labels)
        l_v2l = model.v2l_loss(img_content, cap_ids, cap_mask)

        l_txt = 0.0
        for style, it in style_iters.items():
            c_ids, c_mask, labels = next(it)
            l_txt = l_txt + model.text_style_loss(
                c_ids.to(device), c_mask.to(device), labels.to(device), style)
        l_txt = l_txt / len(style_iters)

        loss = l_cap + args.w_v2l * l_v2l + args.w_txt * l_txt
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        tot['cap'] += l_cap.item()
        tot['v2l'] += l_v2l.item()
        tot['txt'] += float(l_txt)
        n += 1

        if i % args.log_step == 0 or i == len(fac_loader) - 1:
            print(f"Epoch [{epoch+1}/{args.epochs}] Step [{i}/{len(fac_loader)}] "
                  f"cap {l_cap.item():.3f}  v2l {l_v2l.item():.3f}  "
                  f"txt {float(l_txt):.3f}  lr {opt.param_groups[0]['lr']:.2e}",
                  flush=True)

    return {k: v / max(n, 1) for k, v in tot.items()}


@torch.no_grad()
def validate(model, fac_loader, style_loaders, device):
    model.eval()
    cap_sum = n = 0.0
    for feats, ids, mask, labels in fac_loader:
        loss, _ = model.caption_loss(feats.to(device), labels.to(device))
        cap_sum += loss.item()
        n += 1
    cap = cap_sum / max(n, 1)
    print(f"  val caption loss: {cap:.4f}")

    for style, loader in style_loaders.items():
        s = m = 0.0
        for c_ids, c_mask, labels in loader:
            s += model.text_style_loss(c_ids.to(device), c_mask.to(device),
                                       labels.to(device), style).item()
            m += 1
        print(f"  val {style} text loss: {s/max(m,1):.4f}")

    model.train()
    return cap


@torch.no_grad()
def show_samples(model, feats, styles, lams, n=3):
    model.eval()
    f = feats[:n]
    for i in range(f.size(0)):
        print(f"  --- sample {i+1} ---")
        for style in styles:
            for lam in ([0.0] if style == "factual" else lams):
                ids = model.generate(f[i:i+1], target=style, lam=lam, num_beams=5)
                txt = tokenizer.decode(ids[0], skip_special_tokens=True)
                tag = style if style == "factual" else f"{style} λ={lam}"
                print(f"    [{tag}] {txt}")
    for s in styles:
        print(f"    ||s_{s}|| = {model.style[s].norm().item():.4f}")
    model.train()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    sd = args.split_dir
    styles = [s.strip() for s in args.styles.split(',') if s.strip()]
    all_styles = ["factual"] + styles
    print(f"styles: {all_styles}")

    train_fac = factual_loader(args.cache_dir,
                               os.path.join(sd, 'factual_train.txt'),
                               args.batch_size, shuffle=True)
    val_fac = factual_loader(args.cache_dir,
                             os.path.join(sd, 'factual_val.txt'),
                             args.batch_size, shuffle=False)

    # 'factual' on the text side uses the factual captions as bare text, so
    # s_factual is trained the same way the stylized vectors are.
    text_file = {"factual": "factualtext"}
    train_style_loaders, val_style_loaders = {}, {}
    for s in all_styles:
        stem = text_file.get(s, s)
        train_style_loaders[s] = style_loader(
            os.path.join(sd, f'{stem}_train.txt'), args.style_batch_size)
        val_style_loaders[s] = style_loader(
            os.path.join(sd, f'{stem}_val.txt'), args.style_batch_size,
            shuffle=False)
    style_iters = {s: infinite(l) for s, l in train_style_loaders.items()}

    print(f"steps/epoch: {len(train_fac)}")

    model = StyleNetT5(t5_ckpt=args.t5_ckpt, clip_dim=args.clip_dim,
                       styles=tuple(all_styles),
                       gradient_checkpointing=bool(args.grad_ckpt)).to(device)

    opt = build_optimizer(model, args)
    sched = LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))

    start, best, patience = 0, float('inf'), 0
    ckpt_path = os.path.join(args.save_dir, 'checkpoint-latest.pth')
    best_path = os.path.join(args.save_dir, 'best_model.pth')

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        load_compatible(model, ck['model'])
        try:
            opt.load_state_dict(ck['opt'])
            sched.load_state_dict(ck['sched'])
        except Exception as e:
            print(f"[WARN] optimizer state unusable ({e}); starting fresh")
        start = ck['epoch'] + 1
        best = ck.get('best', float('inf'))
        patience = ck.get('patience', 0)
        print(f"[resume] from epoch {ck['epoch']+1}, best {best:.4f}")

    last_feats = None
    for epoch in range(start, args.epochs):
        print(f"\n=== epoch {epoch+1}/{args.epochs} ===")
        avg = run_epoch(model, train_fac, style_iters, opt, sched,
                        device, args, epoch)
        print(f"[EPOCH {epoch+1}] cap {avg['cap']:.4f}  "
              f"v2l {avg['v2l']:.4f}  txt {avg['txt']:.4f}")

        if last_feats is None:
            last_feats = next(iter(val_fac))[0][:3].to(device)
        print(f"[EPOCH {epoch+1}] samples:")
        show_samples(model, last_feats, all_styles,
                     [float(x) for x in args.lams.split(',')])

        print(f"[EPOCH {epoch+1}] validation:")
        val = validate(model, val_fac, val_style_loaders, device)

        state = {'epoch': epoch, 'model': model.state_dict(),
                 'opt': opt.state_dict(), 'sched': sched.state_dict(),
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
    p.add_argument('--styles', default='romantic,humorous')
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--style_batch_size', type=int, default=12)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--lr_proj', type=float, default=1e-3)
    p.add_argument('--w_v2l', type=float, default=1.0)
    p.add_argument('--w_txt', type=float, default=1.0)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--grad_ckpt', type=int, default=1)
    p.add_argument('--adam8bit', type=int, default=0)
    p.add_argument('--log_step', type=int, default=500)
    p.add_argument('--lams', default='1.0,3.0')
    args = p.parse_args()
    main(args)
