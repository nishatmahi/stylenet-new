import os
import random
import argparse
import torch

from data_loader import get_data_loader, get_styled_data_loader, tokenizer
from models import BanglaT5StyleCaptioner


def split_caption_file(input_file, train_file, val_file, train_ratio=0.8, seed=42):
    if not os.path.exists(input_file):
        print(f"[WARN] {input_file} not found, skipping")
        return False

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.seed(seed)
    random.shuffle(lines)

    split_idx = int(len(lines) * train_ratio)
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(val_file), exist_ok=True)
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[:split_idx])
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[split_idx:])
    print(f"Split {os.path.basename(input_file)}: "
          f"{split_idx} train / {len(lines)-split_idx} val")
    return True


def is_style_param(name, style=None):
    if ".S." not in name:
        return False
    return True if style is None else f".S.{style}." in name


def set_stage(model, stage, style=None):
    """Task 1 (captioning) updates A, embeddings, {U},{V},{W} and S_factual.
    Task 2 (style LM) updates ONLY S_style. Paper sec 3.3."""
    if stage == "factual":
        for n, p in model.named_parameters():
            if n.startswith("t5.encoder."):
                p.requires_grad = False
            elif is_style_param(n):
                p.requires_grad = is_style_param(n, "factual")
            else:
                p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        for n, p in model.named_parameters():
            if is_style_param(n, style):
                p.requires_grad = True


def build_optimizers(model, args, styles):
    adapter_keys = ("DenseReluDense.V", "DenseReluDense.U",
                    "DenseReluDense.S", "projector.")
    slow, fast = [], []
    for n, p in model.named_parameters():
        if n.startswith("t5.encoder."):
            continue
        if is_style_param(n) and not is_style_param(n, "factual"):
            continue
        (fast if any(k in n for k in adapter_keys) else slow).append(p)

    Opt = torch.optim.AdamW
    if args.adam8bit:
        import bitsandbytes as bnb
        Opt = bnb.optim.AdamW8bit

    optimizer_fac = Opt([
        {'params': slow, 'lr': args.lr_caption},
        {'params': fast, 'lr': args.lr_adapter},
    ], weight_decay=0.01)

    optimizer_style = {}
    for s in styles:
        ps = [p for n, p in model.named_parameters() if is_style_param(n, s)]
        optimizer_style[s] = Opt(ps, lr=args.lr_style, weight_decay=0.0)
        print(f"  S_{s}: {sum(p.numel() for p in ps)/1e6:.2f}M trainable")

    print(f"  captioning stage: {sum(p.numel() for p in slow+fast)/1e6:.1f}M trainable")
    return optimizer_fac, optimizer_style


def optim_step(loss, optimizer, model):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()


def run_factual_epoch(model, loader, optimizer, device, args, epoch):
    total, ntok_all, last_feats = 0.0, 0, None

    for i, (raw_feats, labels) in enumerate(loader):
        raw_feats = raw_feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        last_feats = raw_feats

        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(labels, raw_features=raw_feats, mode="factual")
        optim_step(loss, optimizer, model)

        ntok = (labels != -100).sum().item()
        total += loss.item() * ntok
        ntok_all += ntok

        if i % args.log_step == 0 or i == len(loader) - 1:
            print(f"Epoch [{epoch+1}/{args.epoch_num}], CAP, "
                  f"Step [{i}/{len(loader)}], Loss: {loss.item():.4f}", flush=True)

    return (total / ntok_all if ntok_all else 0.0), last_feats


def run_style_epoch(model, loader, optimizer, device, args, epoch, style):
    total, ntok_all = 0.0, 0

    for i, labels in enumerate(loader):
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(labels, raw_features=None, mode=style)   # noise memory
        optim_step(loss, optimizer, model)

        ntok = (labels != -100).sum().item()
        total += loss.item() * ntok
        ntok_all += ntok

        if i % args.log_step == 0 or i == len(loader) - 1:
            print(f"Epoch [{epoch+1}/{args.epoch_num}], {style.upper()[:3]}, "
                  f"Step [{i}/{len(loader)}], Loss: {loss.item():.4f}", flush=True)

    return total / ntok_all if ntok_all else 0.0


@torch.no_grad()
def validate(model, val_loader, val_styled, device):
    model.eval()

    fac, ntok = 0.0, 0
    for raw_feats, labels in val_loader:
        raw_feats = raw_feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        loss, _ = model(labels, raw_features=raw_feats, mode="factual")
        n = (labels != -100).sum().item()
        fac += loss.item() * n
        ntok += n
    fac = fac / ntok if ntok else float('inf')
    print(f"Validation Factual Loss: {fac:.4f}")

    for style, loader in val_styled.items():
        s, sn = 0.0, 0
        for labels in loader:
            labels = labels.to(device, non_blocking=True)
            loss, _ = model(labels, raw_features=None, mode=style)
            n = (labels != -100).sum().item()
            s += loss.item() * n
            sn += n
        if sn:
            print(f"Validation {style.capitalize()} Loss (monitor only): {s/sn:.4f}")

    model.train()
    return fac


@torch.no_grad()
def show_samples(model, raw_feats, styles, n=3):
    """Real beam search. Teacher-forced argmax over pad positions is what
    produced the 'Composed Chakraborty' tails before."""
    model.eval()
    feats = raw_feats[:n]
    for mode in styles:
        ids = model.generate_caption(raw_features=feats, mode=mode, num_beams=5)
        for i, txt in enumerate(tokenizer.batch_decode(ids, skip_special_tokens=True)):
            print(f"  [{mode}] {i+1}: {txt}")
    model.train()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    save_folder = "stylenet_t5_models/"
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    tdir, vdir = '/kaggle/working/train_split', '/kaggle/working/val_split'
    fac_train = os.path.join(tdir, 'factual_train.txt')
    fac_val = os.path.join(vdir, 'factual_val.txt')
    split_caption_file(args.factual_caption_path, fac_train, fac_val,
                       args.train_split_ratio)

    wanted = [s.strip() for s in args.styles.split(',') if s.strip()]
    style_paths, active = {}, []
    for style, src in [("romantic", args.romantic_caption_path),
                       ("humorous", args.humorous_caption_path)]:
        if style not in wanted:
            continue
        st = os.path.join(tdir, f'{style}_train.txt')
        sv = os.path.join(vdir, f'{style}_val.txt')
        if split_caption_file(src, st, sv, args.train_split_ratio):
            style_paths[style] = (st, sv)
            active.append(style)

    all_modes = ["factual"] + active
    print(f"Active modes: {all_modes}")

    train_loader = get_data_loader(args.vit_cache_dir, fac_train,
                                   args.caption_batch_size, shuffle=True)
    val_loader = get_data_loader(args.vit_cache_dir, fac_val,
                                 args.caption_batch_size, shuffle=False)

    train_styled, val_styled = {}, {}
    for s in active:
        train_styled[s] = get_styled_data_loader(style_paths[s][0],
                                                 args.style_batch_size, shuffle=True)
        val_styled[s] = get_styled_data_loader(style_paths[s][1],
                                               args.style_batch_size, shuffle=False)

    print(f"Steps/epoch: factual={len(train_loader)}, " +
          ", ".join(f"{s}={len(l)}" for s, l in train_styled.items()))

    model = BanglaT5StyleCaptioner(
        t5_ckpt=args.t5_ckpt,
        vit_hidden=args.vit_hidden,
        factored_dim=args.factored_dim,
        styles=tuple(all_modes),
        memory_len=args.memory_len,
        gradient_checkpointing=bool(args.gradient_checkpointing),
    ).to(device)

    optimizer_fac, optimizer_style = build_optimizers(model, args, active)

    start_epoch, best_val, patience = 0, float('inf'), 0
    ckpt_path = os.path.join(save_folder, 'checkpoint-latest.pth')
    best_path = os.path.join(save_folder, 'best_model.pth')

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model_state_dict'], strict=False)
        optimizer_fac.load_state_dict(ck['optimizer_fac_state_dict'])
        for s, st in ck.get('optimizer_style_state_dict', {}).items():
            if s in optimizer_style:
                optimizer_style[s].load_state_dict(st)
        start_epoch = ck['epoch'] + 1
        best_val = ck.get('best_val_loss', float('inf'))
        patience = ck.get('patience_counter', 0)
        print(f"[DEBUG] Resumed from epoch {ck['epoch']+1}, best val {best_val:.4f}")
    else:
        print("[DEBUG] No checkpoint found. Training from scratch.")

    for epoch in range(start_epoch, args.epoch_num):
        print(f"\n[DEBUG] Training epoch {epoch+1} of {args.epoch_num}")
        model.train()

        # Task 1 then Task 2, alternating one epoch each, starting with
        # captioning — paper sec 5.1.1.
        set_stage(model, "factual")
        avg_fac, last_feats = run_factual_epoch(
            model, train_loader, optimizer_fac, device, args, epoch)

        avg_style = {}
        for s in active:
            set_stage(model, "style", style=s)
            avg_style[s] = run_style_epoch(
                model, train_styled[s], optimizer_style[s], device, args, epoch, s)

        print(f"\n[EPOCH {epoch+1}] Factual Training Loss:  {avg_fac:.4f}")
        for s, v in avg_style.items():
            print(f"[EPOCH {epoch+1}] {s.capitalize()} Training Loss: {v:.4f}")

        if last_feats is not None:
            print(f"[EPOCH {epoch+1}] Samples:")
            show_samples(model, last_feats, all_modes)

        print(f"[EPOCH {epoch+1}] Running validation...")
        val_loss = validate(model, val_loader, val_styled, device)
        print(f"[EPOCH {epoch+1}] Factual Validation Loss (early stopping): {val_loss:.4f}")

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_fac_state_dict': optimizer_fac.state_dict(),
            'optimizer_style_state_dict': {s: o.state_dict() for s, o in optimizer_style.items()},
            'best_val_loss': best_val,
            'factual_train_loss': avg_fac,
            'style_train_loss': avg_style,
            'val_loss': val_loss,
            'patience_counter': patience,
            'styles': all_modes,
            'factored_dim': args.factored_dim,
            'memory_len': args.memory_len,
        }

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            state['best_val_loss'] = best_val
            state['patience_counter'] = patience
            torch.save(state, best_path)
            print(f"[EPOCH {epoch+1}] New best model saved! Factual val loss: {val_loss:.4f}")
        else:
            patience += 1
            state['patience_counter'] = patience
            print(f"[EPOCH {epoch+1}] No improvement. Patience: {patience}/{args.patience}")

        torch.save(state, ckpt_path)
        torch.save(state, os.path.join(args.model_path, 'checkpoint-latest.pth'))
        print(f"[EPOCH {epoch+1}] Checkpoint saved.")

        if patience >= args.patience:
            print(f"[EARLY STOPPING] Best factual validation loss: {best_val:.4f}")
            break

    if os.path.exists(best_path):
        print(f"\nTraining completed. Loading best model ({best_val:.4f})...")
        model.load_state_dict(
            torch.load(best_path, map_location=device)['model_state_dict'], strict=False)
        print("Best model loaded successfully!")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='StyleNet Bangla — BanglaT5 decoder, factored style FFNs, monolingual style corpus')
    p.add_argument('--model_path', type=str, default='pretrained_models')
    p.add_argument('--vit_cache_dir', type=str, default='/kaggle/working/vit_feature_cache')
    p.add_argument('--factual_caption_path', type=str,
                   default='/kaggle/input/datasets/kaggleperfect/dataset/data/factual_caption.txt')
    p.add_argument('--romantic_caption_path', type=str,
                   default='/kaggle/input/datasets/kaggleperfect/dataset/data/romantic_data.txt')
    p.add_argument('--humorous_caption_path', type=str,
                   default='/kaggle/input/datasets/kaggleperfect/dataset/data/humorous_generated.txt')
    p.add_argument('--styles', type=str, default='romantic')
    p.add_argument('--t5_ckpt', type=str, default='csebuetnlp/banglat5')
    p.add_argument('--vit_hidden', type=int, default=768)
    p.add_argument('--memory_len', type=int, default=197)
    p.add_argument('--caption_batch_size', type=int, default=16)
    p.add_argument('--style_batch_size', type=int, default=24)
    p.add_argument('--factored_dim', type=int, default=512)
    p.add_argument('--lr_caption', type=float, default=3e-5)
    p.add_argument('--lr_adapter', type=float, default=1e-4)
    p.add_argument('--lr_style', type=float, default=1e-4)
    p.add_argument('--epoch_num', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--train_split_ratio', type=float, default=0.8)
    p.add_argument('--gradient_checkpointing', type=int, default=0)
    p.add_argument('--adam8bit', type=int, default=0)
    p.add_argument('--log_step', type=int, default=500)
    args = p.parse_args()
    main(args)
