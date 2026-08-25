"""
train_style.py — the stylized model. Paper Sec. 3.3, Eq. 6-8.

Trained on the unpaired style corpus and the factual text corpus. No images.
No pairing between a factual sentence and a styled one. Nothing corrupted,
masked, or deleted. The CLIP TEXT embedding stands in for the image, and the
style control token says which class the sentence belongs to.

    L_g = mean  -(1/T) log P(y | s_true, x)          make it likely as itself
    L_d = mean  -log P(s_true | x, y)                and unlikely as the other
    L   = lam * L_g + (1 - lam) * L_d                lam = 0.8

L_d is what makes this a discriminator rather than a language model. Modelling
the style corpus alone gives the style token nothing to explain — the loss
falls just as fast whether or not the token is read. Forcing the same sentence
to be *less* likely under the contrasting token is what makes the model
represent the difference between the two registers.

    python train_style.py --style romantic \
        --style_pt   /kaggle/working/style_feats/romantic_train.pt \
        --factual_pt /kaggle/working/style_feats/factualtext_train.pt \
        --save_dir   /kaggle/working/style_romantic
"""

import os
import argparse
import contextlib

import torch
from torch.optim.lr_scheduler import LambdaLR

from data import build_tokenizer, Encoder, style_loader
from models import StyleModel, ppcap_loss, noise_injection, load_compatible
from train_factual import amp_context


def run_epoch(model, loader, opt, sched, device, args, epoch, amp, scaler):
    model.train()
    tot = tot_g = tot_d = 0.0
    n = 0
    params = [p for p in model.parameters() if p.requires_grad]

    for i, (emb, true_ids, other_ids, labels) in enumerate(loader):
        emb = noise_injection(emb.to(device), args.variance,
                              normalize=args.normalize_prefix)
        true_ids = true_ids.to(device)
        other_ids = other_ids.to(device)
        labels = labels.to(device)

        opt.zero_grad(set_to_none=True)
        with amp():
            loss, l_g, l_d = ppcap_loss(model, emb, true_ids, other_ids,
                                        labels, lam=args.lam)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        tot += loss.item(); tot_g += l_g.item(); tot_d += l_d.item()
        n += 1
        if i % args.log_step == 0 or i == len(loader) - 1:
            print(f"Epoch [{epoch+1}] [{i}/{len(loader)}] "
                  f"loss {loss.item():.3f}  gen {l_g.item():.3f}  "
                  f"disc {l_d.item():.3f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)

    return tot / max(n, 1), tot_g / max(n, 1), tot_d / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    tot = tot_g = tot_d = 0.0
    n = 0
    for emb, true_ids, other_ids, labels in loader:
        # no noise at validation — we want the clean number
        emb = noise_injection(emb.to(device), 0.0,
                              normalize=args.normalize_prefix)
        loss, l_g, l_d = ppcap_loss(model, emb, true_ids.to(device),
                                    other_ids.to(device), labels.to(device),
                                    lam=args.lam)
        tot += loss.item(); tot_g += l_g.item(); tot_d += l_d.item()
        n += 1
    model.train()
    return tot / max(n, 1), tot_g / max(n, 1), tot_d / max(n, 1)


@torch.no_grad()
def show(model, loader, tokenizer, style_ids, style, device, args, k=3):
    """Sanity check: same CLIP embedding, both control codes.

    The two outputs SHOULD diverge. If they read the same, the style token is
    being ignored and the discriminative loss is not doing its job — that is
    the failure to watch for, and it shows up here before it shows up in the
    guided decoding.
    """
    model.eval()
    emb, _, _, _ = next(iter(loader))
    emb = noise_injection(emb[:k].to(device), 0.0,
                          normalize=args.normalize_prefix)
    for name in (style, 'factual'):
        sid = torch.full((emb.size(0),), style_ids[name],
                         dtype=torch.long, device=device)
        h, mask = model.prefix_states(emb, sid)
        from transformers.modeling_outputs import BaseModelOutput
        ids = model.t5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=h),
            attention_mask=mask, num_beams=5, max_new_tokens=48,
            early_stopping=True, repetition_penalty=1.15,
            no_repeat_ngram_size=3)
        for j, t in enumerate(tokenizer.batch_decode(ids, skip_special_tokens=True)):
            print(f"    [{name:<8} {j+1}] {t}")
    model.train()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    amp, scaler = amp_context(args, device.type == 'cuda')

    tokenizer, style_ids = build_tokenizer(args.t5_ckpt)
    enc = Encoder(tokenizer)
    print(f"style control tokens: {style_ids}")

    train_dl, clip_dim = style_loader(
        args.style_pt, args.factual_pt, args.style, style_ids, enc,
        args.batch_size, shuffle=True)
    val_dl = None
    if args.style_val_pt and args.factual_val_pt:
        val_dl, _ = style_loader(args.style_val_pt, args.factual_val_pt,
                                 args.style, style_ids, enc,
                                 args.batch_size, shuffle=False)
    print(f"CLIP embedding dim read from the feature file: {clip_dim}")

    model = StyleModel(args.t5_ckpt, clip_dim, args.prefix_len,
                       vocab_size=len(tokenizer)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))
    print(f"  style model {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

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

    for epoch in range(start, args.epochs):
        print(f"\n=== epoch {epoch+1}/{args.epochs} ===")
        loss, l_g, l_d = run_epoch(model, train_dl, opt, sched, device,
                                   args, epoch, amp, scaler)
        print(f"[EPOCH {epoch+1}] loss {loss:.4f}  gen {l_g:.4f}  disc {l_d:.4f}")
        print(f"[EPOCH {epoch+1}] same embedding, both control codes:")
        show(model, train_dl, tokenizer, style_ids, args.style, device, args)

        if val_dl is not None:
            v, vg, vd = validate(model, val_dl, device, args)
            print(f"[EPOCH {epoch+1}] val loss {v:.4f}  gen {vg:.4f}  disc {vd:.4f}")
            score = v
        else:
            score = loss

        state = {'epoch': epoch, 'model': model.state_dict(),
                 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                 'scaler': scaler.state_dict(), 'best': best,
                 'patience': patience, 'style': args.style,
                 'clip_dim': clip_dim, 'prefix_len': args.prefix_len,
                 't5_ckpt': args.t5_ckpt}
        if score < best:
            best, patience = score, 0
            state['best'], state['patience'] = best, patience
            torch.save(state, best_path)
            print(f"[EPOCH {epoch+1}] new best {score:.4f}")
        else:
            patience += 1
            state['patience'] = patience
            print(f"[EPOCH {epoch+1}] no improvement {patience}/{args.patience}")
        torch.save(state, ck_path)
        if patience >= args.patience:
            print(f"[early stop] best {best:.4f}")
            break

    print(f"\nfinished. best {best:.4f}")
    print("On its own this model hallucinates — the paper measures cls 100.0 "
          "with CIDEr 28.1 for the style model alone. That is expected. It is "
          "a discriminator, not a captioner. Run generate.py.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--style', required=True, choices=['romantic', 'humorous'])
    p.add_argument('--style_pt', required=True)
    p.add_argument('--factual_pt', required=True)
    p.add_argument('--style_val_pt', default='')
    p.add_argument('--factual_val_pt', default='')
    p.add_argument('--save_dir', required=True)
    p.add_argument('--t5_ckpt', default='csebuetnlp/banglat5')
    p.add_argument('--prefix_len', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)     # paper: 64
    p.add_argument('--epochs', type=int, default=20)         # paper: 20
    p.add_argument('--lam', type=float, default=0.8)         # paper: 0.8
    p.add_argument('--variance', type=float, default=0.016)  # paper: 0.016
    p.add_argument('--normalize_prefix', type=int, default=1)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--amp', type=int, default=1)
    p.add_argument('--force_bf16', type=int, default=1)
    p.add_argument('--log_step', type=int, default=100)
    main(p.parse_args())
