"""
generate.py — the plug-and-play part. Paper Eq. 1, 2 and 5.

The factual model proposes; the generative style discriminator re-ranks:

    P(y_t | s, x, y_<t)  ∝  P(y_t | x, y_<t) · P(s | x, y_<t, y_t)^w        (2)

and the discriminator's posterior comes from running the ONE stylized model
under both control codes and normalising between them:

    P(s | x, y_<t, y_t) = P(s)·P(y_<t, y_t | s, x)
                          ────────────────────────────────                  (5)
                          Σ_{s'∈{s,s̄}} P(s')·P(y_<t, y_t | s', x)

Neither model is fine-tuned here. Nothing is merged. w=0 gives you the plain
factual caption; raising w bends the output toward the style. On FlickrStyle10k
the paper uses w=30 for romantic and w=39 for humorous — Bangla will need its
own sweep, which is what --w accepts a list for.

    python generate.py --style romantic --w 0,10,20,30,40 \
        --factual_ckpt /kaggle/working/factual/best_model.pth \
        --style_ckpt   /kaggle/working/style_romantic/best_model.pth \
        --cache_dir    /kaggle/working/clip_feature_cache \
        --style_img_pt /kaggle/working/style_feats/test_images.pt \
        --test_file    /kaggle/working/splits/factual_test.txt
"""

import os
import json
import argparse
import contextlib

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

from data import build_tokenizer, Encoder, split_line, normalize
from models import FactualCaptioner, StyleModel, load_compatible

# The discriminator's probabilities are clamped before being logged. A style
# model trained on ~35k sentences is confidently wrong about rare tokens, and
# without a floor one such token drives the guided score to -inf and the beam
# dies. Same values PPCap uses.
P_FLOOR, P_CEIL = 1e-3, 0.8


def clamped_logprobs(logits):
    p = torch.softmax(logits.float(), -1).clamp(min=P_FLOOR, max=P_CEIL)
    return torch.log(p)


@torch.no_grad()
def guided_beam_search(factual, style, feats, clip_emb, sid_desired,
                       sid_undesired, tok, w, num_beams=5, max_new_tokens=48,
                       length_penalty=1.0, amp=contextlib.nullcontext):
    """Beam search over the guided distribution of Eq. 2.

    No KV cache: the decoder prefix is re-run each step. That is slower than
    the cached path but it is identical across Transformers versions, and the
    sequences here are at most 48 tokens.
    """
    device = feats.device
    B = feats.size(0)
    K = num_beams
    V = factual.t5.config.vocab_size
    pad = tok.pad_token_id
    eos = tok.eos_token_id

    with amp():
        f_h, f_mask = factual.encode_image(feats)
        s_h, s_mask = style.prefix_states(
            clip_emb, torch.full((B,), sid_desired, dtype=torch.long, device=device))
        u_h, u_mask = style.prefix_states(
            clip_emb, torch.full((B,), sid_undesired, dtype=torch.long, device=device))

    def expand(h, m):
        return (h.repeat_interleave(K, 0), m.repeat_interleave(K, 0))

    f_h, f_mask = expand(f_h, f_mask)
    s_h, s_mask = expand(s_h, s_mask)
    u_h, u_mask = expand(u_h, u_mask)

    # decoder starts from T5's decoder_start_token_id (the pad id)
    ys = torch.full((B * K, 1), factual.t5.config.decoder_start_token_id,
                    dtype=torch.long, device=device)
    scores = torch.full((B, K), float('-inf'), device=device)
    scores[:, 0] = 0.0
    scores = scores.view(-1)                       # [B*K]
    ll_s = torch.zeros(B * K, device=device)       # running logP under s
    ll_u = torch.zeros(B * K, device=device)       # running logP under s-bar
    done = torch.zeros(B * K, dtype=torch.bool, device=device)

    def run(model, h, m, ys):
        out = model.t5(encoder_outputs=BaseModelOutput(last_hidden_state=h),
                       attention_mask=m, decoder_input_ids=ys)
        return out.logits[:, -1, :]

    for step in range(max_new_tokens):
        t = step + 1
        with amp():
            logp_f = F.log_softmax(run(factual, f_h, f_mask, ys).float(), -1)
            logp_s = clamped_logprobs(run(style, s_h, s_mask, ys))
            logp_u = clamped_logprobs(run(style, u_h, u_mask, ys))

        if w != 0.0:
            # Eq. 5, length-normalised exactly as the reference code does
            a_s = (ll_s.unsqueeze(1) + logp_s) / t
            a_u = (ll_u.unsqueeze(1) + logp_u) / t
            logp_desired = F.log_softmax(torch.stack([a_s, a_u], -1), -1)[..., 0]
            guided = F.log_softmax(logp_f + w * logp_desired, -1)
        else:
            guided = logp_f                        # w=0 -> the factual model alone

        # a finished beam may only extend with pad, at no cost
        guided[done] = float('-inf')
        guided[done, pad] = 0.0

        cand = scores.unsqueeze(1) + guided                    # [B*K, V]
        cand = cand.view(B, K * V)
        top_scores, top_idx = cand.topk(K, dim=-1)             # [B,K]
        beam_idx = top_idx // V                                # which beam
        token_idx = top_idx % V                                # which token

        flat_beam = (torch.arange(B, device=device).unsqueeze(1) * K
                     + beam_idx).view(-1)                      # [B*K]
        flat_token = token_idx.view(-1)

        ys = torch.cat([ys[flat_beam], flat_token.unsqueeze(1)], dim=1)
        scores = top_scores.view(-1)
        done = done[flat_beam] | (flat_token == eos)

        sel = flat_token.unsqueeze(1)
        ll_s = ll_s[flat_beam] + logp_s[flat_beam].gather(1, sel).squeeze(1)
        ll_u = ll_u[flat_beam] + logp_u[flat_beam].gather(1, sel).squeeze(1)

        if done.all():
            break

    # length-normalise and take the best beam per image
    lengths = (ys != pad).sum(1).clamp(min=1).float()
    final = (scores / lengths.pow(length_penalty)).view(B, K)
    best = final.argmax(-1)
    picked = ys.view(B, K, -1)[torch.arange(B, device=device), best]
    return tok.batch_decode(picked, skip_special_tokens=True)


def test_images(path, limit):
    """Image ids with their factual references, normalised through the same
    pipeline the model trained under."""
    order, refs = [], {}
    for line in open(path, encoding='utf-8'):
        parsed = split_line(line.strip())
        if parsed is None or not parsed[1]:
            continue
        img, cap = parsed
        if img not in refs:
            if limit and len(order) >= limit:
                continue
            order.append(img)
            refs[img] = []
        refs[img].append(normalize(cap))
    return order, refs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--style', required=True, choices=['romantic', 'humorous'])
    p.add_argument('--factual_ckpt', required=True)
    p.add_argument('--style_ckpt', required=True)
    p.add_argument('--cache_dir', required=True,
                   help='CLIP ViT-B/32 patch-token cache, for the factual model')
    p.add_argument('--style_img_pt', required=True,
                   help='NLLB-CLIP image embeddings, for the style model')
    p.add_argument('--test_file', required=True)
    p.add_argument('--out_json', default='/kaggle/working/ppcap_generations.json')
    p.add_argument('--w', default='0,10,20,30,40')
    p.add_argument('--n_images', type=int, default=0, help='0 = all')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--beam_size', type=int, default=5)
    p.add_argument('--max_new', type=int, default=48)
    p.add_argument('--amp', type=int, default=1)
    p.add_argument('--force_bf16', type=int, default=1)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ws = [float(x) for x in args.w.split(',')]

    cuda = device.type == 'cuda'
    major = torch.cuda.get_device_capability()[0] if cuda else 0
    if args.amp and cuda and (major >= 8 or args.force_bf16):
        amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    elif args.amp and cuda:
        amp = lambda: torch.autocast('cuda', dtype=torch.float16)
    else:
        amp = contextlib.nullcontext

    tokenizer, style_ids = build_tokenizer()

    fck = torch.load(args.factual_ckpt, map_location=device)
    factual = FactualCaptioner(fck.get('t5_ckpt', 'csebuetnlp/banglat5'),
                               fck.get('clip_dim', 768)).to(device)
    factual.t5.resize_token_embeddings(len(tokenizer))
    load_compatible(factual, fck['model'])
    factual.eval()

    sck = torch.load(args.style_ckpt, map_location=device)
    style = StyleModel(sck.get('t5_ckpt', 'csebuetnlp/banglat5'),
                       sck['clip_dim'], sck.get('prefix_len', 10),
                       vocab_size=len(tokenizer)).to(device)
    load_compatible(style, sck['model'])
    style.eval()
    print(f"[ckpt] factual epoch {fck.get('epoch',-1)+1}, "
          f"style '{sck.get('style')}' epoch {sck.get('epoch',-1)+1}")

    sid_desired = style_ids[args.style]
    sid_undesired = style_ids['factual']       # paper: factual is the undesired
    print(f"desired <{args.style}>={sid_desired}   undesired <factual>={sid_undesired}")

    ids_all, refs = test_images(args.test_file, args.n_images)
    simg = torch.load(args.style_img_pt, map_location='cpu')
    lut = {k: i for i, k in enumerate(simg['ids'])}

    keep = [i for i in ids_all
            if i in lut and os.path.exists(os.path.join(args.cache_dir, f"{i}.pt"))]
    dropped = len(ids_all) - len(keep)
    if dropped:
        print(f"[WARN] {dropped} test images lack a patch cache or an "
              f"NLLB-CLIP embedding and were skipped")
    print(f"\ndecoding {len(keep)} images, w = {ws}\n")

    records = {i: {"image_id": i, "references": refs[i]} for i in keep}

    for s in range(0, len(keep), args.batch_size):
        chunk = keep[s:s + args.batch_size]
        feats = torch.stack([
            torch.load(os.path.join(args.cache_dir, f"{i}.pt"),
                       map_location='cpu') for i in chunk]).to(device)
        clip_emb = torch.stack([simg['emb'][lut[i]] for i in chunk]).to(device)
        clip_emb = clip_emb / clip_emb.norm(2, dim=-1, keepdim=True).clamp(min=1e-6)

        for w in ws:
            txts = guided_beam_search(
                factual, style, feats, clip_emb, sid_desired, sid_undesired,
                tokenizer, w, num_beams=args.beam_size,
                max_new_tokens=args.max_new, amp=amp)
            key = "factual" if w == 0.0 else f"{args.style}_w{w:g}"
            for img, t in zip(chunk, txts):
                records[img][key] = t

        print(f"  {min(s+args.batch_size, len(keep))}/{len(keep)}", flush=True)

    out = [records[i] for i in keep]
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    for rec in out[:5]:
        print(f"\n{rec['image_id']}.jpg")
        print(f"  [ref]      {rec['references'][0]}")
        for w in ws:
            key = "factual" if w == 0.0 else f"{args.style}_w{w:g}"
            print(f"  [w={w:<4g}]  {rec[key]}")

    print(f"\nwrote {args.out_json} ({len(out)} images)")
    print("Pick w from the sweep the way the paper does (Fig. 5): raise it "
          "until style accuracy passes ~90%, then stop — CIDEr falls "
          "monotonically as w rises, so the smallest w that reaches the style "
          "is the right one.")


if __name__ == '__main__':
    main()
