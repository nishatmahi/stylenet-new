"""
generate.py — step 3. The two models meet, only here.

    P(y_t | s, x, y_<t)  is proportional to  P(y_t | x, y_<t) * P(s | x, y_<t, y_t)^w
                                             ^factual model    ^style discriminator

The discriminator posterior is Eq. 5: the style model is run twice on the same
image embedding, once with the desired control code and once with the
undesired one, and the two are contrasted.

Numerical details taken from the reference implementation, not the paper:
  * the discriminator's probabilities are clamped before log
  * the posterior is length-normalised by seq_len
  * repetition_penalty / no_repeat_ngram, so w=0 matches the factual model
"""
import os, json, argparse, torch, torch.nn.functional as F
from transformers import VisionEncoderDecoderModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from data import split_line, add_style_tokens
from models import StyleModel, noise_injection

P_FLOOR, P_CEIL = 1e-3, 0.8


def clamped_logprobs(logits):
    return torch.log(torch.softmax(logits.float(), -1).clamp(P_FLOOR, P_CEIL))


def block_repeats(logits, ys, n, penalty):
    if penalty and penalty != 1.0:
        prev = torch.gather(logits, 1, ys)
        logits = logits.scatter(1, ys, torch.where(prev < 0, prev * penalty,
                                                   prev / penalty))
    if n and ys.size(1) >= n:
        for b in range(ys.size(0)):
            seq = ys[b].tolist()
            tail = tuple(seq[-(n - 1):])
            for i in range(len(seq) - n + 1):
                if tuple(seq[i:i + n - 1]) == tail:
                    logits[b, seq[i + n - 1]] = float('-inf')
    return logits


@torch.no_grad()
def guided_beam(fac, sty, feats, clip_emb, sid_d, sid_u, tok, w,
                num_beams=5, max_new=40, rep=1.15, nrng=3):
    dev = feats.device
    B, K = feats.size(0), num_beams
    # the factual model's real output width. The style model's is 3 larger
    # (the control tokens), so its logits are sliced to V before mixing.
    V = fac.decoder.get_output_embeddings().weight.size(0)
    bos = fac.config.decoder_start_token_id
    eos = fac.config.eos_token_id
    pad = fac.config.pad_token_id

    enc = BaseModelOutput(last_hidden_state=feats.repeat_interleave(K, 0))
    ce = clip_emb.repeat_interleave(K, 0)
    d_ids = torch.full((B * K,), sid_d, dtype=torch.long, device=dev)
    u_ids = torch.full((B * K,), sid_u, dtype=torch.long, device=dev)

    ys = torch.full((B * K, 1), bos, dtype=torch.long, device=dev)
    scores = torch.full((B, K), float('-inf'), device=dev)
    scores[:, 0] = 0.0
    scores = scores.view(-1)
    ll_d = torch.zeros(B * K, device=dev)
    ll_u = torch.zeros(B * K, device=dev)
    done = torch.zeros(B * K, dtype=torch.bool, device=dev)

    for step in range(max_new):
        t = step + 1
        lf = F.log_softmax(fac(encoder_outputs=enc, decoder_input_ids=ys
                               ).logits[:, -1, :].float(), -1)
        if w != 0.0:
            ld = clamped_logprobs(sty.step_logits(ce, d_ids, ys)[:, :V])
            lu = clamped_logprobs(sty.step_logits(ce, u_ids, ys)[:, :V])
            a_d = (ll_d.unsqueeze(1) + ld) / t
            a_u = (ll_u.unsqueeze(1) + lu) / t
            post = F.log_softmax(torch.stack([a_d, a_u], -1), -1)[..., 0]
            g = F.log_softmax(lf + w * post, -1)
        else:
            g = lf
        g = block_repeats(g, ys[:, 1:], nrng, rep) if ys.size(1) > 1 else g

        g[done] = float('-inf')
        g[done, pad] = 0.0
        cand = (scores.unsqueeze(1) + g).view(B, K * V)
        top, idx = cand.topk(K, -1)
        beam, tokn = idx // V, idx % V
        flat = (torch.arange(B, device=dev).unsqueeze(1) * K + beam).view(-1)
        ft = tokn.view(-1)
        ys = torch.cat([ys[flat], ft.unsqueeze(1)], 1)
        scores = top.view(-1)
        done = done[flat] | (ft == eos)
        if w != 0.0:
            sel = ft.unsqueeze(1)
            ll_d = ll_d[flat] + ld[flat].gather(1, sel).squeeze(1)
            ll_u = ll_u[flat] + lu[flat].gather(1, sel).squeeze(1)
        if done.all():
            break

    lens = (ys != pad).sum(1).clamp(min=1).float()
    best = (scores / lens).view(B, K).argmax(-1)
    return tok.batch_decode(ys.view(B, K, -1)[torch.arange(B, device=dev), best],
                            skip_special_tokens=True)


def test_images(path, limit):
    order, refs = [], {}
    for ln in open(path, encoding='utf-8'):
        p = split_line(ln.strip())
        if not p or not p[1]:
            continue
        img, cap = p
        if img not in refs:
            if len(order) >= limit:
                continue
            order.append(img); refs[img] = []
        refs[img].append(cap)
    return order, refs


def main(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(a.factual_ckpt)
    sid = add_style_tokens(tok)

    fac = VisionEncoderDecoderModel.from_pretrained(a.factual_ckpt).to(dev).eval()
    ck = torch.load(a.style_ckpt, map_location=dev)
    sty = StyleModel(ck['clip_dim'], tok, prefix_len=ck['prefix_len']).to(dev)
    sty.load_state_dict(ck['model']); sty.eval()
    style = ck['style']
    print(f"[ckpt] factual {a.factual_ckpt}   style '{style}'")
    print(f"desired <{style}>={sid[style]}   undesired <factual>={sid['factual']}")

    ws = [float(x) for x in a.w.split(',')]
    ids_all, refs = test_images(a.test_file, a.n_images)
    keep = [i for i in ids_all
            if os.path.exists(os.path.join(a.cache_dir, f"{i}.pt"))]
    img_emb = torch.load(a.style_img_pt, map_location='cpu')
    lut = {k: v for k, v in zip(img_emb['ids'], img_emb['emb'].float())}
    keep = [i for i in keep if i in lut]
    print(f"\ndecoding {len(keep)} images, w = {ws}\n")

    rec = {i: {"image_id": i, "references": refs[i]} for i in keep}
    for s in range(0, len(keep), a.batch_size):
        ch = keep[s:s + a.batch_size]
        feats = torch.stack([torch.load(os.path.join(a.cache_dir, f"{i}.pt"),
                                        map_location='cpu').float() for i in ch]).to(dev)
        ce = noise_injection(torch.stack([lut[i] for i in ch]).to(dev), 0.0, True)
        for w in ws:
            out = guided_beam(fac, sty, feats, ce, sid[style], sid['factual'],
                              tok, w, a.beams, a.max_new)
            for i, t in zip(ch, out):
                rec[i][f"w{w}"] = t
        print(f"  {min(s+a.batch_size, len(keep))}/{len(keep)}", flush=True)

    out = [rec[i] for i in keep]
    json.dump(out, open(a.out_json, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    for r in out[:5]:
        print(f"\n{r['image_id']}.jpg\n  [ref]  {r['references'][0]}")
        for w in ws:
            print(f"  [w={w:<5}] {r[f'w{w}']}")
    print(f"\nwrote {a.out_json} ({len(out)} images)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--factual_ckpt', default='/kaggle/working/p2_factual/best')
    p.add_argument('--style_ckpt',   default='/kaggle/working/p2_style_romantic/best_model.pth')
    p.add_argument('--cache_dir',    default='/kaggle/working/clip_feature_cache')
    p.add_argument('--style_img_pt', default='/kaggle/working/style_feats/test_images.pt')
    p.add_argument('--test_file',    default='/kaggle/working/splits/factual_test.txt')
    p.add_argument('--out_json',     default='/kaggle/working/p2_generations.json')
    p.add_argument('--n_images', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--beams', type=int, default=5)
    p.add_argument('--max_new', type=int, default=40)
    p.add_argument('--w', default='0,1,2,3,5,8')
    main(p.parse_args())
