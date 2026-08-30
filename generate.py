"""
generate.py — guided decoding with a TEXT-ONLY style discriminator.

    P(y_t | s, x, y_<t)  ∝  P(y_t | x, y_<t) * P(s | y_<t, y_t)^w
                            ^factual (image)   ^style discriminator (text only)

The discriminator no longer takes any image. It is run twice on the running
token sequence, once with the desired code and once with <factual>, and the two
are contrasted (PPCap Eq. 5). The first generated token is factual-only so the
caption anchors on the image before style is applied.
"""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import json, argparse, torch, torch.nn.functional as F
from transformers import VisionEncoderDecoderModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from data import split_line, add_style_tokens
from models import StyleModel

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
def guided_beam(fac, sty, feats, sid_d, sid_u, tok, w,
                num_beams=5, max_new=40, rep=1.15, nrng=3):
    dev = feats.device
    B, K = feats.size(0), num_beams
    V = fac.decoder.get_output_embeddings().weight.size(0)
    bos = fac.config.decoder_start_token_id
    eos = fac.config.eos_token_id
    pad = fac.config.pad_token_id

    enc = BaseModelOutput(last_hidden_state=feats.repeat_interleave(K, 0))
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
        styling = (w != 0.0) and (step > 0)   # first token: factual-only
        if styling:
            ld = clamped_logprobs(sty.step_logits(d_ids, ys)[:, :V])
            lu = clamped_logprobs(sty.step_logits(u_ids, ys)[:, :V])
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
        if styling:
            sel = ft.unsqueeze(1)
            ll_d = ll_d[flat] + ld[flat].gather(1, sel).squeeze(1)
            ll_u = ll_u[flat] + lu[flat].gather(1, sel).squeeze(1)
        else:
            ll_d = ll_d[flat]
            ll_u = ll_u[flat]
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
    sty = StyleModel(tok).to(dev)
    sty.load_state_dict(ck['model']); sty.eval()
    style = ck['style']
    print(f"[ckpt] factual {a.factual_ckpt}   style '{style}'")
    print(f"desired <{style}>={sid[style]}   undesired <factual>={sid['factual']}")

    ws = [float(x) for x in a.w.split(',')]
    ids_all, refs = test_images(a.test_file, a.n_images)
    keep = [i for i in ids_all
            if os.path.exists(os.path.join(a.cache_dir, f"{i}.pt"))]
    print(f"\ndecoding {len(keep)} images, w = {ws}\n")

    rec = {i: {"image_id": i, "references": refs[i]} for i in keep}
    for s in range(0, len(keep), a.batch_size):
        ch = keep[s:s + a.batch_size]
        feats = torch.stack([torch.load(os.path.join(a.cache_dir, f"{i}.pt"),
                                        map_location='cpu').float() for i in ch]).to(dev)
        for w in ws:
            out = guided_beam(fac, sty, feats, sid[style], sid['factual'],
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
    p.add_argument('--test_file',    default='/kaggle/working/splits/factual_test.txt')
    p.add_argument('--out_json',     default='/kaggle/working/ppcap_romantic.json')
    p.add_argument('--n_images', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--beams', type=int, default=5)
    p.add_argument('--max_new', type=int, default=40)
    p.add_argument('--w', default='0,2,4,6,8,10')
    main(p.parse_args())
