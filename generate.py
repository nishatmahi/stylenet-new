"""
generate.py — factual model's OWN generate() + a GeDi style LogitsProcessor.
Decoding, EOS, and min/max length are HuggingFace's; the text-only style
discriminator enters only as a logits processor that adds w * log P(style | tokens).
"""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import json, argparse, torch, torch.nn.functional as F
from transformers import (VisionEncoderDecoderModel, AutoTokenizer,
                          LogitsProcessor, LogitsProcessorList)
from transformers.modeling_outputs import BaseModelOutput
from data import split_line, add_style_tokens
from models import StyleModel

P_FLOOR, P_CEIL = 1e-3, 0.8


class GeDiProcessor(LogitsProcessor):
    """Adds w * log P(desired style | sequence so far, next token) to the factual
    log-probs. Stateless, recomputed from the whole decoder sequence each step.
    The first generated token is left to the factual model (skip_first)."""

    def __init__(self, sty, code_d, code_u, w, V, skip_first=True):
        self.sty, self.cd, self.cu, self.w, self.V = sty, code_d, code_u, w, V
        self.skip_first = skip_first

    @torch.no_grad()
    def __call__(self, input_ids, scores):
        if self.w == 0:
            return scores
        B, L = input_ids.shape
        if self.skip_first and L <= 1:
            return scores
        dev = input_ids.device

        def branch(code_id):
            code = torch.full((B, 1), code_id, dtype=torch.long, device=dev)
            seq = torch.cat([code, input_ids], dim=1)
            logits = self.sty.gpt(input_ids=seq).logits[:, :, :self.V].float()
            lp = torch.log(torch.softmax(logits, -1).clamp(P_FLOOR, P_CEIL))
            prefix = lp[:, :L, :].gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
            ll = prefix.sum(1)
            nxt = lp[:, L, :]
            return ll, nxt

        ll_d, ld = branch(self.cd)
        ll_u, lu = branch(self.cu)
        t = L + 1
        a_d = (ll_d.unsqueeze(1) + ld) / t
        a_u = (ll_u.unsqueeze(1) + lu) / t
        post = F.log_softmax(torch.stack([a_d, a_u], -1), -1)[..., 0]
        out = scores.clone()
        out[:, :self.V] = out[:, :self.V] + self.w * post
        return out


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
    V = fac.decoder.get_output_embeddings().weight.size(0)
    print(f"[ckpt] factual {a.factual_ckpt}   style '{style}'   V={V}")

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
        enc = BaseModelOutput(last_hidden_state=feats)
        for w in ws:
            proc = LogitsProcessorList([GeDiProcessor(sty, sid[style], sid['factual'], w, V)])
            gen = fac.generate(
                encoder_outputs=enc,
                num_beams=a.beams,
                min_new_tokens=a.min_new,
                max_new_tokens=a.max_new,
                no_repeat_ngram_size=3,
                early_stopping=True,
                length_penalty=1.0,
                logits_processor=proc,
            )
            for i, t in zip(ch, tok.batch_decode(gen, skip_special_tokens=True)):
                rec[i][f"w{w}"] = t
        print(f"  {min(s+a.batch_size, len(keep))}/{len(keep)}", flush=True)

    out = [rec[i] for i in keep]
    json.dump(out, open(a.out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    for r in out[:6]:
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
    p.add_argument('--out_json',     default='/kaggle/working/test_final.json')
    p.add_argument('--n_images', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--beams', type=int, default=4)
    p.add_argument('--min_new', type=int, default=8)
    p.add_argument('--max_new', type=int, default=40)
    p.add_argument('--w', default='0,30,40')
    args, _ = p.parse_known_args()
    main(args)
