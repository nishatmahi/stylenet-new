# evaluate.py
# Reference-free, no-API evaluation for one style.
# Prints 3 numbers:
#   CLIPScore     -> does the caption match the image   (higher = better)
#   Style acc     -> does it sound like the target style (higher = better, it's a %)
#   Perplexity    -> is it fluent Bangla                 (LOWER = better)
#
# Run:
#   python evaluate.py --style romantic --pred /kaggle/working/gen_romantic.json
#   python evaluate.py --style humorous --pred /kaggle/working/gen_humorous.json

import json, argparse, torch
import numpy as np

SPLITS = '/kaggle/working/splits'
FEATS  = '/kaggle/working/style_feats'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------- load the generated captions (tolerant of a few json shapes) ----------
def load_preds(path):
    data = json.load(open(path, encoding='utf-8'))
    pairs = []  # list of (image_id, caption)
    if isinstance(data, dict):
        for k, v in data.items():
            cap = v if isinstance(v, str) else (v.get('caption') or v.get('text'))
            pairs.append((str(k), cap))
    else:  # list
        for d in data:
            if isinstance(d, str):
                pairs.append((None, d))
            else:
                iid = d.get('image') or d.get('image_id') or d.get('id')
                cap = d.get('caption') or d.get('text')
                pairs.append((str(iid) if iid is not None else None, cap))
    pairs = [(i, c) for i, c in pairs if c and c.strip()]
    return pairs


# --------------------------- 1. CLIPScore (image match) ---------------------------
def clipscore(pairs):
    from open_clip import create_model_from_pretrained, get_tokenizer
    model, _ = create_model_from_pretrained('nllb-clip-base-siglip', 'v1', device=device)
    tok = get_tokenizer('nllb-clip-base-siglip'); tok.set_language('ben_Beng'); model.eval()

    test = torch.load(f'{FEATS}/test_images.pt')           # {'ids':[...], 'emb':tensor}
    id2emb = {i: test['emb'][k] for k, i in enumerate(test['ids'])}

    caps, img = [], []
    for iid, cap in pairs:
        if iid in id2emb:
            caps.append(cap); img.append(id2emb[iid])
    if not caps:
        return float('nan'), 0
    img = torch.stack(img).to(device)
    with torch.no_grad():
        txt = []
        for s in range(0, len(caps), 64):
            txt.append(model.encode_text(tok(caps[s:s+64]).to(device)).float())
        txt = torch.cat(txt, 0)
        img = img.float()
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        sim = (img * txt).sum(-1)                            # cosine per caption
    return sim.mean().item(), len(caps)


# ------------------------- 2. Style accuracy (style strength) ----------------------
def style_acc(pairs, style):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def read(p, n=8000):
        L = [l.strip() for l in open(p, encoding='utf-8') if l.strip()]
        return L[:n]

    style_txt   = read(f'{SPLITS}/{style}_train.txt')          # label 1
    factual_txt = read(f'{SPLITS}/factualtext_train.txt')      # label 0
    X = style_txt + factual_txt
    y = [1] * len(style_txt) + [0] * len(factual_txt)

    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    Xv = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=1000).fit(Xv, y)

    gen = [c for _, c in pairs]
    pred = clf.predict(vec.transform(gen))
    return 100.0 * float(np.mean(pred)), len(gen)           # % called target style


# ----------------------------- 3. Perplexity (fluency) ----------------------------
def perplexity(pairs):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained('flax-community/gpt2-bengali')
    lm  = AutoModelForCausalLM.from_pretrained('flax-community/gpt2-bengali').to(device).eval()

    nlls, ntok = [], 0
    with torch.no_grad():
        for _, cap in pairs:
            ids = tok(cap, return_tensors='pt').input_ids.to(device)
            if ids.size(1) < 2:
                continue
            out = lm(ids, labels=ids)
            n = ids.size(1) - 1
            nlls.append(out.loss.item() * n); ntok += n
    if ntok == 0:
        return float('nan')
    return float(np.exp(sum(nlls) / ntok))


# ------------------------------------ main ---------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--style', required=True)      # romantic | humorous | factual
    ap.add_argument('--pred',  required=True)      # gen_<style>.json
    a, _ = ap.parse_known_args()

    pairs = load_preds(a.pred)
    print(f'\nstyle: {a.style}   captions: {len(pairs)}')

    cs, n_cs = clipscore(pairs)
    print(f'CLIPScore   : {cs:.4f}   (higher better, over {n_cs} images)')

    if a.style != 'factual':
        sa, n_sa = style_acc(pairs, a.style)
        print(f'Style acc   : {sa:.1f}%   (higher better, over {n_sa} captions)')

    pp = perplexity(pairs)
    print(f'Perplexity  : {pp:.1f}   (LOWER better)')
    print()
