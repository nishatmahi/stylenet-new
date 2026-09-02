# evaluate.py  (THE WORKING ONE - prints "=== evaluate.py v2 ===")
import os, sys, json, argparse, torch
import numpy as np

SPLITS = '/kaggle/working/splits'
FEATS  = '/kaggle/working/style_feats'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ID_KEYS = {'image_id', 'image', 'id', 'img', 'file', 'filename', 'references', 'ref'}


def load_preds(path, style=None):
    if not os.path.exists(path):
        return None
    data = json.load(open(path, encoding='utf-8'))
    items = data.values() if isinstance(data, dict) else data
    pairs = []
    for d in items:
        if isinstance(d, str):
            pairs.append((None, d)); continue
        iid = None
        for k in ('image_id', 'image', 'id', 'img'):
            if d.get(k): iid = str(d[k]); break
        cap = None
        if style and isinstance(d.get(style), str) and d[style].strip():
            cap = d[style]
        else:
            for k, v in d.items():
                if k in ID_KEYS: continue
                if isinstance(v, str) and v.strip():
                    cap = v; break
        if cap:
            pairs.append((iid, cap))
    return pairs


def clipscore(pairs):
    from open_clip import create_model_from_pretrained, get_tokenizer
    model, _ = create_model_from_pretrained('nllb-clip-base-siglip', 'v1', device=device)
    tok = get_tokenizer('nllb-clip-base-siglip'); tok.set_language('ben_Beng'); model.eval()
    test = torch.load(f'{FEATS}/test_images.pt')
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
        sim = (img * txt).sum(-1)
    return sim.mean().item(), len(caps)


def style_acc(pairs, style):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    def read(p, n=8000):
        return [l.strip() for l in open(p, encoding='utf-8') if l.strip()][:n]
    style_txt   = read(f'{SPLITS}/{style}_train.txt')
    factual_txt = read(f'{SPLITS}/factualtext_train.txt')
    X = style_txt + factual_txt
    y = [1] * len(style_txt) + [0] * len(factual_txt)
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(X), y)
    gen = [c for _, c in pairs]
    pred = clf.predict(vec.transform(gen))
    return 100.0 * float(np.mean(pred)), len(gen)


def perplexity(pairs):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained('flax-community/gpt2-bengali')
    lm  = AutoModelForCausalLM.from_pretrained('flax-community/gpt2-bengali').to(device).eval()
    nlls, ntok = [], 0
    with torch.no_grad():
        for _, cap in pairs:
            ids = tok(cap, return_tensors='pt').input_ids.to(device)
            if ids.size(1) < 2: continue
            out = lm(ids, labels=ids)
            n = ids.size(1) - 1
            nlls.append(out.loss.item() * n); ntok += n
    return float('nan') if ntok == 0 else float(np.exp(sum(nlls) / ntok))


if __name__ == '__main__':
    print('=== evaluate.py v2 ===')
    ap = argparse.ArgumentParser()
    ap.add_argument('--style', required=True)
    ap.add_argument('--pred',  required=True)
    a, _ = ap.parse_known_args()
    pairs = load_preds(a.pred, a.style)
    if pairs is None:
        print(f'FILE NOT FOUND: {a.pred}  -> generate captions for {a.style} first.')
        sys.exit(0)
    print(f'style: {a.style}   captions read: {len(pairs)}')
    if not pairs:
        print(f'0 captions found inside {a.pred}.')
        sys.exit(0)
    cs, n_cs = clipscore(pairs)
    print(f'CLIPScore   : {cs:.4f}   (higher better, over {n_cs} images)')
    if a.style != 'factual':
        sa, n_sa = style_acc(pairs, a.style)
        print(f'Style acc   : {sa:.1f}%   (higher better, over {n_sa} captions)')
    pp = perplexity(pairs)
    print(f'Perplexity  : {pp:.1f}   (LOWER better)')
