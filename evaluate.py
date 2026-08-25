"""
evaluate.py — the paper's three axes (Sec. 4.1).

  relevance to the image   BLEU-1, BLEU-3, CIDEr   (METEOR optional, needs Java)
  fluency                  ppl under an independent Bangla LM
  style accuracy           cls, a logistic-regression classifier trained on
                           your own styled vs factual corpora, exactly as the
                           paper describes

Relevance is scored against the FACTUAL references, because the style corpus
is monolingual — no styled sentence belongs to any particular image, so a
styled reference would have to be invented. That is the same choice MemCap and
PPCap make, and it is why the published romantic CIDEr is 32.6 rather than
something near 100: the metric is measuring content survival, not style.

    pip install pycocoevalcap scikit-learn
    python evaluate.py --pred /kaggle/working/ppcap_generations.json \
        --style romantic \
        --style_corpus  /kaggle/working/splits/romantic_train.txt \
        --factual_corpus /kaggle/working/splits/factualtext_train.txt
"""

import json
import argparse

from normalizer import normalize


def load_corpus(path):
    return [normalize(l.strip()) for l in open(path, encoding='utf-8') if l.strip()]


def build_classifier(style_lines, factual_lines, seed=0):
    """Paper Sec. 4.1: 'we train a logistic regression classifier using
    stylized captions and factual captions for each of the four styles.'
    Character n-grams, because Bangla is morphologically rich and word-level
    features miss inflected forms of the same stem."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split

    X = style_lines + factual_lines
    y = [1] * len(style_lines) + [0] * len(factual_lines)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1,
                                          random_state=seed, stratify=y)
    clf = make_pipeline(
        TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                        min_df=2, max_features=200_000),
        LogisticRegression(max_iter=2000, C=4.0))
    clf.fit(Xtr, ytr)
    held = clf.score(Xte, yte)
    print(f"[cls] held-out accuracy of the style classifier itself: {held:.3f}")
    if held < 0.9:
        print("[cls] WARNING: the classifier cannot separate your two corpora "
              "well, so the cls numbers below mean little.")
    return clf


def coco_scores(preds, refs):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    gts = {i: r for i, r in enumerate(refs)}
    res = {i: [p] for i, p in enumerate(preds)}
    bleu, _ = Bleu(4).compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)
    return {'bleu1': bleu[0], 'bleu3': bleu[2], 'bleu4': bleu[3], 'cider': cider}


def perplexity(lines, model_name, device='cuda'):
    """Fluency under an independent Bangla LM. The paper uses SRILM; any LM
    the system was not trained against works, as long as the SAME one is used
    for every row of the table."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    tot_nll, tot_tok = 0.0, 0
    with torch.no_grad():
        for s in lines:
            ids = tok(s, return_tensors='pt').input_ids.to(device)
            if ids.size(1) < 2:
                continue
            out = lm(ids, labels=ids)
            tot_nll += out.loss.item() * (ids.size(1) - 1)
            tot_tok += ids.size(1) - 1
    return float(torch.exp(torch.tensor(tot_nll / max(tot_tok, 1))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', required=True)
    p.add_argument('--style', required=True)
    p.add_argument('--style_corpus', required=True)
    p.add_argument('--factual_corpus', required=True)
    p.add_argument('--ppl_model', default='',
                   help='e.g. flax-community/gpt2-bengali. empty = skip ppl')
    p.add_argument('--out', default='')
    args = p.parse_args()

    data = json.load(open(args.pred, encoding='utf-8'))
    refs = [d['references'] for d in data]

    keys = [k for k in data[0] if k == 'factual' or k.startswith(f"{args.style}_w")]
    keys.sort(key=lambda k: 0.0 if k == 'factual' else float(k.split('_w')[1]))

    clf = build_classifier(load_corpus(args.style_corpus),
                           load_corpus(args.factual_corpus))

    print(f"\n{'setting':<18}{'BLEU-1':>9}{'BLEU-3':>9}{'CIDEr':>9}"
          f"{'cls':>9}{'ppl':>9}")
    print("-" * 63)

    table = {}
    for k in keys:
        preds = [d[k] for d in data]
        m = coco_scores(preds, refs)
        m['cls'] = float(clf.predict([normalize(t) for t in preds]).mean())
        m['ppl'] = perplexity(preds, args.ppl_model) if args.ppl_model else float('nan')
        table[k] = m
        print(f"{k:<18}{m['bleu1']*100:>9.1f}{m['bleu3']*100:>9.1f}"
              f"{m['cider']*100:>9.1f}{m['cls']*100:>9.1f}{m['ppl']:>9.1f}")

    print("\nReference points from the PPCap paper on English FlickrStyle10k:")
    print("  romantic  BLEU-1 22.3  BLEU-3  5.3  CIDEr 32.6  ppl 35.8  cls 95.9")
    print("  humorous  BLEU-1 21.3  BLEU-3  3.9  CIDEr 27.5  ppl 43.9  cls 90.3")
    print("  MemCap    romantic CIDEr 19.7 / humorous 18.5 for comparison")
    print("\nRead the sweep the way Fig. 5 does: cls climbs and CIDEr falls as w "
          "rises. Take the smallest w whose cls clears ~90%.")

    if args.out:
        json.dump(table, open(args.out, 'w'), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
