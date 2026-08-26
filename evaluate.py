"""
evaluate.py — step 4. Score the sweep and pick w.

  relevance : BLEU-1/3, CIDEr against the FACTUAL references. The style corpus
              is monolingual, so no styled sentence belongs to any image and a
              styled reference would have to be invented. Same choice MemCap
              and PPCap make -- which is why published romantic CIDEr is 32.6,
              not something near 100.
  style     : accuracy of a logistic-regression classifier trained on your own
              style vs factual corpora (paper Sec 4.1).

Pick the SMALLEST w whose style accuracy clears ~90%: cls rises and CIDEr
falls as w rises (paper Fig. 5).
"""
import json, argparse, re


def classifier(style_corpus, factual_corpus, n=20000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    rd = lambda p: [l.strip() for l in open(p, encoding='utf-8') if l.strip()][:n]
    s, f = rd(style_corpus), rd(factual_corpus)
    X, y = s + f, [1] * len(s) + [0] * len(f)
    clf = make_pipeline(TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                                        max_features=60000),
                        LogisticRegression(max_iter=2000, C=4.0))
    clf.fit(X, y)
    print(f"[cls] trained on {len(s)} style + {len(f)} factual, "
          f"train acc {clf.score(X, y):.3f}")
    return clf


def relevance(preds, refs):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    gts = {i: r for i, r in enumerate(refs)}
    res = {i: [p] for i, p in enumerate(preds)}
    bleu, _ = Bleu(4).compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)
    return bleu[0], bleu[2], cider


def main(a):
    data = json.load(open(a.pred, encoding='utf-8'))
    ws = sorted({k for r in data for k in r if re.fullmatch(r'w[\d.]+', k)},
                key=lambda k: float(k[1:]))
    refs = [r['references'] for r in data]
    clf = classifier(a.style_corpus, a.factual_corpus)

    print(f"\n{'w':>6}{'BLEU-1':>9}{'BLEU-3':>9}{'CIDEr':>9}{'cls %':>9}")
    rows = {}
    for k in ws:
        preds = [r[k] for r in data]
        b1, b3, cd = relevance(preds, refs)
        cls = clf.predict(preds).mean()
        rows[k] = dict(bleu1=b1, bleu3=b3, cider=cd, cls=cls)
        print(f"{k[1:]:>6}{b1*100:9.1f}{b3*100:9.1f}{cd*100:9.1f}{cls*100:9.1f}")

    ok = [k for k in ws if rows[k]['cls'] >= a.target_cls]
    print(f"\npick: {ok[0][1:] if ok else 'none reaches ' + str(a.target_cls)}"
          f"   (smallest w clearing {a.target_cls*100:.0f}% style accuracy)")
    print("\npaper, FlickrStyle10k (English):")
    print("  romantic  BLEU-1 22.3  BLEU-3 5.3  CIDEr 32.6  cls 95.9")
    print("  humorous  BLEU-1 21.3  BLEU-3 3.9  CIDEr 27.5  cls 90.3")
    if a.out:
        json.dump(rows, open(a.out, 'w'), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pred', required=True)
    p.add_argument('--style_corpus', required=True)
    p.add_argument('--factual_corpus', required=True)
    p.add_argument('--target_cls', type=float, default=0.90)
    p.add_argument('--out', default='')
    main(p.parse_args())
