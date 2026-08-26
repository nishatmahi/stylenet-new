"""
data.py — datasets for both models.

  CaptionData   (image features, factual caption)                -> factual
  StyleData     (CLIP text embedding, styled text, style token)  -> style

The style side never sees an image during training. That is the whole point.

EOS handling: gpt2-bengali has bos == eos == pad == token 0, and GPT-2's
tokenizer never appends EOS (tokenization_gpt2.py:229). Masking labels by
`lab == pad_token_id` would therefore erase the stop token too, and the model
would never learn to finish a sentence. So EOS is appended explicitly and
labels are masked from the ATTENTION MASK, not from the token id.
"""
import os, torch
from torch.utils.data import Dataset

STYLE_TOKENS = {"factual": "<factual>",
                "romantic": "<romantic>",
                "humorous": "<humorous>"}


def add_style_tokens(tok):
    tok.add_special_tokens({"additional_special_tokens": list(STYLE_TOKENS.values())})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return {k: tok.convert_tokens_to_ids(v) for k, v in STYLE_TOKENS.items()}


def encode_with_eos(tok, text, max_len):
    """ids, attention mask, labels -- with a real EOS the model can learn."""
    ids = tok(text, max_length=max_len - 1, truncation=True)["input_ids"]
    ids = ids + [tok.eos_token_id]
    attn = [1] * len(ids)
    n = max_len - len(ids)
    ids = ids + [tok.pad_token_id] * n
    attn = attn + [0] * n
    lab = [i if a else -100 for i, a in zip(ids, attn)]
    return (torch.tensor(ids), torch.tensor(attn), torch.tensor(lab))


def split_line(line):
    """'1000092795.jpg#0<TAB>caption' -> ('1000092795', 'caption')"""
    if '\t' in line:
        k, c = line.split('\t', 1)
    elif '#' in line:
        k, c = line.split('#', 1)
        p = c.split(None, 1)
        c = p[1] if len(p) > 1 else ''
    else:
        return None
    k = k.split('#')[0].strip()
    for e in ('.jpg', '.jpeg', '.png'):
        if k.lower().endswith(e):
            k = k[:-len(e)]
    return k, c.strip()


def read_lines(p):
    return [l.strip() for l in open(p, encoding='utf-8') if l.strip()]


class CaptionData(Dataset):
    """One row per (image, caption). Rows for an image are consecutive."""

    def __init__(self, cache_dir, path, tok, max_len=48):
        self.cache, self.tok, self.max_len = cache_dir, tok, max_len
        self.rows, seen = [], {}
        for ln in read_lines(path):
            p = split_line(ln)
            if not p or not p[1]:
                continue
            img, cap = p
            if img not in seen:
                seen[img] = os.path.exists(os.path.join(cache_dir, f"{img}.pt"))
            if seen[img]:
                self.rows.append((img, cap))
        if not self.rows:
            raise RuntimeError(f"no usable rows in {path}")
        print(f"[data] {os.path.basename(path)}: {len(self.rows)} captions, "
              f"{sum(seen.values())} images")

    def distinct_images(self, k):
        """First k DISTINCT images -- features and ids, aligned."""
        seen, feats, ids = set(), [], []
        for img, _ in self.rows:
            if img in seen:
                continue
            seen.add(img); ids.append(img)
            feats.append(torch.load(os.path.join(self.cache, f"{img}.pt"),
                                    map_location='cpu').float())
            if len(ids) == k:
                break
        return torch.stack(feats), ids

    def first_caption(self):
        g = {}
        for img, cap in self.rows:
            g.setdefault(img, cap)
        return g

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img, cap = self.rows[i]
        f = torch.load(os.path.join(self.cache, f"{img}.pt"), map_location='cpu')
        ids, attn, lab = encode_with_eos(self.tok, cap, self.max_len)
        return f.float(), ids, attn, lab


class StyleData(Dataset):
    """Styled text + the factual text corpus, each tagged with its control code.

    The CLIP embedding is the TEXT tower's output at training time; at
    inference the IMAGE tower's output goes in the same slot. Both come from
    the same NLLB-CLIP model, which is what makes the substitution valid.
    """

    def __init__(self, style_pt, factual_pt, style, tok, sid, max_len=48):
        self.tok, self.max_len = tok, max_len
        self.rows = []
        for path, name in ((style_pt, style), (factual_pt, "factual")):
            d = torch.load(path, map_location='cpu')
            emb, lines = d['emb'].float(), d['lines']
            assert len(emb) == len(lines), f"{path}: {len(emb)} vs {len(lines)}"
            for e, t in zip(emb, lines):
                self.rows.append((e, t, sid[name], 1 if name == style else 0))
        print(f"[data] style set: {sum(r[3] for r in self.rows)} {style} + "
              f"{sum(1-r[3] for r in self.rows)} factual = {len(self.rows)} rows, "
              f"dim {self.rows[0][0].numel()}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        emb, text, sid, is_style = self.rows[i]
        ids, attn, _ = encode_with_eos(self.tok, text, self.max_len)
        return emb, ids, attn, torch.tensor(sid), torch.tensor(is_style)
