"""
data.py — CaptionData (factual, unchanged) + StyleData (now text-only).

StyleData reads plain text: the style corpus and the factual text corpus, each
one caption per line. No image, no CLIP embedding. That is the whole point of a
text-only discriminator, and it is why unpaired data is all it ever needs.
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


def read_style_lines(p):
    """One caption per line. Tolerates an optional leading 'id<TAB>' by keeping
    only the text after the first tab; plain lines are kept whole."""
    out = []
    for l in open(p, encoding='utf-8'):
        l = l.rstrip('\n')
        if '\t' in l:
            l = l.split('\t', 1)[1]
        l = l.strip()
        if l:
            out.append(l)
    return out


class CaptionData(Dataset):
    """One row per (image, caption). UNCHANGED."""

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
    """Style corpus (label 1, its own code) + factual text corpus (label 0,
    <factual>). Text only; no image ever loaded."""

    def __init__(self, style_file, factual_file, style, tok, sid, max_len=48,
                 ratio=1.0, seed=0):
        self.tok, self.max_len = tok, max_len
        self.rows = []
        s_lines = read_style_lines(style_file)
        for t in s_lines:
            self.rows.append((t, sid[style], 1))

        f_lines = read_style_lines(factual_file)
        keep = min(len(f_lines), int(round(len(s_lines) * ratio))) if ratio > 0 \
            else len(f_lines)
        idx = torch.randperm(len(f_lines),
                             generator=torch.Generator().manual_seed(seed))[:keep]
        for j in idx.tolist():
            self.rows.append((f_lines[j], sid['factual'], 0))
        if keep < len(f_lines):
            print(f"[data] factual text subsampled {len(f_lines)} -> {keep} "
                  f"(ratio {ratio} : 1 vs {len(s_lines)} {style})")
        print(f"[data] style set: {sum(r[2] for r in self.rows)} {style} + "
              f"{sum(1 - r[2] for r in self.rows)} factual = {len(self.rows)} rows")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        text, sid, is_style = self.rows[i]
        ids, attn, _ = encode_with_eos(self.tok, text, self.max_len)
        return ids, attn, torch.tensor(sid), torch.tensor(is_style)
