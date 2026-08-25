"""Verification harness for guided_beam_search. Not part of the pipeline.

Checks the three things most likely to be wrong in a hand-written beam search
with two auxiliary models: the beam/token index algebra, that w=0 reduces
exactly to plain beam search over the factual model, and that a confident
style model actually moves the output when w > 0.
"""
import sys, types, math, itertools
import torch
import torch.nn.functional as F

# ---- stub the imports generate.py makes, so it can be imported standalone --
class BaseModelOutput:
    def __init__(self, last_hidden_state): self.last_hidden_state = last_hidden_state

mo = types.ModuleType("transformers.modeling_outputs"); mo.BaseModelOutput = BaseModelOutput
tr = types.ModuleType("transformers"); tr.modeling_outputs = mo
sys.modules["transformers"] = tr; sys.modules["transformers.modeling_outputs"] = mo
for name, attrs in [("data", ["build_tokenizer", "Encoder", "split_line", "normalize"]),
                    ("models", ["FactualCaptioner", "StyleModel", "load_compatible"])]:
    m = types.ModuleType(name)
    for a in attrs: setattr(m, a, lambda *x, **k: None)
    sys.modules[name] = m

from generate import guided_beam_search   # noqa: E402

V, PAD, EOS, DSTART = 7, 0, 1, 0


class FakeT5:
    """Emits a fixed logit table indexed by the last generated token."""
    def __init__(self, table):
        self.table = table                      # [V, V]
        self.config = types.SimpleNamespace(vocab_size=V, decoder_start_token_id=DSTART)

    def __call__(self, encoder_outputs=None, attention_mask=None,
                 decoder_input_ids=None, **kw):
        last = decoder_input_ids[:, -1]                       # [N]
        bias = encoder_outputs.last_hidden_state[:, 0, 0]     # per-row offset
        logits = self.table[last] + bias.unsqueeze(1)
        return types.SimpleNamespace(logits=logits.unsqueeze(1))


class FakeFactual:
    def __init__(self, table): self.t5 = FakeT5(table)
    def encode_image(self, feats):
        h = feats.unsqueeze(-1)                               # [B,1] -> [B,1,1]
        return h, torch.ones(h.shape[:2], dtype=torch.long)


class FakeStyle:
    def __init__(self, table): self.t5 = FakeT5(table)
    def prefix_states(self, clip_emb, style_ids):
        h = (clip_emb[:, :1] * 0 + style_ids.float().unsqueeze(1) * 0).unsqueeze(1)
        return h, torch.ones(h.shape[:2], dtype=torch.long)


class FakeTok:
    pad_token_id, eos_token_id = PAD, EOS
    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(str(int(t)) for t in row if int(t) not in (PAD, EOS))
                for row in ids]


def brute_force_best(table, max_len):
    """Exhaustive search under the SAME objective the beam uses: total
    log-probability divided by length (HF's length_penalty=1.0 convention).
    A sequence is valid if it ends at EOS, or runs the full max_len."""
    logp = F.log_softmax(table.float(), -1)
    best, best_s = None, -1e18
    for L in range(1, max_len + 1):
        for seq in itertools.product(range(V), repeat=L):
            if PAD in seq or EOS in seq[:-1]:
                continue
            if seq[-1] != EOS and L < max_len:
                continue
            s, prev = 0.0, DSTART
            for tkn in seq:
                s += logp[prev, tkn].item(); prev = tkn
            s /= L                                   # length-normalised
            if s > best_s:
                best_s, best = s, seq
    return best


torch.manual_seed(0)
table_f = torch.randn(V, V) * 2.0
table_f[:, PAD] = -20.0                       # never emit pad mid-sequence

fac = FakeFactual(table_f)
sty = FakeStyle(torch.zeros(V, V))            # uniform -> discriminator says nothing
tok = FakeTok()

B, K, T = 3, 4, 4
feats = torch.zeros(B, 1)
clip = torch.zeros(B, 2)

# --- 1. index algebra: runs, right shape, no crash ---------------------------
out = guided_beam_search(fac, sty, feats, clip, 2, 3, tok, w=0.0,
                         num_beams=K, max_new_tokens=T)
assert len(out) == B, out
print("1. shapes ok:", out)

# --- 2. w=0 must equal plain beam search over the factual model --------------
gold = brute_force_best(table_f, T)
gold_str = " ".join(str(t) for t in gold if t not in (PAD, EOS))
print(f"2. brute-force optimum: {gold_str!r}   beam output: {out[0]!r}")
assert out[0] == gold_str, f"w=0 diverges from plain beam search: {out[0]} vs {gold_str}"
print("   w=0 reduces to the factual model exactly")

# --- 3. a confident discriminator must move the output ----------------------
# pick a token the FACTUAL model did not choose, so any appearance of it is
# attributable to the discriminator and nothing else
chosen = {int(t) for t in out[0].split()}
hot_tok = next(v for v in range(2, V) if v not in chosen)
print(f"   (factual chose {sorted(chosen)}; discriminator will push token {hot_tok})")


class StyleT5:
    """Style enters through the ENCODER STATES, exactly as the real
    StyleModel does — prefix_states puts the control code in, and the decoder
    reads it from cross-attention. The earlier version of this fake carried
    the style on the model object, which the real code never does, so both
    branches silently read the same table and the discriminator was a no-op."""
    def __init__(self, hot_tok):
        self.hot_tok = hot_tok
        self.config = types.SimpleNamespace(vocab_size=V, decoder_start_token_id=DSTART)

    def __call__(self, encoder_outputs=None, attention_mask=None,
                 decoder_input_ids=None, **kw):
        marker = encoder_outputs.last_hidden_state[:, 0, 0]       # 1 desired, 0 not
        logits = torch.zeros(marker.size(0), V)
        logits[:, self.hot_tok] = marker * 8.0
        return types.SimpleNamespace(logits=logits.unsqueeze(1))


SID_DESIRED, SID_UNDESIRED = 2, 3


class StyleFake:
    def __init__(self, hot_tok): self.t5 = StyleT5(hot_tok)
    def prefix_states(self, clip_emb, style_ids):
        marker = (style_ids == SID_DESIRED).float().view(-1, 1, 1)
        return marker, torch.ones(marker.shape[:2], dtype=torch.long)


sty2 = StyleFake(hot_tok)
out_w = guided_beam_search(fac, sty2, feats, clip, SID_DESIRED, SID_UNDESIRED,
                           tok, w=50.0, num_beams=K, max_new_tokens=T)
print(f"3. w=0  -> {out[0]!r}")
print(f"   w=50 -> {out_w[0]!r}")
assert str(hot_tok) in out_w[0], "the discriminator had no effect at w=50"
assert out_w[0] != out[0], "output identical with and without guidance"
print("   guidance moves the output toward the style token")

print("\nall three checks passed")
