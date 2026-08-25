"""
models.py — the two models of PPCap, in Bangla.

They are trained separately, share no weights, and meet only in generate.py.

  FactualCaptioner   image -> factual caption.   Off-the-shelf, frozen at
                     inference. Sees images only, so it keeps your existing
                     CLIP ViT-B/32 patch-token cache.

  StyleModel         a small class-conditional captioner trained ONLY on the
                     unpaired style corpus. Its prefix is
                     [style embedding] ++ [CLIP embedding -> MLP], where the
                     CLIP embedding is the TEXT tower's output at training
                     time and the IMAGE tower's at inference (paper Fig. 3).
                     This is what lets it train with no images at all.

Both use BanglaT5, so they share a tokenizer. The paper had to zero out
mismatched vocabulary between its GPT-2 discriminator and its factual model
(Sec. 4.3); sharing a tokenizer removes that problem instead of patching it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


def load_banglat5(t5_ckpt):
    """Load BanglaT5 and verify it actually works before anything trains on it.

    BanglaT5's config.json says tie_word_embeddings=True, but the checkpoint
    ships a genuine SEPARATE output head. Measured on the real checkpoint:

        shared.weight   std 22.04     <- large, meant to be used as an output
                                         head only after T5 shrinks the hidden
                                         state by d_model**-0.5 (1/27.7)
        lm_head.weight  std  1.20     <- normal scale, expects it UNSCALED

    Transformers 4.x honoured the flag: it tied lm_head to shared AND applied
    the rescale. Self-consistent, and what the earlier working run used.
    Transformers 5.x keeps the stored lm_head but STILL applies the rescale,
    so the hidden state is shrunk 27.7x before a head that does not want it.
    The logits come out flat, and on BanglaT5's own span-fill objective the
    loss sits at ln(32100) = 10.4, i.e. chance. Training then blows up: cap
    starts near 103 instead of 22 and validation diverges by epoch 2.

    Forcing the tie is NOT the fix -- measured, that scores 49.87, five times
    worse than leaving it alone. The stored head is the real one; the rescale
    is what is wrong. Clearing the flag keeps the head and drops the rescale,
    which is exactly what the v5 warning tells you to do.

    Then it is checked rather than assumed. If the span-fill loss is still at
    chance this raises immediately instead of training for two epochs.
    """
    model = T5ForConditionalGeneration.from_pretrained(t5_ckpt)
    model.config.tie_word_embeddings = False

    from transformers import T5Tokenizer
    tok = T5Tokenizer.from_pretrained(t5_ckpt, use_fast=False)
    enc = tok(["একটি ছোট ছেলে <extra_id_0> উপর বসে আছে।"], return_tensors="pt")
    lab = tok(["<extra_id_0> একটি বড় পাথরের <extra_id_1>"],
              return_tensors="pt").input_ids
    model.eval()
    with torch.no_grad():
        loss = model(input_ids=enc.input_ids,
                     attention_mask=enc.attention_mask,
                     labels=lab).loss.item()
    model.train()

    n_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[banglat5] {n_m:.1f}M params, span-fill loss {loss:.2f} "
          f"(chance is 10.4; healthy is roughly 2-5)")

    if loss > 6.0:
        raise RuntimeError(
            f"BanglaT5 loaded but is at chance on its own pretraining task "
            f"(span-fill loss {loss:.2f}, chance = 10.4). The output head is "
            f"not being applied correctly by transformers "
            f"{__import__('transformers').__version__}. Do NOT train on this. "
            f"Try:  pip install 'transformers==4.46.3'  and restart the kernel.")
    return model

STYLE_TOKENS = {
    "factual":  "<factual>",
    "romantic": "<romantic>",
    "humorous": "<humorous>",
}


def add_style_tokens(tokenizer):
    """Add the control codes as real single tokens.

    The paper uses the literal strings ' romantic' / ' factual' and warns to
    keep the leading space, because otherwise they tokenize inconsistently.
    Dedicated tokens make that failure impossible.
    """
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(STYLE_TOKENS.values())})
    return {k: tokenizer.convert_tokens_to_ids(v) for k, v in STYLE_TOKENS.items()}


class MLP(nn.Module):
    """CLIP embedding -> prefix_len vectors in T5's embedding space.
    Same shape of mapping ClipCap uses, one hidden layer, tanh."""

    def __init__(self, in_dim, d_model, prefix_len):
        super().__init__()
        out = d_model * prefix_len
        self.net = nn.Sequential(
            nn.Linear(in_dim, out // 2),
            nn.Tanh(),
            nn.Linear(out // 2, out),
        )
        self.d_model, self.prefix_len = d_model, prefix_len

    def forward(self, x):
        return self.net(x).view(-1, self.prefix_len, self.d_model)


# ---------------------------------------------------------------- factual
class VisualProjection(nn.Module):
    """CLIP patch tokens -> T5 encoder input space."""

    def __init__(self, clip_dim, d_model, n_layers=2, n_heads=8):
        super().__init__()
        self.inp = nn.Linear(clip_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, feats):
        return self.ln(self.tr(self.inp(feats)))


class FactualCaptioner(nn.Module):
    """Image -> factual caption. No style machinery anywhere in here."""

    def __init__(self, t5_ckpt="csebuetnlp/banglat5", clip_dim=768):
        super().__init__()
        self.t5 = load_banglat5(t5_ckpt)
        self.d_model = self.t5.config.d_model
        self.projector = VisualProjection(clip_dim, self.d_model)

    def encode_image(self, feats):
        embeds = self.projector(feats.to(self.projector.inp.weight.dtype))
        out = self.t5.encoder(inputs_embeds=embeds)
        mask = torch.ones(out.last_hidden_state.shape[:2],
                          dtype=torch.long, device=feats.device)
        return out.last_hidden_state, mask

    def encode_text(self, ids, mask):
        return self.t5.encoder(input_ids=ids, attention_mask=mask).last_hidden_state

    def caption_loss(self, feats, labels):
        content, mask = self.encode_image(feats)
        out = self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=content),
                      attention_mask=mask, labels=labels)
        return out.loss, content

    def v2l_loss(self, img_content, text_ids, text_mask):
        """Pull the image representation toward its caption's representation.

        Text branch is no_grad — a fixed anchor. If both branches carried
        gradient the encoder could drive this to zero by emitting one constant
        direction for every input.

        Without this the projector's only gradient comes back through the
        decoder's cross-entropy, which the decoder can minimise by learning a
        generic caption prior: fluent Bangla about no particular picture.
        """
        with torch.no_grad():
            t = self.encode_text(text_ids, text_mask)
            m = text_mask.unsqueeze(-1).to(t.dtype)
            txt = F.normalize((t * m).sum(1) / m.sum(1).clamp(min=1e-6), dim=-1)
        img = F.normalize(img_content.mean(dim=1), dim=-1)
        return F.mse_loss(img, txt.to(img.dtype))

    @torch.no_grad()
    def generate(self, feats, num_beams=5, max_new_tokens=48,
                 repetition_penalty=1.15, no_repeat_ngram_size=3):
        content, mask = self.encode_image(feats)
        return self.t5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=content),
            attention_mask=mask, num_beams=num_beams,
            max_new_tokens=max_new_tokens, early_stopping=True, use_cache=True,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size)


# ---------------------------------------------------------------- stylized
def noise_injection(x, variance, normalize=True):
    """Paper Sec. 3.3: CLIP text and image embeddings are close but not
    interchangeable, so noise is injected during training to stop the model
    keying on the exact text-side geometry. variance = 0.016 in the paper."""
    if normalize:
        x = x / x.norm(2, dim=-1, keepdim=True).clamp(min=1e-6)
    if variance <= 0:
        return x
    return x + torch.randn_like(x) * variance


class StyleModel(nn.Module):
    """Class-conditional captioner trained on the unpaired style corpus.

    encoder input = [ style token embedding ] ++ [ MLP(clip embedding) ]
    decoder       = writes the caption

    Feeding the prefix through the encoder is the T5 form of ClipCap's
    "prepend to GPT-2" — the decoder cross-attends to style and content
    together, which is the same conditioning the paper describes.
    """

    def __init__(self, t5_ckpt="csebuetnlp/banglat5", clip_dim=768,
                 prefix_len=10, vocab_size=None):
        super().__init__()
        self.t5 = load_banglat5(t5_ckpt)
        if vocab_size is not None:
            self.t5.resize_token_embeddings(vocab_size)
        self.d_model = self.t5.config.d_model
        self.prefix_len = prefix_len
        self.clip_project = MLP(clip_dim, self.d_model, prefix_len)

    def prefix_states(self, clip_emb, style_ids):
        """[B, 1+prefix_len, d] encoder hidden states."""
        clip_emb = clip_emb.to(self.clip_project.net[0].weight.dtype)
        style = self.t5.shared(style_ids).unsqueeze(1)          # [B,1,d]
        prefix = self.clip_project(clip_emb)                    # [B,k,d]
        embeds = torch.cat([style, prefix], dim=1)
        out = self.t5.encoder(inputs_embeds=embeds)
        mask = torch.ones(out.last_hidden_state.shape[:2],
                          dtype=torch.long, device=embeds.device)
        return out.last_hidden_state, mask

    def token_nll(self, clip_emb, style_ids, labels):
        """Per-token negative log-likelihood, and the token count.

        Returned unreduced so the caller can build both losses of Eq. 6-8:
        the generative loss needs the length-normalised mean, and the
        discriminative loss needs that same mean as a class logit.
        """
        h, mask = self.prefix_states(clip_emb, style_ids)
        out = self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=h),
                      attention_mask=mask, labels=labels)
        logits = out.logits                                     # [B,T,V]
        nll = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            labels.view(-1), ignore_index=-100, reduction='none'
        ).view(labels.shape)                                    # [B,T]
        n_tok = (labels != -100).sum(1).clamp(min=1)
        return nll.sum(1), n_tok                                # [B], [B]

    def step_logits(self, h, mask, decoder_input_ids, past=None):
        """One decoding step. Returns (logits over vocab, new past)."""
        out = self.t5(
            encoder_outputs=BaseModelOutput(last_hidden_state=h),
            attention_mask=mask,
            decoder_input_ids=decoder_input_ids if past is None
            else decoder_input_ids[:, -1:],
            past_key_values=past, use_cache=True)
        return out.logits[:, -1, :], out.past_key_values


def ppcap_loss(model, clip_emb, true_ids, other_ids, labels, lam=0.8):
    """Paper Eq. 6-8.

        L_g = mean_i  -(1/T_i) log P(y_i | s_i, x_i)          generative
        L_d = mean_i  -log P(s_i | x_i, y_i)                  discriminative
        L   = lam * L_g + (1 - lam) * L_d

    P(s|x,y) is a softmax over the two styles of the length-normalised
    log-likelihood, which is what makes this a *discriminator*: the model
    must make the sentence likely under its own style AND unlikely under the
    contrasting one. Language modelling alone would only do the first, and
    that is exactly the signal that was missing before.
    """
    nll_true, n = model.token_nll(clip_emb, true_ids, labels)
    nll_other, _ = model.token_nll(clip_emb, other_ids, labels)

    mean_true = nll_true / n
    mean_other = nll_other / n

    l_g = mean_true.mean()

    class_logits = torch.stack([-mean_true, -mean_other], dim=1)   # [B,2]
    target = torch.zeros(class_logits.size(0), dtype=torch.long,
                         device=class_logits.device)               # index 0 = true
    l_d = F.cross_entropy(class_logits, target)

    return lam * l_g + (1.0 - lam) * l_d, l_g.detach(), l_d.detach()


def load_compatible(model, state_dict, verbose=True, min_frac=0.95):
    own = model.state_dict()
    ok, bad = {}, []
    for k, v in state_dict.items():
        if k in own and own[k].shape == v.shape:
            ok[k] = v
        else:
            why = "absent" if k not in own else f"{tuple(v.shape)}!={tuple(own[k].shape)}"
            bad.append(f"{k} ({why})")
    missing = [k for k in own if k not in ok]
    model.load_state_dict(ok, strict=False)
    frac = len(ok) / max(len(own), 1)
    if verbose:
        print(f"[LOAD] restored {len(ok)}/{len(own)} tensors ({frac:.1%})")
        for k in bad[:8]:
            print(f"[LOAD]   skipped: {k}")
        for k in missing[:8]:
            print(f"[LOAD]   left at init: {k}")
    if frac < min_frac:
        raise RuntimeError(
            f"checkpoint restored only {frac:.1%} (threshold {min_frac:.0%}). "
            f"The architecture changed since it was written.")
    return len(ok)
