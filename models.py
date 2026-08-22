import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class VisualProjection(nn.Module):
    """CLIP tokens -> T5 encoder input space. The encoder handles the picture
    and nothing else — it never sees style."""
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


class StyleContext:
    """Shared holder for the style dial. lam=0 means the adapters are off and
    the model is exactly the plain factual captioner."""
    def __init__(self, lam=0.0):
        self.lam = lam


class StyleAdapter(nn.Module):
    """The style knobs. One small bottleneck per decoder block.

    Up-projection is zero-initialised, so at step 0 the adapter output is
    exactly zero: factual and romantic start as the same model and every
    difference that appears later was learned.
    """
    def __init__(self, d_model, bottleneck=64, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.dropout(self.up(self.act(self.down(x))))


class AdapterFF(nn.Module):
    """Wraps a T5 decoder feed-forward block and adds the style delta.

        h = ff(x) + lam * adapter(ff(x))

    Residual, exactly like S_f + lam*S_r in your LSTM: the model's normal
    writing behaviour is never switched off, style is only ever added on top.
    lam multiplies the delta directly with nothing renormalising it, so the
    dial actually controls intensity.
    """
    def __init__(self, orig_ff, d_model, bottleneck, ctx: StyleContext):
        super().__init__()
        self.orig = orig_ff
        self.adapter = StyleAdapter(d_model, bottleneck)
        self.ctx = ctx
        self.last_ratio = 0.0          # diagnostic: ||delta|| / ||h||

    def forward(self, hidden_states):
        h = self.orig(hidden_states)
        if self.ctx.lam == 0.0:
            return h
        d = self.adapter(h)
        if not self.training:
            self.last_ratio = (d.norm(dim=-1).mean()
                               / h.norm(dim=-1).mean().clamp(min=1e-6)).item()
        return h + self.ctx.lam * d


class StyleNetT5(nn.Module):
    """One style per run, matching how you trained the LSTM.

    Encoder: picture only.
    Decoder: writing, with the style knobs inside it.
    lam=0 -> plain factual captioner.  lam=1 -> trained style strength.
    """
    def __init__(self, t5_ckpt="csebuetnlp/banglat5", clip_dim=768,
                 style="romantic", bottleneck=64):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_ckpt)
        self.d_model = self.t5.config.d_model
        self.style = style
        self.ctx = StyleContext(0.0)

        self.projector = VisualProjection(clip_dim, self.d_model)
        for block in self.t5.decoder.block:
            block.layer[-1] = AdapterFF(block.layer[-1], self.d_model,
                                        bottleneck, self.ctx)

    def adapter_modules(self):
        return [b.layer[-1] for b in self.t5.decoder.block]

    def adapter_parameters(self):
        return [p for m in self.adapter_modules() for p in m.adapter.parameters()]

    # ---------- encoders ----------
    def encode_image(self, feats):
        embeds = self.projector(feats.to(self.projector.inp.weight.dtype))
        out = self.t5.encoder(inputs_embeds=embeds)
        mask = torch.ones(out.last_hidden_state.shape[:2],
                          dtype=torch.long, device=feats.device)
        return out.last_hidden_state, mask

    def encode_text(self, ids, mask):
        return self.t5.encoder(input_ids=ids, attention_mask=mask).last_hidden_state, mask

    def _decode(self, content, mask, labels, lam):
        self.ctx.lam = lam
        out = self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=content),
                      attention_mask=mask, labels=labels)
        self.ctx.lam = 0.0
        return out.loss

    # ---------- training ----------
    def caption_loss(self, feats, labels):
        """Image -> factual caption. Knobs off."""
        content, mask = self.encode_image(feats)
        return self._decode(content, mask, labels, 0.0), content

    def v2l_loss(self, img_content, text_ids, text_mask):
        """Pull the visual representation toward the text representation so the
        knobs, which are trained on text, still apply when the input is an image.

        Text branch is under no_grad and is the fixed anchor: with both branches
        trainable through one encoder and both normalised, collapse to a single
        constant direction is the global optimum.

        Printed value is a mean over 768 dims of two unit vectors, so it looks
        tiny by construction:  cos = 1 - 384*mse.
        0.004 -> -0.54,  0.002 -> 0.23,  0.001 -> 0.62.
        """
        with torch.no_grad():
            tc, _ = self.encode_text(text_ids, text_mask)
            m = text_mask.unsqueeze(-1).to(tc.dtype)
            tp = F.normalize((tc * m).sum(1) / m.sum(1).clamp(min=1e-6), dim=-1)
        ip = F.normalize(img_content.mean(dim=1), dim=-1)
        return F.mse_loss(ip, tp.to(ip.dtype))

    def text_style_loss(self, ids, mask, labels, lam=1.0):
        """Rebuild a styled sentence whose style words were masked out of the
        input. The knobs are the only place the missing words can come from."""
        content, mask = self.encode_text(ids, mask)
        return self._decode(content, mask, labels, lam)

    # ---------- diagnostics ----------
    @torch.no_grad()
    def adapter_report(self, feats, lam=1.0, max_new_tokens=8):
        """||style delta|| / ||hidden|| averaged over decoder blocks.

        ~0.00  -> knobs are dead, nothing was learned. raise --lr_style.
        0.05-0.4 -> healthy.
        >1.0   -> style is swamping the content; lower lam or --lr_style.
        """
        was = self.training
        self.eval()
        self.ctx.lam = lam
        content, mask = self.encode_image(feats)
        self.t5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=content),
            attention_mask=mask, max_new_tokens=max_new_tokens, num_beams=1)
        self.ctx.lam = 0.0
        r = [m.last_ratio for m in self.adapter_modules()]
        if was:
            self.train()
        return sum(r) / max(len(r), 1)

    # ---------- inference ----------
    @torch.no_grad()
    def generate(self, feats, lam=1.0, num_beams=5, max_new_tokens=40,
                 repetition_penalty=1.0, no_repeat_ngram_size=0,
                 length_penalty=1.0):
        content, mask = self.encode_image(feats)
        self.ctx.lam = lam
        kw = dict(encoder_outputs=BaseModelOutput(last_hidden_state=content),
                  attention_mask=mask, num_beams=num_beams,
                  max_new_tokens=max_new_tokens, length_penalty=length_penalty,
                  early_stopping=True, use_cache=True)
        if repetition_penalty and repetition_penalty != 1.0:
            kw["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size:
            kw["no_repeat_ngram_size"] = no_repeat_ngram_size
        ids = self.t5.generate(**kw)
        self.ctx.lam = 0.0
        return ids


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
            f"The architecture changed since it was written — delete it, or "
            f"pass min_frac=0 to resume from a partly random model on purpose.")
    return len(ok)
