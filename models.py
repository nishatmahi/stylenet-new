import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class VisualProjection(nn.Module):
    """CLIP tokens -> T5 encoder input space, following FS-StyleCap's M_{V->L}."""
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


class StyleModulator(nn.Module):
    """Per-position style application with a hard magnitude bound.

    MSSRNet (KDD 2023): a fixed-sized vector "applies the same style
    information to all words forming coarse-grained control". TED
    (Neurocomputing 2023) reports the consequence as "suboptimal preservation
    of non-stylistic semantic content" — swimmers becoming a boat. Adding one
    vector identically to every position has a single degree of freedom, so
    style strength and content damage are the same knob.

    A sigmoid gate computed per position and per dimension lets the model
    learn WHERE style applies, separating those two.

    The delta is then renormalized so its per-position norm never exceeds
    alpha * ||content|| at that position. This is a structural guarantee, not
    a monitored hope: the modulator trains partly on text-only data while
    sitting on the image path, and the bound means the worst it can do is
    perturb content by a bounded fraction — it cannot overwrite it.

    proj is zero-initialized, so at step 0 this is an exact no-op.
    """
    def __init__(self, d_model, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(2 * d_model, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.gate.bias)

    def _delta(self, content, s):
        s_exp = s.unsqueeze(0).unsqueeze(0).expand_as(content)
        g = torch.sigmoid(self.gate(torch.cat([content, s_exp], dim=-1)))
        return g * self.proj(s_exp), g

    def forward(self, content, s):
        delta, _ = self._delta(content, s)
        cn = content.norm(dim=-1, keepdim=True)
        dn = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = torch.clamp(self.alpha * cn / dn, max=1.0)
        return content + delta * scale

    @torch.no_grad()
    def report(self, content, s):
        """Gate spread and the realized delta-to-content ratio.
        gate std ~0 means it degenerated back to a global vector."""
        delta, g = self._delta(content, s)
        cn = content.norm(dim=-1, keepdim=True)
        dn = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = torch.clamp(self.alpha * cn / dn, max=1.0)
        ratio = ((delta * scale).norm(dim=-1) / cn.squeeze(-1).clamp(min=1e-6))
        return g.mean().item(), g.std().item(), ratio.mean().item()


class StyleNetT5(nn.Module):
    """
    StyleNet's alternating multi-task procedure, FS-StyleCap's shared-encoder
    fusion, and MSSRNet's per-position style application.

    Not StyleNet's W = U*S*V: a factored matrix inside a T5 decoder FFN sits
    downstream of cross-attention and cannot transfer from text-only training.
    """
    def __init__(self, t5_ckpt="csebuetnlp/banglat5", clip_dim=768,
                 styles=("factual", "romantic"), alpha=0.5):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_ckpt)
        self.d_model = self.t5.config.d_model
        self.styles = list(styles)

        self.projector = VisualProjection(clip_dim, self.d_model)
        self.modulator = StyleModulator(self.d_model, alpha=alpha)

        self.style = nn.ParameterDict({
            s: nn.Parameter(torch.randn(self.d_model) * 0.02)
            for s in self.styles
        })

    # ---------- encoders ----------
    def encode_image(self, feats):
        embeds = self.projector(feats.to(self.projector.inp.weight.dtype))
        out = self.t5.encoder(inputs_embeds=embeds)
        mask = torch.ones(out.last_hidden_state.shape[:2],
                          dtype=torch.long, device=feats.device)
        return out.last_hidden_state, mask

    def encode_text(self, ids, mask):
        out = self.t5.encoder(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state, mask

    # ---------- style vector ----------
    @staticmethod
    def _masked_scale(content, mask):
        """Mean per-position norm over REAL positions only. Padded slots are
        masked out of attention but still carry hidden states, so an unmasked
        mean drifts with how much padding a batch contains."""
        m = mask.unsqueeze(-1).to(content.dtype)
        norms = content.norm(dim=-1, keepdim=True) * m
        return (norms.sum() / m.sum().clamp(min=1e-6)).detach()

    def _style_at(self, target, lam, scale):
        """s = s_src + lam * (s_tgt - s_src), both rescaled to content
        magnitude so lam operates on a comparable scale."""
        def at_scale(v):
            return v / (v.norm() + 1e-6) * scale

        src = at_scale(self.style["factual"])
        if target == "factual" or lam == 0.0:
            return src
        return src + lam * (at_scale(self.style[target]) - src)

    def _fuse(self, content, mask, target, lam):
        s = self._style_at(target, lam, self._masked_scale(content, mask))
        return self.modulator(content, s)

    def _decode(self, content, mask, labels, target, lam=1.0):
        mem = self._fuse(content, mask, target, lam)
        out = self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=mem),
                      attention_mask=mask, labels=labels)
        return out.loss

    # ---------- training losses ----------
    def caption_loss(self, feats, labels):
        content, mask = self.encode_image(feats)
        loss = self._decode(content, mask, labels, "factual")
        return loss, content

    def v2l_loss(self, img_content, text_ids, text_mask):
        """L2-normalize both pooled vectors before the MSE. Unnormalized this
        read 0.001 at every step and was effectively switched off."""
        text_content, _ = self.encode_text(text_ids, text_mask)
        img_pool = F.normalize(img_content.mean(dim=1), dim=-1)
        m = text_mask.unsqueeze(-1).to(text_content.dtype)
        txt_pool = (text_content * m).sum(1) / m.sum(1).clamp(min=1e-6)
        txt_pool = F.normalize(txt_pool, dim=-1)
        return F.mse_loss(img_pool, txt_pool)

    def text_style_loss(self, corrupt_ids, corrupt_mask, labels, style, lam=1.0):
        """lam is sampled during training so the interpolation path is trained,
        not just its two endpoints."""
        content, mask = self.encode_text(corrupt_ids, corrupt_mask)
        return self._decode(content, mask, labels, style, lam=lam)

    # ---------- diagnostics ----------
    @torch.no_grad()
    def style_geometry(self):
        """If cos(fac,style) ~ 1 the directions coincide, s_tgt - s_src is
        near zero, and lam does nothing at any value."""
        f = self.style["factual"]
        f = f / (f.norm() + 1e-6)
        out = {}
        for s in self.styles:
            if s == "factual":
                continue
            v = self.style[s] / (self.style[s].norm() + 1e-6)
            out[f"cos(fac,{s})"] = torch.dot(f, v).item()
            out[f"gap(fac,{s})"] = (v - f).norm().item()
        return out

    @torch.no_grad()
    def gate_report(self, feats, target="romantic", lam=1.0):
        content, mask = self.encode_image(feats)
        s = self._style_at(target, lam, self._masked_scale(content, mask))
        return self.modulator.report(content, s)

    # ---------- inference ----------
    @torch.no_grad()
    def generate(self, feats, target="factual", lam=1.0, num_beams=5,
                 max_new_tokens=40, repetition_penalty=1.2,
                 no_repeat_ngram_size=3, length_penalty=1.0):
        content, mask = self.encode_image(feats)
        mem = self._fuse(content, mask, target, lam)
        return self.t5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=mem),
            attention_mask=mask, num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=True, use_cache=True)


def load_compatible(model, state_dict, verbose=True):
    own = model.state_dict()
    ok = {k: v for k, v in state_dict.items()
          if k in own and own[k].shape == v.shape}
    model.load_state_dict(ok, strict=False)
    if verbose:
        print(f"[LOAD] restored {len(ok)}/{len(own)} tensors")
    return len(ok)
