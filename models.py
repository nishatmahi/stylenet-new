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


class StyleNetT5(nn.Module):
    """
    StyleNet's alternating multi-task training procedure combined with
    FS-StyleCap's shared-encoder additive fusion, using one learned vector per
    named style rather than a few-shot style extractor.

    This is NOT StyleNet's W = U*S*V factorization: a factored matrix inside a
    T5 decoder FFN sits downstream of cross-attention and cannot transfer from
    text-only training. An additive vector keeps the style path and the image
    path separate, which is the property FactoredLSTM has by construction.

    Stated limitation: an additive vector shifts the mean of the content
    representations but cannot reweight their structure the way a factored
    matrix can. It is a blunter instrument.
    """
    def __init__(self, t5_ckpt="csebuetnlp/banglat5", clip_dim=768,
                 styles=("factual", "romantic"), gradient_checkpointing=False):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_ckpt)
        self.d_model = self.t5.config.d_model
        self.styles = list(styles)

        self.projector = VisualProjection(clip_dim, self.d_model)

        # Small random init, not zeros: only direction matters, since
        # magnitude is set at fusion time.
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
        """Mean per-position norm over REAL positions only.

        text_style_loss passes padded, variable-length sentences. Padded slots
        are masked out of attention but still carry hidden states, so an
        unmasked mean drifts with how much padding a batch contains — which
        defeats the point of matching style magnitude to content magnitude.
        """
        m = mask.unsqueeze(-1).to(content.dtype)
        norms = content.norm(dim=-1, keepdim=True) * m
        return (norms.sum() / m.sum().clamp(min=1e-6)).detach()

    def _style_at(self, target, lam, scale):
        """Unit-normalize each style vector, rescale to content magnitude,
        then interpolate: s = s_src + lam * (s_tgt - s_src).

        Without the rescale ||s|| was 0.33 against per-position content norms
        in the tens — a ~1% perturbation, so lam=1 read as factual and lam=3
        was needed for any visible effect. FS-StyleCap avoids this because its
        style vector is a mean-pooled encoder output, already at content scale.
        """
        def at_scale(v):
            return v / (v.norm() + 1e-6) * scale

        src = at_scale(self.style["factual"])
        if target == "factual" or lam == 0.0:
            return src
        return src + lam * (at_scale(self.style[target]) - src)

    def _fuse(self, content, mask, target, lam):
        s = self._style_at(target, lam, self._masked_scale(content, mask))
        return content + s.unsqueeze(0).unsqueeze(0)

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
        """L2-normalize both pooled vectors before the MSE.

        Unnormalized this read 0.001 at every step from the first — the vectors
        are small, so squared error and gradient are negligible and the loss
        was effectively switched off. FS-StyleCap's w/o L_V2L ablation drops
        CIDEr 66.26 -> 54.75, so it is not optional.
        """
        text_content, _ = self.encode_text(text_ids, text_mask)
        img_pool = F.normalize(img_content.mean(dim=1), dim=-1)
        m = text_mask.unsqueeze(-1).to(text_content.dtype)
        txt_pool = (text_content * m).sum(1) / m.sum(1).clamp(min=1e-6)
        txt_pool = F.normalize(txt_pool, dim=-1)
        return F.mse_loss(img_pool, txt_pool)

    def text_style_loss(self, corrupt_ids, corrupt_mask, labels, style, lam=1.0):
        """lam is SAMPLED during training so the model sees the whole
        interpolation path, not just its two endpoints.

        With lam fixed at 1.0 the forward pass only ever computes s_factual and
        s_romantic — two points in a 768-d space. Generation at lam=1.5 or 2.0
        is then extrapolation into untrained territory, which is why no single
        lam gave both grounding and style: lam=1 was the one known point and
        everything above was a guess that landed in the corpus mode.
        """
        content, mask = self.encode_text(corrupt_ids, corrupt_mask)
        return self._decode(content, mask, labels, style, lam=lam)

    # ---------- diagnostics ----------
    @torch.no_grad()
    def scales(self, feats):
        """Content vs style magnitude at fusion time. Should be the same order;
        before the rescale they differed ~50x."""
        content, mask = self.encode_image(feats)
        c = self._masked_scale(content, mask).item()
        out = {"content": c}
        for s in self.styles:
            out[s] = self._style_at(s, 1.0, c).norm().item()
        return out

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
