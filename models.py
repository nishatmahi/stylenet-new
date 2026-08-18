%%writefile models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class VisualProjection(nn.Module):
    """CLIP tokens -> T5 encoder input space. A small transformer rather than
    a single linear layer, following FS-StyleCap's M_{V->L}."""
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
    Style is a VECTOR added to content vectors, not a weight matrix inside the
    decoder. StyleNet's factored S sits on the word-input path, which never
    touches the image; placed in a T5 decoder FFN it sits downstream of
    cross-attention instead, so it can never transfer from text-only training.
    As an additive vector on encoder outputs, the style path and the content
    path stay separate and combine additively — the FactoredLSTM property.

    The encoder is SHARED between the image path and the text path. The V2L
    loss pulls the two into the same space, which is what makes a style vector
    learned on text valid to apply to image content.

    Style vectors are zero-initialized, so at step 0 the model is a plain
    captioner and nothing is disturbed.
    """
    def __init__(self, t5_ckpt="csebuetnlp/banglat5", clip_dim=768,
                 styles=("factual", "romantic", "humorous"),
                 gradient_checkpointing=False):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_ckpt)
        self.d_model = self.t5.config.d_model
        self.styles = list(styles)

        self.projector = VisualProjection(clip_dim, self.d_model)
        self.style = nn.ParameterDict({
            s: nn.Parameter(torch.zeros(self.d_model)) for s in self.styles
        })

        if gradient_checkpointing:
            self.t5.gradient_checkpointing_enable()
            self.t5.config.use_cache = False

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

    # ---------- fusion + decode ----------
    def _fuse(self, content, style_vec):
        return content + style_vec.unsqueeze(0).unsqueeze(0)

    def _decode(self, content, mask, labels, style_vec):
        mem = self._fuse(content, style_vec)
        out = self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=mem),
                      attention_mask=mask, labels=labels)
        return out.loss

    # ---------- training losses ----------
    def caption_loss(self, feats, labels):
        content, mask = self.encode_image(feats)
        loss = self._decode(content, mask, labels, self.style["factual"])
        return loss, content

    def v2l_loss(self, img_content, text_ids, text_mask):
        """Pull mean-pooled image content toward mean-pooled text content for
        the SAME caption. Without this the two encoder outputs live in
        different regions and a text-learned style vector means nothing when
        added to image content."""
        with torch.no_grad():
            pass
        text_content, _ = self.encode_text(text_ids, text_mask)
        img_pool = img_content.mean(dim=1)
        m = text_mask.unsqueeze(-1).to(text_content.dtype)
        txt_pool = (text_content * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return F.mse_loss(img_pool, txt_pool)

    def text_style_loss(self, corrupt_ids, corrupt_mask, labels, style):
        content, mask = self.encode_text(corrupt_ids, corrupt_mask)
        return self._decode(content, mask, labels, self.style[style])

    # ---------- inference ----------
    def style_vector(self, target, lam=1.0):
        """s = lam * (s_tgt - s_factual) + s_factual. lam is the style-strength
        dial from FS-StyleCap sec 3.3; lam=0 is pure factual."""
        src = self.style["factual"]
        if target == "factual" or lam == 0.0:
            return src
        return src + lam * (self.style[target] - src)

    @torch.no_grad()
    def generate(self, feats, target="factual", lam=1.0, num_beams=5,
                 max_new_tokens=40, repetition_penalty=1.2,
                 no_repeat_ngram_size=3, length_penalty=1.0):
        content, mask = self.encode_image(feats)
        mem = self._fuse(content, self.style_vector(target, lam))
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
