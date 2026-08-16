import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class VisualProjector(nn.Module):
    """ViT CLS token -> T5 d_model. The paper's matrix A."""
    def __init__(self, vit_hidden, d_model):
        super().__init__()
        self.A = nn.Linear(vit_hidden, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, raw_features):
        return self.ln(self.A(raw_features))


class FactoredStyleFFN(nn.Module):
    """
    StyleNet's style factor inline in the decoder FFN:

        y = base_ffn( S_style(x) )

    Paper eq. 8 is W_x = U * S * V — a factorization, not a side branch, so
    every bit of signal passes through S. The pretrained FFN projection plays
    the shared U*V; S_style is the only style-specific parameter (sec 3.2).

    S is identity-initialized: exact no-op at step 0, full-strength gradient.
    An additive branch base(x) + U(S(V(x))) with zero-init shared U gives S
    EXACTLY zero gradient in the style stage, since dL/dS = U^T . grad . V(x)^T
    and U is frozen at zero there.
    """
    def __init__(self, base_ffn, d_model, styles):
        super().__init__()
        self.base = base_ffn
        self.S = nn.ModuleDict({s: nn.Linear(d_model, d_model, bias=False)
                                for s in styles})
        for lin in self.S.values():
            nn.init.eye_(lin.weight)
        self.mode = "factual"

    def forward(self, x):
        return self.base(self.S[self.mode](x))


class BanglaT5StyleCaptioner(nn.Module):
    def __init__(self, t5_ckpt="csebuetnlp/banglat5", vit_hidden=768,
                 styles=("factual", "romantic"), gradient_checkpointing=False):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_ckpt)
        self.config = self.t5.config
        self.d_model = self.config.d_model
        self.styles = list(styles)

        self.projector = VisualProjector(vit_hidden, self.d_model)
        self.register_buffer("null_memory", torch.zeros(1, 1, self.d_model))

        # Text encoder is unreachable once encoder_outputs is supplied.
        # Dropping its blocks frees ~1.2GB; shared embeddings stay on t5.shared.
        self.t5.encoder.block = nn.ModuleList()
        for p in self.t5.encoder.parameters():
            p.requires_grad = False

        self._adapters = []
        for block in self.t5.decoder.block:
            ff = block.layer[-1]
            ff.DenseReluDense = FactoredStyleFFN(
                ff.DenseReluDense, self.d_model, self.styles)
            self._adapters.append(ff.DenseReluDense)

        if gradient_checkpointing:
            self.t5.gradient_checkpointing_enable()
            self.t5.config.use_cache = False

    def set_mode(self, mode):
        if mode not in self.styles:
            raise ValueError(f"Unknown mode: {mode}. Available: {self.styles}")
        for a in self._adapters:
            a.mode = mode

    def _memory(self, raw_features, batch_size, device, dtype):
        """Memory length is ALWAYS 1 — StyleNet's single global visual vector.

        With one memory position the cross-attention softmax is exactly 1.0 for
        any query, so the layer reduces to W_o(V(mem)): a constant additive term
        at every decoder position. Zero memory (style stage) gives exactly 0;
        the CLS vector (inference) gives a per-image constant. The two regimes
        then differ by that constant alone rather than by the entire attention
        distribution — which is what made 197-token memory untransferable for
        S_style, since it was fit with a 1-way softmax and used with a 197-way one.
        """
        if raw_features is None:
            mem = self.null_memory.expand(batch_size, 1, -1).to(dtype)
        else:
            mem = self.projector(raw_features[:, 0:1, :].to(dtype))   # CLS token
        mask = torch.ones(mem.size(0), mem.size(1), dtype=torch.long, device=device)
        return mem, mask

    def forward(self, labels, raw_features=None, mode="factual"):
        self.set_mode(mode)
        dtype = self.projector.A.weight.dtype
        mem, mask = self._memory(raw_features, labels.size(0), labels.device, dtype)
        out = self.t5(
            encoder_outputs=BaseModelOutput(last_hidden_state=mem),
            attention_mask=mask,
            labels=labels,
        )
        return out.loss, out.logits

    @torch.no_grad()
    def generate_caption(self, raw_features=None, mode="factual", batch_size=1,
                         num_beams=5, max_new_tokens=40, repetition_penalty=1.2,
                         no_repeat_ngram_size=3, length_penalty=1.0):
        self.set_mode(mode)
        device = next(self.parameters()).device
        dtype = self.projector.A.weight.dtype
        bs = raw_features.size(0) if raw_features is not None else batch_size
        mem, mask = self._memory(raw_features, bs, device, dtype)
        return self.t5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=mem),
            attention_mask=mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=True,
            use_cache=True,
        )


def load_compatible(model, state_dict, verbose=True):
    """Load only keys present with matching shape."""
    own = model.state_dict()
    ok, skipped = {}, []
    for k, v in state_dict.items():
        if k in own and own[k].shape == v.shape:
            ok[k] = v
        else:
            skipped.append(k)
    model.load_state_dict(ok, strict=False)
    if verbose:
        print(f"[LOAD] restored {len(ok)}/{len(own)} tensors, skipped {len(skipped)}")
    return len(ok)


def load_vit_for_inference(vit_name="google/vit-base-patch16-224-in21k", device="cuda"):
    from transformers import ViTModel
    vit = ViTModel.from_pretrained(vit_name).to(device).eval()
    for p in vit.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def extract(images):
        return vit(pixel_values=images.to(device)).last_hidden_state

    return extract
