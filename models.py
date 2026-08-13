import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class VisualProjector(nn.Module):
    """Cached ViT patch tokens -> T5 d_model. The paper's matrix A."""
    def __init__(self, vit_hidden, d_model):
        super().__init__()
        self.A = nn.Linear(vit_hidden, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, raw_features):
        return self.ln(self.A(raw_features))


class FactoredStyleFFN(nn.Module):
    """
    StyleNet's W = U * S_style * V (paper eq. 8), applied to a decoder FFN:

        y = base_ffn(x) + U( S_style( V(x) ) )

    {U} and {V} shared across styles; {S} style-specific — exactly the split
    in paper sec 3.2. U zero-init and every S identity-init means the branch
    is an exact no-op at step 0, so the style path perturbs the grounded
    captioner instead of replacing it. Rewriting the FFN as U*S*V outright
    would discard BanglaT5's pretrained weights.
    """
    def __init__(self, base_ffn, d_model, factored_dim, styles):
        super().__init__()
        self.base = base_ffn
        self.V = nn.Linear(d_model, factored_dim, bias=False)
        self.U = nn.Linear(factored_dim, d_model, bias=False)
        self.S = nn.ModuleDict({s: nn.Linear(factored_dim, factored_dim, bias=False)
                                for s in styles})

        nn.init.normal_(self.V.weight, std=0.02)
        nn.init.zeros_(self.U.weight)
        for lin in self.S.values():
            nn.init.eye_(lin.weight)

        self.mode = "factual"

    def forward(self, x):
        return self.base(x) + self.U(self.S[self.mode](self.V(x)))


class BanglaT5StyleCaptioner(nn.Module):
    def __init__(self, t5_ckpt="csebuetnlp/banglat5", vit_hidden=768,
                 factored_dim=512, styles=("factual", "romantic"),
                 memory_len=197, gradient_checkpointing=True):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_ckpt)
        self.config = self.t5.config
        self.d_model = self.config.d_model
        self.styles = list(styles)
        self.memory_len = memory_len

        # No resize_token_embeddings: tokenizer len (32100) < embedding rows
        # (32128); the spare rows are never indexed.

        self.projector = VisualProjector(vit_hidden, self.d_model)

        # The T5 text encoder is unreachable once encoder_outputs is supplied.
        # Dropping its blocks frees ~1.2GB, which pays for unfreezing the
        # decoder. Shared embeddings live on self.t5.shared and are kept.
        self.t5.encoder.block = nn.ModuleList()
        for p in self.t5.encoder.parameters():
            p.requires_grad = False

        self._adapters = []
        for block in self.t5.decoder.block:
            ff = block.layer[-1]
            ff.DenseReluDense = FactoredStyleFFN(
                ff.DenseReluDense, self.d_model, factored_dim, self.styles)
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
        """raw_features=None -> RANDOM NOISE memory, not zeros.

        Paper sec 3.3: the decoder starts from a visual vector with paired
        images and "a random noise vector otherwise". This matters: zero
        memory makes T5 cross-attention output exactly zero (no biases), so
        the style stage would train S_style in a regime where the image
        pathway is dead, and inference with a real image would be a condition
        S_style never saw. Noise keeps the pathway live and uninformative.
        Scale is ~N(0,1) to match the LayerNorm'd projector output.
        """
        if raw_features is None:
            mem = torch.randn(batch_size, self.memory_len, self.d_model,
                              device=device, dtype=dtype)
        else:
            mem = self.projector(raw_features.to(dtype))
        mask = torch.ones(mem.size(0), mem.size(1), dtype=torch.long, device=device)
        return mem, mask

    def forward(self, labels, raw_features=None, mode="factual"):
        """labels: [B, L] with -100 on pad. Returns (loss, logits)."""
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


def load_vit_for_inference(vit_name="google/vit-base-patch16-224-in21k", device="cuda"):
    """Only for images missing from the feature cache."""
    from transformers import ViTModel
    vit = ViTModel.from_pretrained(vit_name).to(device).eval()
    for p in vit.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def extract(images):
        return vit(pixel_values=images.to(device)).last_hidden_state

    return extract
