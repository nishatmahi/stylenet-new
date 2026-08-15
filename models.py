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


class StyleEmbedding(nn.Module):
    """
    StyleNet's style factor on the word-embedding path:

        x_t -> S_style( B(x_t) )

    Paper eq. 9-12 apply S_x to V_x . x_t, where x_t is the word embedding.
    The visual vector only initializes h_0 and never passes through S, so S's
    input distribution is IDENTICAL in the captioning task and the style LM
    task. What S learns with no image therefore transfers to inference with one.

    Placing S inside the decoder FFN breaks that: there its input is the
    post-cross-attention residual stream, whose distribution depends on whether
    memory is present. S fits the zero-memory regime it trains in and misfires
    at generation — low style-stage loss and incoherent captions at once.

    S is identity-initialized: exact no-op at step 0, full gradient.
    """
    def __init__(self, base_embed, d_model, styles):
        super().__init__()
        self.emb = base_embed
        self.S = nn.ModuleDict({s: nn.Linear(d_model, d_model, bias=False)
                                for s in styles})
        for lin in self.S.values():
            nn.init.eye_(lin.weight)
        self.mode = "factual"

    def forward(self, input_ids):
        return self.S[self.mode](self.emb(input_ids))

    @property
    def weight(self):                      # some HF code paths probe this
        return self.emb.weight


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
        # Dropping its blocks frees ~1.2GB. Shared embeddings stay on t5.shared
        # and lm_head is a separate tensor in this checkpoint, so wrapping the
        # decoder's embedding lookup does not disturb the output head.
        self.t5.encoder.block = nn.ModuleList()
        for p in self.t5.encoder.parameters():
            p.requires_grad = False

        self.style_embed = StyleEmbedding(
            self.t5.decoder.embed_tokens, self.d_model, self.styles)
        self.t5.decoder.embed_tokens = self.style_embed
        self._adapters = [self.style_embed]

        if gradient_checkpointing:
            self.t5.gradient_checkpointing_enable()
            self.t5.config.use_cache = False

    def set_mode(self, mode):
        if mode not in self.styles:
            raise ValueError(f"Unknown mode: {mode}. Available: {self.styles}")
        for a in self._adapters:
            a.mode = mode

    def _memory(self, raw_features, batch_size, device, dtype):
        """raw_features=None -> ZERO memory. T5Attention has bias=False
        everywhere, so K=V=0 makes cross-attention output exactly zero and the
        residual stream passes through unchanged. The decoder is then a pure
        LM, which is StyleNet's second task."""
        if raw_features is None:
            mem = self.null_memory.expand(batch_size, 1, -1).to(dtype)
        else:
            mem = self.projector(raw_features.to(dtype))
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
