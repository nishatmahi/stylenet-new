import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, MT5Config, MT5ForConditionalGeneration

# --------- EncoderViT ---------
class EncoderViT(nn.Module):
    """Frozen ViT + single trainable linear projection.
    CLS token only — exactly matching original StyleNet encoder behavior."""
    def __init__(self, decoder_hidden_size):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for p in self.vit.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(self.vit.config.hidden_size, decoder_hidden_size)

    def forward(self, images):
        outputs = self.vit(images)
        features = outputs.last_hidden_state[:, 0, :]  # CLS token
        features = self.proj(features)                  # [batch, decoder_hidden]
        return features.unsqueeze(1)                    # [batch, 1, decoder_hidden]

# --------- Factored FFN ---------
class FactoredFFN(nn.Module):
    """
    Mirrors paper eq (8): Wx = Ux Sx Vx
    V, U: shared. S_<style>: style-specific.
    U zero-initialized so adapter starts as no-op residual.
    """
    def __init__(self, hidden_dim, factored_dim):
        super().__init__()
        self.V = nn.Linear(hidden_dim, factored_dim)
        self.S_factual  = nn.Linear(factored_dim, factored_dim)
        self.S_humorous = nn.Linear(factored_dim, factored_dim)
        self.U = nn.Linear(factored_dim, hidden_dim)

        nn.init.zeros_(self.U.weight)
        nn.init.zeros_(self.U.bias)

    def forward(self, x, mode):
        h = self.V(x)
        if mode == "factual":
            h = self.S_factual(h)
        elif mode == "humorous":
            h = self.S_humorous(h)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return self.U(F.relu(h))

def _patch_cross_attn_zero_in_style_mode(block):
    """Style modes: skip cross-attention entirely (no image).
    Equivalent to features=None in original FactoredLSTM."""
    original_forward = block.layer[1].forward

    def new_forward(hidden_states, *args, **kwargs):
        if getattr(block, "current_mode", "factual") == "factual":
            return original_forward(hidden_states, *args, **kwargs)
        else:
            return (hidden_states,)

    block.layer[1].forward = new_forward

def _patch_ffn_with_factored_adapter(block, hidden_dim, factored_dim):
    block.factored_adapter = FactoredFFN(hidden_dim, factored_dim)
    original_ffn_forward = block.layer[-1].forward

    def new_ffn_forward(hidden_states, *args, **kwargs):
        out = original_ffn_forward(hidden_states, *args, **kwargs)
        adapter_out = block.factored_adapter(
            hidden_states, getattr(block, "current_mode", "factual"))
        return out + adapter_out

    block.layer[-1].forward = new_ffn_forward

def build_factored_mt5_decoder(factored_dim=512, pretrained_name='google/mt5-base'):
    """
    FIX 1: Load ONLY the decoder half of mT5 — the mT5 encoder
    (~1.2GB) is dead weight since EncoderViT is our real encoder.
    We do this by loading the full model then immediately deleting
    the encoder and freeing that memory.
    """
    model = MT5ForConditionalGeneration.from_pretrained(pretrained_name)

    # Delete mT5 encoder — we never use it, EncoderViT replaces it
    del model.encoder
    torch.cuda.empty_cache()

    # FIX 2: Freeze ALL parameters to start — including model.shared
    # (250K x 768 embedding = 192M params = ~1.5GB optimizer state if unfrozen)
    for p in model.parameters():
        p.requires_grad = False

    # Patch decoder blocks with factored adapters + cross-attn zeroing
    hidden_dim = model.config.d_model
    for block in model.decoder.block:
        block.current_mode = "factual"
        _patch_ffn_with_factored_adapter(block, hidden_dim, factored_dim)
        _patch_cross_attn_zero_in_style_mode(block)

    return model

def set_mode(model, mode):
    for block in model.decoder.block:
        block.current_mode = mode

def get_trainable_param_groups(model):
    """
    FIX 3: Only train factored adapters + cross-attention projection
    for factual (cap_params), and style-specific S matrices for
    humorous (lang_params).

    Cross-attention: only the query projection (Wq) needs to learn
    to attend to ViT features — key/value come from ViT which is
    fixed. Much lighter than unfreezing all 12 full cross-attn blocks.
    model.shared and all other mT5 weights stay frozen.
    """
    cap_params  = []
    lang_params = []

    for block in model.decoder.block:
        adapter = block.factored_adapter

        # Shared V, U + factual S: trained on image+caption task
        for p in (list(adapter.V.parameters()) +
                  list(adapter.U.parameters()) +
                  list(adapter.S_factual.parameters())):
            p.requires_grad = True
            cap_params.append(p)

        # Humorous S: trained on text-only humorous task
        for p in adapter.S_humorous.parameters():
            p.requires_grad = True
            lang_params.append(p)

        # Cross-attention query projection only — learns to query ViT features
        # Key/value come from ViT (fixed), only query needs to adapt
        cross_attn = block.layer[1].layer  # T5LayerCrossAttention -> T5Attention
        if hasattr(cross_attn, 'EncDecAttention'):
            for p in cross_attn.EncDecAttention.q.parameters():
                p.requires_grad = True
                cap_params.append(p)

    return cap_params, lang_params
