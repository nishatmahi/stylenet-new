import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, MT5ForConditionalGeneration

# --------- EncoderViT ---------
class EncoderViT(nn.Module):
    """Encodes image into a patch-sequence for cross-attention.
    (paper fed a single CLS/pooled vector at LSTM step 0; here we keep the
    full patch sequence since transformer cross-attention needs a sequence
    to attend over, not one vector.)"""
    def __init__(self, decoder_hidden_size):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for p in self.vit.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(self.vit.config.hidden_size, decoder_hidden_size)  # A matrix equivalent

    def forward(self, images):
        out = self.vit(images).last_hidden_state       # [batch, num_patches+1, 768]
        return self.proj(out)                            # [batch, num_patches+1, decoder_hidden]

# --------- Factored FFN (transformer analog of Ux Sx Vx) ---------
class FactoredFFN(nn.Module):
    """
    Mirrors paper eq. (8): Wx = Ux Sx Vx
    - V: shared (paper's Vx)
    - S_<style>: style-specific (paper's Sx — S_F, S_R, S_H)
    - U: shared (paper's Ux)
    Applied as a residual addition on top of mT5's pretrained FFN output,
    so pretrained language knowledge isn't destroyed.
    """
    def __init__(self, hidden_dim, factored_dim):
        super().__init__()
        self.V = nn.Linear(hidden_dim, factored_dim)      # shared
        self.S_factual = nn.Linear(factored_dim, factored_dim)   # style-specific
        self.S_romantic = nn.Linear(factored_dim, factored_dim)  # style-specific
        self.S_humorous = nn.Linear(factored_dim, factored_dim)  # style-specific
        self.U = nn.Linear(factored_dim, hidden_dim)      # shared

    def forward(self, x, mode):
        h = self.V(x)
        style_map = {"factual": self.S_factual, "romantic": self.S_romantic, "humorous": self.S_humorous}
        if mode not in style_map:
            raise ValueError(f"Unknown mode: {mode}. Only 'factual', 'romantic', 'humorous' supported.")
        h = style_map[mode](h)
        return self.U(F.relu(h))

def _patch_cross_attn_zero_in_style_mode(block):
    """When mode != 'factual', skip cross-attention entirely — equivalent to
    your original code's features=None path (no visual info at all)."""
    original_forward = block.layer[1].forward

    def new_forward(hidden_states, *args, **kwargs):
        if getattr(block, "current_mode", "factual") == "factual":
            return original_forward(hidden_states, *args, **kwargs)
        else:
            return (hidden_states,)  # pass through unchanged, no cross-attn contribution

    block.layer[1].forward = new_forward

def _patch_ffn_with_factored_adapter(block, hidden_dim, factored_dim):
    block.factored_adapter = FactoredFFN(hidden_dim, factored_dim)
    original_ffn_forward = block.layer[-1].forward

    def new_ffn_forward(hidden_states, *args, **kwargs):
        out = original_ffn_forward(hidden_states, *args, **kwargs)
        adapter_out = block.factored_adapter(hidden_states, getattr(block, "current_mode", "factual"))
        return out + adapter_out

    block.layer[-1].forward = new_ffn_forward

def build_factored_mt5_decoder(vocab_size=None, factored_dim=512, pretrained_name='google/mt5-base'):
    """
    Builds the StyleNet decoder equivalent: pretrained mT5 + factored adapters
    patched into every decoder block, with cross-attention disabled for
    non-factual (style) modes.
    """
    model = MT5ForConditionalGeneration.from_pretrained(pretrained_name)
    if vocab_size is not None and vocab_size != model.config.vocab_size:
        model.resize_token_embeddings(vocab_size)

    hidden_dim = model.config.d_model
    for block in model.decoder.block:
        block.current_mode = "factual"
        _patch_ffn_with_factored_adapter(block, hidden_dim, factored_dim)
        _patch_cross_attn_zero_in_style_mode(block)

    return model

def set_mode(model, mode):
    """Call before each forward pass — sets which style branch is active,
    same role as the `mode` argument in your original FactoredLSTM.forward_step"""
    for block in model.decoder.block:
        block.current_mode = mode

def get_trainable_param_groups(model):
    """
    Splits parameters into cap_params (factual/image-grounding) and
    lang_params (style-specific only), mirroring your original train.py split.
    Everything else in mT5 is frozen.
    """
    for p in model.parameters():
        p.requires_grad = False

    cap_params = []
    lang_params = []

    for block in model.decoder.block:
        adapter = block.factored_adapter
        # shared V, U + factual S: trained during factual (image) task
        for p in list(adapter.V.parameters()) + list(adapter.U.parameters()) + list(adapter.S_factual.parameters()):
            p.requires_grad = True
            cap_params.append(p)
        # style-specific S: trained only during their respective style task
        for p in adapter.S_romantic.parameters():
            p.requires_grad = True
            lang_params.append(p)
        for p in adapter.S_humorous.parameters():
            p.requires_grad = True
            lang_params.append(p)
        # cross-attention: only meaningful in factual mode, part of cap_params
        for p in block.layer[1].parameters():
            p.requires_grad = True
            cap_params.append(p)

    # embeddings: shared, needed if vocab was resized or fine-tuning language
    for p in model.shared.parameters():
        p.requires_grad = True
        cap_params.append(p)

    return cap_params, lang_params
