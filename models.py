import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, MT5ForConditionalGeneration

# --------- EncoderViT ---------
class EncoderViT(nn.Module):
    """CLS-token only, exactly matching original StyleNet encoder behavior.
    Frozen ViT, single trainable linear projection (equivalent to paper's A matrix)."""
    def __init__(self, decoder_hidden_size):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for p in self.vit.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(self.vit.config.hidden_size, decoder_hidden_size)

    def forward(self, images):
        outputs = self.vit(images)
        features = outputs.last_hidden_state[:, 0, :]   # CLS token
        features = self.proj(features)                    # [batch, decoder_hidden]
        return features.unsqueeze(1)                      # [batch, 1, decoder_hidden] for cross-attn

# --------- Factored FFN (transformer analog of Ux Sx Vx) ---------
class FactoredFFN(nn.Module):
    """
    Mirrors paper eq. (8): Wx = Ux Sx Vx
    - V: shared (paper's Vx)
    - S_<style>: style-specific (paper's Sx — S_F, S_R, S_H)
    - U: shared (paper's Ux) — zero-initialized so the adapter starts as a
      no-op residual and doesn't perturb pretrained mT5 activations early on.
    """
    def __init__(self, hidden_dim, factored_dim):
        super().__init__()
        self.V = nn.Linear(hidden_dim, factored_dim)
        self.S_factual = nn.Linear(factored_dim, factored_dim)
        self.S_romantic = nn.Linear(factored_dim, factored_dim)
        self.S_humorous = nn.Linear(factored_dim, factored_dim)
        self.U = nn.Linear(factored_dim, hidden_dim)

        # Zero-init U: adapter output starts at 0, grows gradually during training.
        # Prevents large-magnitude residual additions on top of pretrained weights.
        nn.init.zeros_(self.U.weight)
        nn.init.zeros_(self.U.bias)

    def forward(self, x, mode):
        h = self.V(x)
        style_map = {"factual": self.S_factual, "romantic": self.S_romantic, "humorous": self.S_humorous}
        if mode not in style_map:
            raise ValueError(f"Unknown mode: {mode}. Only 'factual', 'romantic', 'humorous' supported.")
        h = style_map[mode](h)
        return self.U(F.relu(h))

def _patch_cross_attn_zero_in_style_mode(block):
    """When mode != 'factual', skip cross-attention entirely — equivalent to
    features=None in the original FactoredLSTM (no visual info at all)."""
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
        adapter_out = block.factored_adapter(hidden_states, getattr(block, "current_mode", "factual"))
        return out + adapter_out

    block.layer[-1].forward = new_ffn_forward

def build_factored_mt5_decoder(vocab_size=None, factored_dim=512, pretrained_name='google/mt5-base'):
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
    for block in model.decoder.block:
        block.current_mode = mode

def get_trainable_param_groups(model):
    for p in model.parameters():
        p.requires_grad = False

    cap_params = []
    lang_params = []

    for block in model.decoder.block:
        adapter = block.factored_adapter
        for p in list(adapter.V.parameters()) + list(adapter.U.parameters()) + list(adapter.S_factual.parameters()):
            p.requires_grad = True
            cap_params.append(p)
        for p in adapter.S_romantic.parameters():
            p.requires_grad = True
            lang_params.append(p)
        for p in adapter.S_humorous.parameters():
            p.requires_grad = True
            lang_params.append(p)
        for p in block.layer[1].parameters():
            p.requires_grad = True
            cap_params.append(p)

    for p in model.shared.parameters():
        p.requires_grad = True
        cap_params.append(p)

    return cap_params, lang_params
