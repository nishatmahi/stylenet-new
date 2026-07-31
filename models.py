"""
models.py — StyleNet Transformer model definitions.

Components:
  - FeatureProjection : lightweight nn.Linear for training (precomputed features)
  - EncoderViT        : full ViT + projection for inference / feature extraction
  - FactoredFFN       : style-switched adapter injected into mT5 decoder FFN
  - build_factored_mt5_decoder : patches mT5-base decoder with adapters
  - set_mode / get_trainable_param_groups : helpers
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, MT5ForConditionalGeneration


# --------- FeatureProjection (training — precomputed features) ---------
class FeatureProjection(nn.Module):
    """Projects pre-extracted ViT CLS features [768] → decoder dim [d_model].

    This tiny layer (~2 MB) replaces the full EncoderViT (~350 MB) during
    training.  ViT features are precomputed and loaded from disk, so the
    heavy ViT model never touches the GPU during training.
    """
    def __init__(self, vit_dim, decoder_dim):
        super().__init__()
        self.proj = nn.Linear(vit_dim, decoder_dim)

    def forward(self, raw_features):
        """
        Args:
            raw_features: [batch, vit_dim]  — pre-extracted ViT CLS tokens
        Returns:
            [batch, 1, decoder_dim]  — ready for mT5 cross-attention
        """
        return self.proj(raw_features).unsqueeze(1)


# --------- EncoderViT (inference & feature extraction) ---------
class EncoderViT(nn.Module):
    """Full encoder: frozen ViT + trainable projection.

    Used for:
      1) Feature extraction  (extract_raw → raw CLS without projection)
      2) Inference / sampling (forward → projected features)

    NOT loaded during training — use FeatureProjection instead.
    """
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
        return features.unsqueeze(1)                      # [batch, 1, decoder_hidden]

    @torch.no_grad()
    def extract_raw(self, images):
        """Extract raw ViT CLS features WITHOUT projection (for precomputation)."""
        outputs = self.vit(images)
        return outputs.last_hidden_state[:, 0, :]         # [batch, vit_dim=768]


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


def _patch_cross_attn_for_style_mode(block):
    """Control cross-attention per style mode.

    - factual:   full cross-attention (1.0 × visual signal)
    - romantic:  scaled cross-attention (0.5 × visual signal) — grounded but styled
    - humorous:  scaled cross-attention (0.5 × visual signal) — grounded but styled

    During STYLED TRAINING (text-only), dummy zero encoder outputs are passed,
    so cross-attention produces zeros regardless of scaling.

    During INFERENCE, real image features are passed for ALL modes.
    The 0.5 scaling ensures styled captions stay grounded to the image
    without being overwhelmed by visual signal (same idea as the LSTM's
    0.5 * visual_i trick).
    """
    original_forward = block.layer[1].forward

    def new_forward(hidden_states, *args, **kwargs):
        mode = getattr(block, "current_mode", "factual")
        result = original_forward(hidden_states, *args, **kwargs)

        if mode == "factual":
            # Full cross-attention output (unchanged)
            return result
        else:
            # Scale cross-attention residual by 0.5 for styled modes
            # result[0] is the output hidden_states (after cross-attn + residual)
            # We scale only the cross-attention contribution, not the full output.
            # Since the layer does: output = hidden_states + cross_attn(hidden_states),
            # we approximate: output = hidden_states + 0.5 * cross_attn(hidden_states)
            # Which is: hidden_states + 0.5 * (result[0] - hidden_states)
            scaled = hidden_states + 0.5 * (result[0] - hidden_states)
            return (scaled,) + result[1:]

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
        _patch_cross_attn_for_style_mode(block)

    # Delete the mT5 encoder — we use precomputed ViT features instead.
    # The encoder is ~290M params (~1.2 GB) sitting in VRAM doing nothing.
    # We always pass encoder_outputs=(features,) directly, so the internal
    # encoder is never called.
    del model.encoder
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[INFO] mT5 encoder deleted. Freed ~1.2 GB VRAM.")

    return model


def set_mode(model, mode):
    for block in model.decoder.block:
        block.current_mode = mode


def get_trainable_param_groups(model, cross_attn_blocks=4):
    """Set up trainable parameter groups.

    Args:
        model: The patched mT5 decoder model.
        cross_attn_blocks: Number of *last* decoder blocks whose cross-attention
            layers are unfrozen.  The original Factored LSTM used only 4 small
            F_* matrices for visual conditioning — unfreezing cross-attn in the
            last N blocks is the closest transformer analog without blowing up
            memory.  Set to 0 to freeze all cross-attn, or len(model.decoder.block)
            to unfreeze all (not recommended — OOM).
    """
    for p in model.parameters():
        p.requires_grad = False

    cap_params = []
    lang_params = []

    num_blocks = len(model.decoder.block)
    cross_attn_start = num_blocks - cross_attn_blocks  # e.g. 12-4 = block 8+

    for idx, block in enumerate(model.decoder.block):
        # Factored adapters (V, U, S_*) — always trainable in every block
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

        # Cross-attention — only unfreeze in the last N blocks
        if idx >= cross_attn_start:
            for p in block.layer[1].parameters():
                p.requires_grad = True
                cap_params.append(p)

    # model.shared (250K × 768 embedding) stays FROZEN.
    # Original LSTM had a tiny trainable embedding (vocab × 300), but mT5's
    # pretrained SentencePiece embeddings already cover Bengali well.
    # Unfreezing it adds ~1.5 GB of Adam optimizer state for no benefit.

    trainable = sum(p.numel() for p in cap_params + lang_params)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable: {trainable:,} / {total:,} params "
          f"({100*trainable/total:.1f}%) | cross-attn unfrozen in last {cross_attn_blocks} blocks")

    return cap_params, lang_params
