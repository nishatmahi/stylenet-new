import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, ViTModel


class EncoderViT(nn.Module):
    """Frozen ViT-base. Returns full patch sequence (197 tokens, 768-dim)."""
    def __init__(self, decoder_hidden_size: int = 768):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()
        self.A = nn.Linear(self.vit.config.hidden_size, decoder_hidden_size)
        self.norm = nn.LayerNorm(decoder_hidden_size)

    @torch.no_grad()
    def _vit_forward(self, images):
        return self.vit(images).last_hidden_state

    def forward(self, images):
        patch_features = self._vit_forward(images)
        projected = self.A(patch_features)
        return self.norm(projected)


class ZeroGatedCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, visual_kv):
        attn_out, _ = self.cross_attn(
            query=hidden_states, key=visual_kv, value=visual_kv, need_weights=False
        )
        return self.norm(hidden_states + self.gate * attn_out)


class StyleInjection(nn.Module):
    def __init__(self, hidden_size: int, style_dim: int):
        super().__init__()
        self.project = nn.Linear(hidden_size + style_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, style_vec):
        style_expanded = style_vec.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        combined = torch.cat([hidden_states, style_expanded], dim=-1)
        delta = self.project(combined)
        return self.norm(hidden_states + self.gate * delta)


class TransformerFactoredDecoder(nn.Module):
    VISUAL_SCALE = {"factual": 1.0, "romantic": 0.5}
    STYLES = ("factual", "romantic")

    def __init__(
        self,
        gpt2_model,           # pass an already-constructed GPT2LMHeadModel
        pad_token_id,
        bos_token_id,
        eos_token_id,
        style_dim: int = 256,
        num_cross_heads: int = 8,
        num_unfrozen_layers: int = 2,
        middle_injection_layer_idx: int = None,
    ):
        super().__init__()
        self.gpt2 = gpt2_model
        self.hidden_size = self.gpt2.config.n_embd
        self.num_unfrozen_layers = num_unfrozen_layers

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        for param in self.gpt2.parameters():
            param.requires_grad = False

        transformer = self.gpt2.transformer
        num_blocks = len(transformer.h)
        assert num_unfrozen_layers < num_blocks

        for block in transformer.h[-num_unfrozen_layers:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in transformer.ln_f.parameters():
            param.requires_grad = True

        frozen_cutoff = num_blocks - self.num_unfrozen_layers
        self.frozen_cutoff = frozen_cutoff

        # Middle injection must be >= frozen_cutoff or gradient gets severed
        # by the frozen blocks' .detach() that follow it. Verified via
        # test_gradient_flow.py.
        self.middle_injection_layer_idx = (
            middle_injection_layer_idx if middle_injection_layer_idx is not None else frozen_cutoff
        )
        assert self.middle_injection_layer_idx >= frozen_cutoff, (
            f"middle_injection_layer_idx ({self.middle_injection_layer_idx}) must be >= "
            f"frozen_cutoff ({frozen_cutoff})"
        )

        self.style_embed = nn.Embedding(len(self.STYLES), style_dim)
        self.style_to_idx = {s: i for i, s in enumerate(self.STYLES)}

        self.cross_attn_middle = ZeroGatedCrossAttention(self.hidden_size, num_cross_heads)
        self.cross_attn_late = ZeroGatedCrossAttention(self.hidden_size, num_cross_heads)

        # A true "start" injection (before block 0) was removed -- it is
        # structurally incompatible with the frozen-block no_grad()+detach()
        # memory-saving strategy: no_grad() severs the autograd graph for
        # ANY tensor passing through it. Since block 0 is always inside the
        # frozen region, an injection before block 0 can NEVER receive
        # gradient here, regardless of index. Verified empirically.
        self.style_injection_middle = StyleInjection(self.hidden_size, style_dim)
        self.style_injection_end = StyleInjection(self.hidden_size, style_dim)

        self.lm_head = self.gpt2.lm_head

    def trainable_parameters(self):
        """All trainable params: unfrozen GPT2 tail + every injection/style module."""
        return [p for _, p in self.trainable_parameters_named()]

    def shared_parameters(self):
        """Just the unfrozen GPT2 tail -- used to build the optimizer's shared param group."""
        return [p for p in self.gpt2.parameters() if p.requires_grad]

    def trainable_parameters_named(self):
        """Returns (name, param) pairs for every trainable parameter, for gradient auditing."""
        out = []
        for n, p in self.gpt2.named_parameters():
            if p.requires_grad:
                out.append((f"gpt2.{n}", p))
        for module_name in ["style_embed", "cross_attn_middle", "cross_attn_late",
                             "style_injection_middle", "style_injection_end"]:
            module = getattr(self, module_name)
            for n, p in module.named_parameters():
                out.append((f"{module_name}.{n}", p))
        return out

    def _build_extended_attention_mask(self, attention_mask, dtype, seq_len, device):
        """
        Builds a COMBINED causal + padding additive mask, shape (B, 1, T, T).

        CRITICAL: this replaces a version that only built a padding mask
        (B, 1, 1, T) with no causal component. HF's standard GPT2Model.forward()
        internally combines causal + padding masking before ever calling
        individual blocks -- our manual block-by-block loop bypassed that,
        so blocks received padding-only masking and could attend to FUTURE
        positions. Confirmed via direct test: perturbing a future token
        changed earlier positions' output by up to 0.39 before this fix;
        0.0 after. This was a real data leak, not a theoretical one.
        """
        causal_allowed = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        causal_allowed = causal_allowed.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        pad_allowed = attention_mask.bool().unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)

        combined_allowed = causal_allowed & pad_allowed  # (B, 1, T, T)

        additive_mask = torch.zeros(combined_allowed.shape, dtype=dtype, device=device)
        additive_mask.masked_fill_(~combined_allowed, torch.finfo(dtype).min)
        return additive_mask

    def _gpt2_backbone_forward(self, input_ids, attention_mask, style_vec=None,
                                visual_kv=None, use_style_gates=True):
        transformer = self.gpt2.transformer
        device = input_ids.device
        seq_len = input_ids.shape[1]

        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        inputs_embeds = transformer.wte(input_ids)
        position_embeds = transformer.wpe(position_ids)
        hidden_states = transformer.drop(inputs_embeds + position_embeds)

        ext_mask = self._build_extended_attention_mask(attention_mask, hidden_states.dtype, seq_len, device)

        num_blocks = len(transformer.h)
        frozen_cutoff = num_blocks - self.num_unfrozen_layers

        for i, block in enumerate(transformer.h):
            if i < frozen_cutoff:
                with torch.no_grad():
                    block_out = block(hidden_states, attention_mask=ext_mask)
                    hidden_states = block_out[0] if isinstance(block_out, tuple) else block_out
                hidden_states = hidden_states.detach()
            else:
                block_out = block(hidden_states, attention_mask=ext_mask)
                hidden_states = block_out[0] if isinstance(block_out, tuple) else block_out

            if use_style_gates and i == self.middle_injection_layer_idx:
                if visual_kv is not None:
                    hidden_states = self.cross_attn_middle(hidden_states, visual_kv)
                if style_vec is not None:
                    hidden_states = self.style_injection_middle(hidden_states, style_vec)

        hidden_states = transformer.ln_f(hidden_states)

        if use_style_gates:
            if visual_kv is not None:
                hidden_states = self.cross_attn_late(hidden_states, visual_kv)
            if style_vec is not None:
                hidden_states = self.style_injection_end(hidden_states, style_vec)

        return hidden_states

    def forward(self, captions, features=None, mode="factual", return_reference=False):
        if mode not in self.STYLES:
            sys.stderr.write("mode name wrong!\n")
            raise ValueError(f"Unknown mode: {mode}")

        input_ids = captions[:, :-1]
        attention_mask = (input_ids != self.pad_token_id).long()

        style_idx = torch.tensor([self.style_to_idx[mode]], device=captions.device)
        style_vec = self.style_embed(style_idx).expand(captions.size(0), -1)

        visual_kv = None
        if features is not None:
            scale = self.VISUAL_SCALE[mode]
            visual_kv = features * scale

        ref_logits = None
        if return_reference:
            with torch.no_grad():
                ref_hidden = self._gpt2_backbone_forward(
                    input_ids, attention_mask, style_vec=None, visual_kv=None, use_style_gates=False
                )
                ref_logits = self.lm_head(ref_hidden)

        hidden = self._gpt2_backbone_forward(
            input_ids, attention_mask, style_vec=style_vec, visual_kv=visual_kv
        )
        logits = self.lm_head(hidden)

        if return_reference:
            return logits, ref_logits
        return logits

    @torch.no_grad()
    def sample(self, feature, beam_size=5, max_len=30, mode="factual", repetition_penalty=1.3):
        device = next(self.parameters()).device
        start_id = self.bos_token_id
        end_id = self.eos_token_id

        style_idx = torch.tensor([self.style_to_idx[mode]], device=device)
        style_vec = self.style_embed(style_idx)

        visual_kv = None
        if feature is not None:
            visual_kv = feature * self.VISUAL_SCALE[mode]

        candidates = [[0.0, [start_id]]]

        for _ in range(max_len - 1):
            tmp_candidates = []
            end_flag = True

            for score, id_seq in candidates:
                if id_seq[-1] == end_id:
                    tmp_candidates.append([score, id_seq])
                    continue
                end_flag = False

                input_ids = torch.tensor([id_seq], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)

                hidden = self._gpt2_backbone_forward(
                    input_ids, attention_mask, style_vec=style_vec, visual_kv=visual_kv
                )
                logits = self.lm_head(hidden)
                next_token_logits = logits[0, -1, :].clone()

                if repetition_penalty != 1.0 and len(id_seq) > 1:
                    for prev_token_id in set(id_seq):
                        if next_token_logits[prev_token_id] < 0:
                            next_token_logits[prev_token_id] *= repetition_penalty
                        else:
                            next_token_logits[prev_token_id] /= repetition_penalty

                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                top_scores, top_indices = torch.topk(log_probs, beam_size)

                for score_val, wid in zip(top_scores, top_indices):
                    new_score = score + score_val.item()
                    new_id_seq = id_seq + [int(wid.item())]
                    tmp_candidates.append([new_score, new_id_seq])

            if end_flag:
                break

            candidates = sorted(tmp_candidates, key=lambda x: x[0] / len(x[1]), reverse=True)[:beam_size]

        return candidates[0][1]
