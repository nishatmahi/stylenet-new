import sys
import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel


# --------- EncoderViT (Transformer version) ---------
class EncoderViT(nn.Module):
    """
    Frozen ViT-base encoder. Unlike the original, returns the FULL patch
    sequence (197 tokens: 1 CLS + 196 patches), not just the pooled CLS
    vector, so the decoder can cross-attend spatially instead of relying
    on one global vector.
    """
    def __init__(self, decoder_hidden_size: int = 768):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()

        # Projects ViT's 768-dim space into the decoder's hidden space.
        # Sizes happen to match (768 == 768) but kept trainable and explicit
        # in case you ever swap either backbone.
        self.A = nn.Linear(self.vit.config.hidden_size, decoder_hidden_size)
        self.norm = nn.LayerNorm(decoder_hidden_size)

    @torch.no_grad()
    def _vit_forward(self, images):
        return self.vit(images).last_hidden_state  # (B, 197, 768), no grad — frozen

    def forward(self, images):
        """
        images: (B, 3, 224, 224)
        returns: (B, 197, decoder_hidden_size) — patch-level visual features
        """
        patch_features = self._vit_forward(images)   # (B, 197, 768)
        projected = self.A(patch_features)            # (B, 197, decoder_hidden_size)
        return self.norm(projected)


# --------- FactoredStyleAdapter ---------
class FactoredStyleAdapter(nn.Module):
    """
    Replaces your per-gate S_fi/S_ff/S_fo/S_fc (or S_ri/S_rf/S_ro/S_rc)
    matrices with a single low-rank bottleneck on the Transformer hidden
    state, since there's one hidden stream here instead of four LSTM gates.
    One instance per style, same "factored_dim" bottleneck concept as your
    original code.
    """
    def __init__(self, hidden_size: int, factored_dim: int):
        super().__init__()
        self.V_style = nn.Linear(hidden_size, factored_dim, bias=False)
        self.U_style = nn.Linear(factored_dim, hidden_size, bias=False)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)

        # Small init so this starts close to identity (residual-friendly),
        # matching the fact that your original S matrices were also just
        # linear layers with default init sitting on top of a working LSTM.
        nn.init.xavier_uniform_(self.V_style.weight, gain=0.1)
        nn.init.xavier_uniform_(self.U_style.weight, gain=0.1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        z = self.act(self.V_style(hidden_states))   # (B, T, hidden) -> (B, T, factored_dim)
        delta = self.U_style(z)                       # (B, T, factored_dim) -> (B, T, hidden)
        return self.norm(hidden_states + delta)


# --------- TransformerFactoredDecoder ---------
class TransformerFactoredDecoder(nn.Module):
    """
    Drop-in replacement for FactoredLSTM, backed by flax-community/gpt2-bengali.
    Preserves your exact mode-dependent behavior:
      - features=None  -> no visual conditioning at all (text-only training path)
      - features given -> cross-attention at a mode-dependent strength
        (1.0 for factual, 0.5 for romantic — your discovered grounding fix)
    """

    VISUAL_SCALE = {
        "factual": 1.0,
        "romantic": 0.5,
        # "humorous": 0.5,   # uncomment when you re-enable humorous mode
    }

    def __init__(
        self,
        tokenizer,
        gpt2_name: str = "flax-community/gpt2-bengali",
        factored_dim: int = 512,
        num_cross_heads: int = 8,
    ):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name)
        self.hidden_size = self.gpt2.config.n_embd  # 768

        # --- Ensure the tokenizer has BOS/EOS/PAD, resize embeddings if needed ---
        added_tokens = 0
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            added_tokens += 1
        if tokenizer.bos_token is None:
            tokenizer.add_special_tokens({"bos_token": "<bos>"})
            added_tokens += 1
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "<eos>"})
            added_tokens += 1
        if added_tokens > 0:
            self.gpt2.resize_token_embeddings(len(tokenizer))

        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # Visual cross-attention: decoder hidden states attend over ViT patches
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_cross_heads,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(self.hidden_size)

        # One style bottleneck per mode
        self.style_adapters = nn.ModuleDict({
            "factual": FactoredStyleAdapter(self.hidden_size, factored_dim),
            "romantic": FactoredStyleAdapter(self.hidden_size, factored_dim),
        })

        self.lm_head = self.gpt2.lm_head  # tied to token embeddings

    def forward(self, captions, features=None, mode="factual"):
        """
        captions: (B, T) full BOS...EOS token sequence
        features: (B, 197, hidden_size) from EncoderViT, or None for text-only
        mode: "factual" | "romantic"

        returns: logits (B, T-1, vocab_size), aligned to predict captions[:, 1:]
        """
        if mode not in self.style_adapters:
            sys.stderr.write("mode name wrong!\n")
            raise ValueError(f"Unknown mode: {mode}. Only 'factual' and 'romantic' supported.")

        input_ids = captions[:, :-1]                          # (B, T-1) teacher-forced input
        attention_mask = (input_ids != self.pad_token_id).long()

        gpt2_out = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = gpt2_out.last_hidden_state  # (B, T-1, hidden_size)

        # Visual conditioning — only if features were actually provided.
        # This is what preserves your exact train/inference asymmetry:
        # romantic training calls forward with features=None, so this
        # branch is skipped entirely during styled-mode training.
        if features is not None:
            scale = self.VISUAL_SCALE[mode]
            visual_kv = features * scale                       # (B, 197, hidden_size)
            attn_out, _ = self.cross_attn(
                query=hidden, key=visual_kv, value=visual_kv, need_weights=False
            )
            hidden = self.cross_attn_norm(hidden + attn_out)   # (B, T-1, hidden_size)

        hidden = self.style_adapters[mode](hidden)              # (B, T-1, hidden_size)
        logits = self.lm_head(hidden)                            # (B, T-1, vocab_size)
        return logits

    @torch.no_grad()
    def sample(self, feature, tokenizer, beam_size=5, max_len=30, mode="factual",
               repetition_penalty=1.3):
        """
        Beam search generation. Same accumulation structure as your original
        (score, sequence, normalized-length sorting) — just driven by GPT2's
        parallel hidden states instead of an LSTM's (h_t, c_t) recurrence.

        feature: (1, 197, hidden_size) from EncoderViT for a single image.
        NOTE: as in your original, ALL modes receive visual features during
        inference; scaling by VISUAL_SCALE[mode] happens inside forward().
        """
        device = feature.device
        start_id = self.bos_token_id
        end_id = self.eos_token_id

        candidates = [[0.0, [start_id]]]

        for _ in range(max_len - 1):
            tmp_candidates = []
            end_flag = True

            for score, id_seq in candidates:
                if id_seq[-1] == end_id:
                    tmp_candidates.append([score, id_seq])
                    continue
                end_flag = False

                input_ids = torch.tensor([id_seq], dtype=torch.long, device=device)  # (1, t)
                logits = self.forward(
                    torch.cat([input_ids, input_ids[:, -1:]], dim=1),  # pad 1 extra so forward's
                    features=feature,                                    # internal [:, :-1] slice keeps full seq
                    mode=mode,
                )
                next_token_logits = logits[0, -1, :]  # (vocab_size,) — prediction for next token

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

            candidates = sorted(
                tmp_candidates,
                key=lambda x: x[0] / len(x[1]),  # normalized log-prob, same as original
                reverse=True,
            )[:beam_size]

        return candidates[0][1]
