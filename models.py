import sys
import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel


class EncoderViT(nn.Module):
    """
    Frozen ViT-base encoder. Returns the FULL patch sequence (197 tokens:
    1 CLS + 196 patches), not just the pooled CLS vector, so the decoder
    can cross-attend spatially instead of relying on one global vector.
    """
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
        return self.vit(images).last_hidden_state  # (B, 197, 768)

    def forward(self, images):
        """
        images: (B, 3, 224, 224)
        returns: (B, 197, decoder_hidden_size)
        """
        patch_features = self._vit_forward(images)     # (B, 197, 768)
        projected = self.A(patch_features)               # (B, 197, decoder_hidden_size)
        return self.norm(projected)


class FactoredStyleAdapter(nn.Module):
    """
    Low-rank style bottleneck applied to decoder hidden states.
    One instance per style ("factual", "romantic"), matching the
    factored_dim concept from the original FactoredLSTM's S-matrices.
    """
    def __init__(self, hidden_size: int, factored_dim: int):
        super().__init__()
        self.V_style = nn.Linear(hidden_size, factored_dim, bias=False)
        self.U_style = nn.Linear(factored_dim, hidden_size, bias=False)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)

        nn.init.xavier_uniform_(self.V_style.weight, gain=0.1)
        nn.init.xavier_uniform_(self.U_style.weight, gain=0.1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (B, T, hidden_size)
        returns: (B, T, hidden_size)
        """
        z = self.act(self.V_style(hidden_states))   # (B, T, hidden_size) -> (B, T, factored_dim)
        delta = self.U_style(z)                        # (B, T, factored_dim) -> (B, T, hidden_size)
        return self.norm(hidden_states + delta)


class TransformerFactoredDecoder(nn.Module):
    """
    Drop-in replacement for FactoredLSTM, backed by flax-community/gpt2-bengali.

    GPT2 backbone is PERMANENTLY FROZEN (requires_grad=False on every param,
    forward pass wrapped in torch.no_grad() + .clone().detach()). Only
    cross_attn, cross_attn_norm, and style_adapters are trainable.

    Mode behavior preserved from the original FactoredLSTM:
      - features=None  -> no visual conditioning (text-only training path,
                           used for romantic-mode training)
      - features given -> cross-attention at a mode-dependent strength
                           (1.0 factual, 0.5 romantic — empirically-found
                           inference-time grounding fix, not a training-time
                           regularizer)
    """

    VISUAL_SCALE = {
        "factual": 1.0,
        "romantic": 0.5,
        # "humorous": 0.5,  # uncomment + add a style_adapters entry to re-enable
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

        # --- Ensure tokenizer has BOS/EOS/PAD, resize embeddings if needed ---
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

        # --- Freeze GPT2 entirely (done AFTER resize, so new embedding
        # rows inherit this frozen state too) ---
        for param in self.gpt2.parameters():
            param.requires_grad = False
        self.gpt2.eval()

        # --- Trainable visual cross-attention ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_cross_heads,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(self.hidden_size)

        # --- Trainable per-style bottlenecks ---
        self.style_adapters = nn.ModuleDict({
            "factual": FactoredStyleAdapter(self.hidden_size, factored_dim),
            "romantic": FactoredStyleAdapter(self.hidden_size, factored_dim),
        })

        self.lm_head = self.gpt2.lm_head  # frozen, tied to frozen embeddings

    def trainable_parameters(self):
        """Everything except the frozen GPT2 backbone."""
        params = []
        params += list(self.cross_attn.parameters())
        params += list(self.cross_attn_norm.parameters())
        params += list(self.style_adapters.parameters())
        return params

    def forward(self, captions, features=None, mode="factual"):
        """
        captions: (B, T) full BOS...EOS token sequence
        features: (B, 197, hidden_size) from EncoderViT, or None
        mode: "factual" | "romantic"

        returns: logits (B, T-1, vocab_size), aligned to predict captions[:, 1:]
        """
        if mode not in self.style_adapters:
            sys.stderr.write("mode name wrong!\n")
            raise ValueError(f"Unknown mode: {mode}. Only 'factual' and 'romantic' supported.")

        input_ids = captions[:, :-1]
        attention_mask = (input_ids != self.pad_token_id).long()

        # Break the computational graph at the GPT2 boundary: no_grad()
        # prevents autograd from building a graph through GPT2's 12 layers,
        # and .clone().detach() is a second explicit guarantee the tensor
        # handed downstream carries zero graph history back into GPT2.
        with torch.no_grad():
            gpt2_out = self.gpt2.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hidden = gpt2_out.last_hidden_state.clone().detach()  # (B, T-1, hidden_size)
        hidden.requires_grad_(False)

        if features is not None:
            scale = self.VISUAL_SCALE[mode]
            visual_kv = features * scale                        # (B, 197, hidden_size)
            attn_out, _ = self.cross_attn(
                query=hidden, key=visual_kv, value=visual_kv, need_weights=False
            )
            hidden = self.cross_attn_norm(hidden + attn_out)     # (B, T-1, hidden_size)

        hidden = self.style_adapters[mode](hidden)                # (B, T-1, hidden_size)
        logits = self.lm_head(hidden)                              # (B, T-1, vocab_size)
        return logits

    @torch.no_grad()
    def sample(self, feature, tokenizer, beam_size=5, max_len=30, mode="factual",
               repetition_penalty=1.3):
        """
        Beam search generation. feature: (1, 197, hidden_size) for one image.
        As in the original, ALL modes receive visual features during
        inference; mode-dependent scaling happens inside forward().
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

                input_ids = torch.tensor([id_seq], dtype=torch.long, device=device)
                logits = self.forward(
                    torch.cat([input_ids, input_ids[:, -1:]], dim=1),
                    features=feature,
                    mode=mode,
                )
                next_token_logits = logits[0, -1, :]

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
                key=lambda x: x[0] / len(x[1]),
                reverse=True,
            )[:beam_size]

        return candidates[0][1]
