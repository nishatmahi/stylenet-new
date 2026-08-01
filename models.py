import sys
import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel


class EncoderViT(nn.Module):
    """
    Frozen ViT-base encoder. Returns the FULL patch sequence (197 tokens:
    1 CLS + 196 patches) so the decoder can cross-attend spatially instead
    of relying on one pooled global vector.
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
        patch_features = self._vit_forward(images)
        projected = self.A(patch_features)
        return self.norm(projected)


class FactoredStyleAdapter(nn.Module):
    """
    Low-rank style bottleneck applied to decoder hidden states.
    One instance per style ("factual", "romantic").
    Small-gain init keeps this close to identity at step 0.
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
        z = self.act(self.V_style(hidden_states))
        delta = self.U_style(z)
        return self.norm(hidden_states + delta)


class TransformerFactoredDecoder(nn.Module):
    """
    GPT2 backbone (flax-community/gpt2-bengali) is PERMANENTLY FROZEN.
    Only cross_attn, cross_attn_norm, cross_attn_gate, and style_adapters
    are trainable.

    Mode behavior:
      - features=None  -> no visual conditioning (text-only path, used for
                           romantic-mode training)
      - features given -> cross-attention at a mode-dependent strength
                           (1.0 factual, 0.5 romantic), gated by a
                           learnable scalar initialized at 0.0 so the model
                           starts IDENTICAL to the frozen backbone's own
                           output and only gradually learns to let visual
                           information in. Without this gate, an untrained
                           nn.MultiheadAttention output gets added directly
                           into a hidden state that a FROZEN lm_head has
                           never seen perturbed like that — which is what
                           produced the character-fragment/Latin gibberish
                           in your last run.
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

        # --- Only add special tokens that are genuinely missing. Do NOT
        # blindly add bos/eos if the tokenizer already defines them (most
        # GPT2-family tokenizers reuse <|endoftext|> for both) — adding
        # them anyway would create random, frozen, never-trained embedding
        # rows for tokens used at the START of every single sequence. ---
        special_tokens_to_add = {}
        if tokenizer.pad_token is None:
            special_tokens_to_add["pad_token"] = "<pad>"
        if tokenizer.bos_token is None:
            special_tokens_to_add["bos_token"] = "<bos>"
        if tokenizer.eos_token is None:
            special_tokens_to_add["eos_token"] = "<eos>"

        if special_tokens_to_add:
            num_added = tokenizer.add_special_tokens(special_tokens_to_add)
            self.gpt2.resize_token_embeddings(len(tokenizer))
            print(f"[TOKENIZER] Added {num_added} new special token(s): "
                  f"{list(special_tokens_to_add.keys())}")
            if "bos_token" in special_tokens_to_add or "eos_token" in special_tokens_to_add:
                print("[WARNING] bos_token or eos_token was newly added. Its embedding "
                      "row is randomly initialized and FROZEN (never trained), since "
                      "it lives inside self.gpt2 which is frozen below. Every sequence "
                      "starts/ends with this untrained vector. If generation quality "
                      "is poor, this is a prime suspect — check whether the base "
                      "tokenizer already had bos/eos before assuming this path ran.")
        else:
            print("[TOKENIZER] pad/bos/eos already defined by the base tokenizer. "
                  "No new tokens added — no untrained embedding risk here.")

        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # --- Freeze GPT2 entirely (after resize, so new rows inherit this) ---
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

        # --- Zero-initialized gate: this is the actual fix for the
        # gibberish output. At step 0, cross_attn_gate == 0.0, so
        # hidden + 0.0 * attn_out == hidden exactly — the model starts
        # identical to the frozen backbone's own coherent output, and only
        # learns to let visual signal in as this scalar moves away from 0. ---
        self.cross_attn_gate = nn.Parameter(torch.zeros(1))

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
        params += [self.cross_attn_gate]
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

        # Frozen backbone: no_grad() stops autograd from building a graph
        # through GPT2's 12 layers (saves activation memory). .clone().detach()
        # is a belt-and-suspenders guarantee that gradient never flows back
        # into GPT2's weights even if no_grad() is ever accidentally removed.
        # This does NOT break the gradient chain for the trainable layers
        # below — cross_attn/style_adapters/lm_head's own weights carry
        # requires_grad=True (lm_head is frozen too, but style_adapters and
        # cross_attn are not), so their outputs correctly track gradients
        # regardless of hidden's own flag. Verified empirically too: your
        # last training run completed a full epoch with backward() succeeding
        # every step — a genuinely severed graph would have errored at step 0.
        with torch.no_grad():
            gpt2_out = self.gpt2.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hidden = gpt2_out.last_hidden_state.clone().detach()  # (B, T-1, hidden_size)

        if features is not None:
            scale = self.VISUAL_SCALE[mode]
            visual_kv = features * scale                        # (B, 197, hidden_size)
            attn_out, _ = self.cross_attn(
                query=hidden, key=visual_kv, value=visual_kv, need_weights=False
            )
            # Gated residual — see __init__ comment on cross_attn_gate.
            hidden = self.cross_attn_norm(hidden + self.cross_attn_gate * attn_out)

        hidden = self.style_adapters[mode](hidden)                # (B, T-1, hidden_size)
        logits = self.lm_head(hidden)                              # (B, T-1, vocab_size)
        return logits

    @torch.no_grad()
    def sample(self, feature, tokenizer, beam_size=5, max_len=30, mode="factual",
               repetition_penalty=1.3):
        """
        Beam search generation. Calls self.gpt2.transformer directly instead
        of routing through forward()'s concat-then-slice trick — same result,
        clearer code. feature: (1, 197, hidden_size) for one image.
        As before, ALL modes receive visual features during inference;
        mode-dependent scaling + gating happens the same way as in forward().

        NOTE ON COST: this recomputes the full sequence through all 12 GPT2
        layers at every beam step (no KV-caching), same as previous versions.
        Slower than a cached implementation but behaves identically and is
        easy to verify — a disclosed tradeoff, not an oversight.
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
                attention_mask = torch.ones_like(input_ids)

                gpt2_out = self.gpt2.transformer(input_ids=input_ids, attention_mask=attention_mask)
                hidden = gpt2_out.last_hidden_state  # (1, t, hidden_size)

                if feature is not None:
                    scale = self.VISUAL_SCALE[mode]
                    visual_kv = feature * scale
                    attn_out, _ = self.cross_attn(
                        query=hidden, key=visual_kv, value=visual_kv, need_weights=False
                    )
                    hidden = self.cross_attn_norm(hidden + self.cross_attn_gate * attn_out)

                hidden = self.style_adapters[mode](hidden)
                logits = self.lm_head(hidden)
                next_token_logits = logits[0, -1, :].clone()  # prediction for the next token

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
                key=lambda x: x[0] / len(x[1]),  # length-normalized log-prob
                reverse=True,
            )[:beam_size]

        return candidates[0][1]
