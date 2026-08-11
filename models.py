import sys
import torch
import torch.nn as nn
from transformers import ViTModel, T5ForConditionalGeneration


# --------- EncoderViT (transformer version) ---------
class EncoderViT(nn.Module):
    """
    Frozen ViT backbone. Returns the FULL patch sequence (not just CLS),
    since the T5 decoder's cross-attention needs a sequence of encoder
    states to attend over, not a single pooled vector.
    """
    def __init__(self, emb_dim):
        super(EncoderViT, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for param in self.vit.parameters():
            param.requires_grad = False
        self.A = nn.Linear(self.vit.config.hidden_size, emb_dim)
        for param in self.A.parameters():
            param.requires_grad = True

    def forward(self, images):
        outputs = self.vit(images)
        features = outputs.last_hidden_state          # [B, N_patches+1, vit_hidden] (includes CLS)
        features = self.A(features)                   # [B, N_patches+1, emb_dim]
        return features


class LoRALayer(nn.Module):
    """Low-rank per-style delta applied on top of shared decoder output.
    Analogous to S_fi / S_ri style-specific transforms in the LSTM version."""
    def __init__(self, dim, rank):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)  # identity at init -> no effect until trained

    def forward(self, x):
        return x + self.up(self.down(x))


# --------- BanglaT5StyleCaptioner (replaces FactoredLSTM) ---------
class BanglaT5StyleCaptioner(nn.Module):
    """
    Full pretrained seq2seq (BanglaT5 encoder + decoder).
      - ViT features fed into the T5 encoder via `inputs_embeds` (bypasses
        the tokenizer/embedding lookup entirely for the image side)
      - Per-style LoRA adapters hooked onto every T5 decoder block
      - Per-style scalar gate controlling visual injection strength
        (mirrors the 1.0 factual / 0.5 romantic split)

    NOTE on tokenizer_len: T5-family checkpoints commonly pad their
    embedding matrix to a round number (e.g. 32128) beyond the tokenizer's
    real vocab size (e.g. 32100 + a few added tokens) for hardware
    efficiency. A small gap between tokenizer_len and the checkpoint's
    embedding size is normal, NOT a sign of a mismatched tokenizer. Only a
    large gap is worth investigating (see the warning below).
    """
    def __init__(self, t5_ckpt, tokenizer_len, style_rank=8,
                 styles=("factual", "romantic"), pad_token_id=0):
        super().__init__()
        self.styles = list(styles)
        self.pad_token_id = pad_token_id

        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_ckpt)

        original_vocab_size = self.t5.get_input_embeddings().weight.shape[0]
        self.t5.resize_token_embeddings(tokenizer_len)

        # Only warn on a large gap — small gaps (tens of tokens) are normal
        # T5 embedding padding, not a mismatched-tokenizer bug.
        gap = original_vocab_size - tokenizer_len
        if gap > 500:
            print(
                f"[WARNING] tokenizer_len ({tokenizer_len}) is {gap} tokens "
                f"smaller than BanglaT5's embedding size ({original_vocab_size}). "
                f"Small gaps (under a few hundred) are normal T5 padding. A gap "
                f"this large may indicate a mismatched tokenizer — verify it was "
                f"loaded via T5Tokenizer.from_pretrained('{t5_ckpt}') plus "
                f"add_tokens() only."
            )

        self.t5_dim = self.t5.config.d_model
        self.n_decoder_layers = self.t5.config.num_decoder_layers

        # Per-style LoRA adapters, one per decoder layer (the S_fi/S_ri set)
        self.style_adapters = nn.ModuleDict({
            style: nn.ModuleList([
                LoRALayer(self.t5_dim, style_rank) for _ in range(self.n_decoder_layers)
            ])
            for style in self.styles
        })

        # Per-style visual gate (1.0 factual / 0.5 romantic scaling)
        init_gate = {"factual": 1.0, "romantic": 0.5}
        self.visual_gate = nn.ParameterDict({
            style: nn.Parameter(torch.tensor(init_gate.get(style, 1.0)))
            for style in self.styles
        })

        self._active_style = "factual"
        self._register_adapter_hooks()

    def _register_adapter_hooks(self):
        """Applies the active style's LoRA delta to each decoder block's
        output, without modifying T5's internals."""
        def make_hook(layer_idx):
            def hook(module, inputs, output):
                hidden_states = output[0]
                adapter = self.style_adapters[self._active_style][layer_idx]
                hidden_states = adapter(hidden_states)
                return (hidden_states,) + output[1:]
            return hook

        for i, block in enumerate(self.t5.decoder.block):
            block.register_forward_hook(make_hook(i))

    def _build_encoder_memory(self, features, mode, batch_size, device):
        """features: [B, N, t5_dim] from EncoderViT, or None for text-only
        (romantic) training — mirrors the `features=None -> zeros` branch."""
        if features is not None:
            memory = self.visual_gate[mode] * features
            attn_mask = torch.ones(memory.shape[:2], dtype=torch.long, device=device)
        else:
            memory = torch.zeros(batch_size, 1, self.t5_dim, device=device)
            attn_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        return memory, attn_mask

    def forward(self, captions, features=None, mode="factual"):
        """
        Args:
            captions: [B, T] token ids, BOS...EOS
            features: [B, N, emb_dim] ViT sequence from EncoderViT, or None
            mode: "factual" or "romantic"

        Returns logits of shape [B, T-1, vocab] — directly comparable to
        captions[:, 1:], same convention as the original LSTM version.
        """
        if mode not in self.styles:
            sys.stderr.write("mode name wrong!\n")
            raise ValueError(f"Unknown mode: {mode}. Only {self.styles} supported.")

        self._active_style = mode
        batch_size = captions.size(0)
        device = captions.device

        memory, attn_mask = self._build_encoder_memory(features, mode, batch_size, device)

        decoder_input_ids = captions[:, :-1]   # teacher forcing input (starts at BOS)
        outputs = self.t5(
            inputs_embeds=memory,
            attention_mask=attn_mask,
            decoder_input_ids=decoder_input_ids,
        )
        return outputs.logits   # [B, T-1, vocab]

    @torch.no_grad()
    def sample(self, feature, tokenizer, beam_size=5, max_len=30, mode="factual",
               repetition_penalty=1.3):
        """
        Beam search generation. `feature` is a single image's ViT sequence
        [1, N, emb_dim]. Same candidate scoring/pruning logic as the
        original, with decoder_input_ids growing each step instead of
        (h_t, c_t) recurrence.
        """
        self._active_style = mode
        device = feature.device
        batch_size = 1

        memory, attn_mask = self._build_encoder_memory(feature, mode, batch_size, device)

        start_id = tokenizer.bos_token_id
        end_id = tokenizer.eos_token_id

        candidates = [[0.0, [start_id]]]   # [score, id_seq]

        for _ in range(max_len - 1):
            tmp_candidates = []
            end_flag = True

            for score, id_seq in candidates:
                if id_seq[-1] == end_id:
                    tmp_candidates.append([score, id_seq])
                    continue

                end_flag = False
                decoder_input_ids = torch.tensor([id_seq], dtype=torch.long, device=device)

                outputs = self.t5(
                    inputs_embeds=memory,
                    attention_mask=attn_mask,
                    decoder_input_ids=decoder_input_ids,
                )
                logits = outputs.logits[0, -1, :]

                if repetition_penalty != 1.0 and len(id_seq) > 1:
                    for prev_token_id in set(id_seq):
                        if logits[prev_token_id] < 0:
                            logits[prev_token_id] *= repetition_penalty
                        else:
                            logits[prev_token_id] /= repetition_penalty

                log_probs = torch.log_softmax(logits, dim=-1)
                top_log_probs, top_ids = torch.sort(log_probs, descending=True)
                top_log_probs = top_log_probs[:beam_size]
                top_ids = top_ids[:beam_size]

                for score_val, wid in zip(top_log_probs, top_ids):
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
