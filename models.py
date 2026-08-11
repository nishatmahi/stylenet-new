import sys
import torch
import torch.nn as nn
from transformers import ViTModel, T5ForConditionalGeneration


class EncoderViT(nn.Module):
    def __init__(self, emb_dim):
        super(EncoderViT, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for param in self.vit.parameters():
            param.requires_grad = False
        self.A = nn.Linear(self.vit.config.hidden_size, emb_dim)
        self.embed_norm = nn.LayerNorm(emb_dim)
        for param in self.A.parameters():
            param.requires_grad = True

    def extract_raw(self, images):
        with torch.no_grad():
            outputs = self.vit(images)
        return outputs.last_hidden_state

    def _project(self, raw_features):
        features = self.A(raw_features)
        features = self.embed_norm(features)
        return features

    def forward_from_cache(self, raw_features):
        return self._project(raw_features)

    def forward(self, images):
        raw_features = self.extract_raw(images)
        return self._project(raw_features)


class LoRALayer(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.down(x))


class BanglaT5StyleCaptioner(nn.Module):
    """
    T5 backbone is FROZEN — only LoRA adapters, visual gates, and
    EncoderViT's A/embed_norm are trainable. Fixes the OOM from full
    backbone fine-tuning on a 14.5GB GPU.
    """
    def __init__(self, t5_ckpt, tokenizer_len, style_rank=8,
                 styles=("factual", "romantic"), pad_token_id=0):
        super().__init__()
        self.styles = list(styles)
        self.pad_token_id = pad_token_id

        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_ckpt)

        original_vocab_size = self.t5.get_input_embeddings().weight.shape[0]
        self.t5.resize_token_embeddings(tokenizer_len)

        gap = original_vocab_size - tokenizer_len
        if gap > 500:
            print(
                f"[WARNING] tokenizer_len ({tokenizer_len}) is {gap} tokens "
                f"smaller than BanglaT5's embedding size ({original_vocab_size})."
            )

        # Freeze the entire T5 backbone.
        for param in self.t5.parameters():
            param.requires_grad = False

        self.t5_dim = self.t5.config.d_model
        self.n_decoder_layers = self.t5.config.num_decoder_layers

        self.style_adapters = nn.ModuleDict({
            style: nn.ModuleList([
                LoRALayer(self.t5_dim, style_rank) for _ in range(self.n_decoder_layers)
            ])
            for style in self.styles
        })

        init_gate = {"factual": 1.0, "romantic": 0.5}
        self.visual_gate = nn.ParameterDict({
            style: nn.Parameter(torch.tensor(init_gate.get(style, 1.0)))
            for style in self.styles
        })

        self._active_style = "factual"
        self._register_adapter_hooks()

    def _register_adapter_hooks(self):
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
        if features is not None:
            memory = self.visual_gate[mode] * features
            attn_mask = torch.ones(memory.shape[:2], dtype=torch.long, device=device)
        else:
            memory = torch.zeros(batch_size, 1, self.t5_dim, device=device)
            attn_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        return memory, attn_mask

    def forward(self, captions, features=None, mode="factual"):
        if mode not in self.styles:
            sys.stderr.write("mode name wrong!\n")
            raise ValueError(f"Unknown mode: {mode}. Only {self.styles} supported.")

        self._active_style = mode
        batch_size = captions.size(0)
        device = captions.device

        memory, attn_mask = self._build_encoder_memory(features, mode, batch_size, device)

        decoder_input_ids = captions[:, :-1]
        outputs = self.t5(
            inputs_embeds=memory,
            attention_mask=attn_mask,
            decoder_input_ids=decoder_input_ids,
        )
        return outputs.logits

    @torch.no_grad()
    def sample(self, feature, tokenizer, beam_size=5, max_len=30, mode="factual",
               repetition_penalty=1.3):
        self._active_style = mode
        device = feature.device
        batch_size = 1

        memory, attn_mask = self._build_encoder_memory(feature, mode, batch_size, device)

        start_id = tokenizer.bos_token_id
        end_id = tokenizer.eos_token_id

        candidates = [[0.0, [start_id]]]

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
