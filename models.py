import sys
import torch
import torch.nn as nn
from transformers import ViTModel
import torch.nn.functional as F

# --------- EncoderViT ---------
# The ViT is FROZEN (requires_grad=False). Only self.A is trainable and is
# part of cap_params, so it updates during factual training.
#
# For the feature cache we split the encoder into two stages:
#   vit_features(images) -> frozen 768-d CLS  (identical every epoch -> CACHEABLE)
#   project(vit_feats)   -> trainable A: 768 -> emb_dim  (updates -> stays LIVE)
# forward() is kept for backward compatibility (validation / any old call).
class EncoderViT(nn.Module):
    def __init__(self, emb_dim):
        super(EncoderViT, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for param in self.vit.parameters():
            param.requires_grad = False
        self.A = nn.Linear(self.vit.config.hidden_size, emb_dim)
        for param in self.A.parameters():
            param.requires_grad = True

    def vit_features(self, images):
        # FROZEN part only. No grad needed, this is what we cache.
        with torch.no_grad():
            outputs = self.vit(images)
            return outputs.last_hidden_state[:, 0, :]  # [B, 768] CLS token

    def project(self, vit_feats):
        # TRAINABLE part. Runs live in the training loop on top of cached vectors.
        return self.A(vit_feats)

    def forward(self, images):
        # Full path (used by validation and any legacy call).
        vit_feats = self.vit_features(images)
        return self.A(vit_feats)

# --------- FactoredGRU ---------
#
# MAPPING FROM LSTM -> GRU:
#   i (input gate)      -> z (update gate)
#   f (forget gate)     -> r (reset gate)
#   o (output gate)     -> REMOVED
#   c (cell candidate)  -> n (candidate hidden state)
# LSTM state: h_t AND c_t     GRU state: h_t ONLY
#
class FactoredGRU(nn.Module):
    def __init__(self, emb_dim, hidden_dim, factored_dim, vocab_size):
        super(FactoredGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        self.B = nn.Embedding(vocab_size, emb_dim)

        # ---- z (update gate) ----
        self.U_z = nn.Linear(factored_dim, hidden_dim)
        self.S_fz = nn.Linear(factored_dim, factored_dim)
        self.V_z = nn.Linear(emb_dim, factored_dim)
        self.W_z = nn.Linear(hidden_dim, hidden_dim)

        # ---- r (reset gate) ----
        self.U_r = nn.Linear(factored_dim, hidden_dim)
        self.S_fr = nn.Linear(factored_dim, factored_dim)
        self.V_r = nn.Linear(emb_dim, factored_dim)
        self.W_r = nn.Linear(hidden_dim, hidden_dim)

        # ---- n (candidate) ----
        self.U_n = nn.Linear(factored_dim, hidden_dim)
        self.S_fn = nn.Linear(factored_dim, factored_dim)
        self.V_n = nn.Linear(emb_dim, factored_dim)
        self.W_n = nn.Linear(hidden_dim, hidden_dim)

        # ---- Feature-to-gate (visual conditioning), 3 gates ----
        self.F_z = nn.Linear(emb_dim, factored_dim)
        self.F_r = nn.Linear(emb_dim, factored_dim)
        self.F_n = nn.Linear(emb_dim, factored_dim)

        # ---- Romantic style matrices, 3 per style ----
        self.S_rz = nn.Linear(factored_dim, factored_dim)
        self.S_rr = nn.Linear(factored_dim, factored_dim)
        self.S_rn = nn.Linear(factored_dim, factored_dim)

        # Output projection
        self.C = nn.Linear(hidden_dim, vocab_size)

        # Dropout (applied on the OUTPUT branch only; see forward_step)
        self.dropout = nn.Dropout(p=0.5)

    def forward_step(self, embedded, h_0, mode, features=None):
        """
        Single GRU step with factored style matrices.
        NOTE: No c_0 — GRU has no cell state.
        """
        z = self.V_z(embedded)
        r = self.V_r(embedded)
        n = self.V_n(embedded)

        if features is not None:
            visual_z = self.F_z(features)
            visual_r = self.F_r(features)
            visual_n = self.F_n(features)
        else:
            batch_size = embedded.size(0)
            visual_z = torch.zeros(batch_size, z.size(1), device=embedded.device)
            visual_r = torch.zeros(batch_size, r.size(1), device=embedded.device)
            visual_n = torch.zeros(batch_size, n.size(1), device=embedded.device)

        if mode == "factual":
            z = self.S_fz(z) + visual_z
            r = self.S_fr(r) + visual_r
            n = self.S_fn(n) + visual_n
        elif mode == "romantic":
            z = self.S_rz(z) + 0.6 * visual_z
            r = self.S_rr(r) + 0.6 * visual_r
            n = self.S_rn(n) + 0.6 * visual_n
        else:
            sys.stderr.write("mode name wrong!\n")
            raise ValueError(f"Unknown mode: {mode}. Only 'factual' and 'romantic' supported.")

        z_t = torch.sigmoid(self.U_z(z) + self.W_z(h_0))
        r_t = torch.sigmoid(self.U_r(r) + self.W_r(h_0))
        n_t = torch.tanh(self.U_n(n) + self.W_n(r_t * h_0))

        h_t = (1 - z_t) * h_0 + z_t * n_t

        # DROPOUT FIX: apply dropout ONLY on the output branch.
        # The clean h_t is returned so the recurrent state (GRU's only
        # memory channel) is never corrupted step-to-step.
        outputs = self.C(self.dropout(h_t))
        return outputs, h_t

    def forward(self, captions, features=None, mode="factual"):
        batch_size = captions.size(0)
        embedded = self.B(captions)  # [batch, max_len, emb_dim]

        if mode == "factual" and features is not None:
            embedded = torch.cat((features.unsqueeze(1), embedded), 1)

        device = embedded.device
        h_t = torch.empty(batch_size, self.hidden_dim, device=device)
        nn.init.uniform_(h_t)

        all_outputs = []
        for ix in range(embedded.size(1) - 1):
            emb = embedded[:, ix, :]
            outputs, h_t = self.forward_step(emb, h_t, mode=mode, features=features)
            all_outputs.append(outputs)
        all_outputs = torch.stack(all_outputs, 1)
        return all_outputs

    def sample(self, feature, tokenizer, beam_size=5, max_len=30, mode="factual", repetition_penalty=1.3):
        with torch.no_grad():
            device = feature.device
            h_t = torch.empty(1, self.hidden_dim, device=device)
            torch.nn.init.uniform_(h_t)

            _, h_t = self.forward_step(feature, h_t, mode=mode, features=feature)

            start_id = tokenizer.bos_token_id
            end_id = tokenizer.eos_token_id

            symbol_id = torch.tensor([start_id], device=device)
            candidates = [[0.0, symbol_id, h_t, [start_id]]]

            t = 0
            while t < max_len - 1:
                t += 1
                tmp_candidates = []
                end_flag = True

                for score, last_id, h_t, id_seq in candidates:
                    if id_seq[-1] == end_id:
                        tmp_candidates.append([score, last_id, h_t, id_seq])
                        continue

                    end_flag = False
                    emb = self.B(last_id)
                    output, h_t = self.forward_step(
                        emb, h_t, mode=mode, features=feature
                    )
                    output = output.squeeze(0).squeeze(0)

                    if repetition_penalty != 1.0 and len(id_seq) > 1:
                        for prev_token_id in set(id_seq):
                            if output[prev_token_id] < 0:
                                output[prev_token_id] *= repetition_penalty
                            else:
                                output[prev_token_id] /= repetition_penalty

                    output = torch.log_softmax(output, dim=-1)
                    output, indices = torch.sort(output, descending=True)
                    output = output[:beam_size]
                    indices = indices[:beam_size]

                    for score_val, wid in zip(output, indices):
                        new_score = score + score_val.item()
                        new_id_seq = id_seq + [int(wid.item())]
                        tmp_candidates.append([
                            new_score,
                            wid.unsqueeze(0),
                            h_t,
                            new_id_seq
                        ])

                if end_flag:
                    break

                candidates = sorted(
                    tmp_candidates,
                    key=lambda x: x[0] / len(x[3]),
                    reverse=True
                )[:beam_size]

            return candidates[0][3]
