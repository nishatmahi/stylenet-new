import sys
import torch
import torch.nn as nn
from transformers import ViTModel
import torch.nn.functional as F

# --------- EncoderViT (UNCHANGED from LSTM version) ---------
class EncoderViT(nn.Module):
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
        features = outputs.last_hidden_state[:, 0, :]  # CLS token
        features = self.A(features)
        return features

# --------- FactoredGRU ---------
#
# MAPPING FROM LSTM → GRU:
# ========================
#
# LSTM has 4 gates:            GRU has 3 gates:
#   i (input gate)      →       z (update gate)
#   f (forget gate)     →       r (reset gate)
#   o (output gate)     →       REMOVED (GRU has no output gate)
#   c (cell candidate)  →       n (candidate hidden state)
#
# LSTM state: h_t AND c_t     GRU state: h_t ONLY (no cell state)
#
# LSTM equations:               GRU equations:
#   i = σ(Ui(Si(Vi(x))) + Wi(h))    z = σ(Uz(Sz(Vz(x))) + Wz(h))
#   f = σ(Uf(Sf(Vf(x))) + Wf(h))    r = σ(Ur(Sr(Vr(x))) + Wr(h))
#   o = σ(Uo(So(Vo(x))) + Wo(h))    n = tanh(Un(Sn(Vn(x))) + Wn(r*h))
#   c̃ = tanh(Uc(Sc(Vc(x))) + Wc(h))
#   c = f*c₀ + i*c̃                  h = (1-z)*h₀ + z*n
#   h = o * tanh(c)
#
# Style matrix mapping:
#   S_fi, S_ff, S_fo, S_fc  →  S_fz, S_fr, S_fn  (factual: 4→3)
#   S_ri, S_rf, S_ro, S_rc  →  S_rz, S_rr, S_rn  (romantic: 4→3)
#
# Feature gate mapping:
#   F_i, F_f, F_o, F_c      →  F_z, F_r, F_n     (visual: 4→3)
#

class FactoredGRU(nn.Module):
    def __init__(self, emb_dim, hidden_dim, factored_dim, vocab_size):
        super(FactoredGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        # Embedding layer (SAME as LSTM)
        self.B = nn.Embedding(vocab_size, emb_dim)

        # ---- GRU Gate: z (update gate) ----
        # Controls how much of the new candidate vs old hidden state to use
        # Replaces BOTH input gate (i) and forget gate (f) from LSTM
        self.U_z = nn.Linear(factored_dim, hidden_dim)
        self.S_fz = nn.Linear(factored_dim, factored_dim)   # factual style
        self.V_z = nn.Linear(emb_dim, factored_dim)
        self.W_z = nn.Linear(hidden_dim, hidden_dim)

        # ---- GRU Gate: r (reset gate) ----
        # Controls how much of previous hidden state to forget
        # before computing the candidate
        self.U_r = nn.Linear(factored_dim, hidden_dim)
        self.S_fr = nn.Linear(factored_dim, factored_dim)   # factual style
        self.V_r = nn.Linear(emb_dim, factored_dim)
        self.W_r = nn.Linear(hidden_dim, hidden_dim)

        # ---- GRU Gate: n (candidate hidden state) ----
        # Proposes new content, similar to cell candidate in LSTM
        self.U_n = nn.Linear(factored_dim, hidden_dim)
        self.S_fn = nn.Linear(factored_dim, factored_dim)   # factual style
        self.V_n = nn.Linear(emb_dim, factored_dim)
        self.W_n = nn.Linear(hidden_dim, hidden_dim)

        # ---- Feature-to-gate transformations (visual conditioning) ----
        # 3 gates instead of 4 (no output gate in GRU)
        self.F_z = nn.Linear(emb_dim, factored_dim)   # Feature → update gate
        self.F_r = nn.Linear(emb_dim, factored_dim)   # Feature → reset gate
        self.F_n = nn.Linear(emb_dim, factored_dim)   # Feature → candidate gate

        # ---- Romantic style matrices ----
        # 3 per style instead of 4 (no output gate)
        self.S_rz = nn.Linear(factored_dim, factored_dim)   # romantic update
        self.S_rr = nn.Linear(factored_dim, factored_dim)   # romantic reset
        self.S_rn = nn.Linear(factored_dim, factored_dim)   # romantic candidate

        # Output projection (SAME as LSTM)
        self.C = nn.Linear(hidden_dim, vocab_size)

        # Dropout (SAME as LSTM)
        self.dropout = nn.Dropout(p=0.5)

    def forward_step(self, embedded, h_0, mode, features=None):
        """
        Single GRU step with factored style matrices.

        Args:
            embedded: [batch_size, emb_dim] - current input embedding
            h_0: [batch_size, hidden_dim] - previous hidden state
            mode: str - "factual" or "romantic"
            features: [batch_size, emb_dim] - visual features (optional)

        NOTE: No c_0 parameter — GRU has no cell state!
        """
        # Step 1: Transform input through V matrices (SAME pattern as LSTM)
        z = self.V_z(embedded)
        r = self.V_r(embedded)
        n = self.V_n(embedded)

        # Step 2: Visual feature conditioning (SAME pattern, 3 gates instead of 4)
        if features is not None:
            visual_z = self.F_z(features)
            visual_r = self.F_r(features)
            visual_n = self.F_n(features)
        else:
            batch_size = embedded.size(0)
            visual_z = torch.zeros(batch_size, z.size(1), device=embedded.device)
            visual_r = torch.zeros(batch_size, r.size(1), device=embedded.device)
            visual_n = torch.zeros(batch_size, n.size(1), device=embedded.device)

        # Step 3: Apply style-specific transformations + visual conditioning
        if mode == "factual":
            z = self.S_fz(z) + visual_z
            r = self.S_fr(r) + visual_r
            n = self.S_fn(n) + visual_n

        elif mode == "romantic":
            # Scale down visual features so romantic style isn't overwhelmed
            # by image signal (which would produce factual-sounding captions).
            # 0.2 = gentle visual grounding; adjust if romantic is still too
            # factual (lower) or too disconnected from image (higher).
            z = self.S_rz(z) + 0.1 * visual_z
            r = self.S_rr(r) + 0.1 * visual_r
            n = self.S_rn(n) + 0.1 * visual_n

        else:
            sys.stderr.write("mode name wrong!\n")
            raise ValueError(f"Unknown mode: {mode}. Only 'factual' and 'romantic' supported.")

        # Step 4: Compute GRU gates
        # Update gate: controls blend of old h vs new candidate
        z_t = torch.sigmoid(self.U_z(z) + self.W_z(h_0))

        # Reset gate: controls how much history to forget before computing candidate
        r_t = torch.sigmoid(self.U_r(r) + self.W_r(h_0))

        # Candidate: new hidden state proposal
        # KEY DIFFERENCE from LSTM: reset gate is applied to h_0 BEFORE computing candidate
        n_t = torch.tanh(self.U_n(n) + self.W_n(r_t * h_0))

        # Step 5: Update hidden state
        # KEY DIFFERENCE from LSTM: no separate cell state, no output gate
        # h_t = (1 - z_t) * h_0 + z_t * n_t
        #        ↑ keep old          ↑ add new
        h_t = (1 - z_t) * h_0 + z_t * n_t

        # Apply dropout
        h_t = self.dropout(h_t)

        # Generate output logits
        outputs = self.C(h_t)
        return outputs, h_t

    def forward(self, captions, features=None, mode="factual"):
        """
        Full sequence forward pass.

        Args:
            captions: [batch, max_len] - caption token sequences
            features: [batch, emb_dim] - visual features from images
            mode: str - caption style ("factual", "romantic")

        NOTE: Returns 2 values from forward_step (outputs, h_t) instead of
              3 (outputs, h_t, c_t) — GRU has no cell state.
        """
        batch_size = captions.size(0)
        embedded = self.B(captions)  # [batch, max_len, emb_dim]

        # Factual: features prepended as first timestep (SAME as LSTM)
        if mode == "factual" and features is not None:
            embedded = torch.cat((features.unsqueeze(1), embedded), 1)

        # Initialize hidden state ONLY (no cell state needed!)
        # Create directly on the correct device to avoid CPU→GPU sync overhead
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
        """
        Generate captions with beam search.

        Args:
            feature: [1, emb_dim] - visual features for an image
            beam_size: int - beam width
            max_len: int - max generation length
            mode: str - "factual" or "romantic"
            repetition_penalty: float - penalty for repeating tokens (1.0 = off)

        NOTE: Only h_t is tracked per beam candidate (no c_t).

        StyleNet design: Visual features are ALWAYS passed regardless of mode.
        - Shared params (U, V, W) + F gates learned visual grounding during factual training.
        - Romantic style matrices (S_rz, S_rr, S_rn) learned romantic style from text-only training.
        - At inference: shared visual grounding + romantic style → romantic caption of the image.
        """
        with torch.no_grad():
            device = feature.device

            # Initialize hidden state ONLY (no cell state)
            h_t = torch.empty(1, self.hidden_dim, device=device)
            torch.nn.init.uniform_(h_t)

            # Forward 1 step with image feature
            _, h_t = self.forward_step(feature, h_t, mode=mode, features=feature)

            start_id = tokenizer.bos_token_id
            end_id = tokenizer.eos_token_id

            # Initialize beam — NOTE: no c_t in the tuple!
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

                    # Repetition penalty
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
