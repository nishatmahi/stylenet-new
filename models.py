import sys
import torch
import torch.nn as nn
from transformers import ViTModel
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable


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

# --------- FactoredLSTM ---------
class FactoredLSTM(nn.Module):
    def __init__(self, emb_dim, hidden_dim, factored_dim, vocab_size):
        super(FactoredLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        self.B = nn.Embedding(vocab_size, emb_dim)

        # Factored LSTM weights for each gate
        self.U_i = nn.Linear(factored_dim, hidden_dim)
        self.S_fi = nn.Linear(factored_dim, factored_dim)
        self.V_i = nn.Linear(emb_dim, factored_dim)
        self.W_i = nn.Linear(hidden_dim, hidden_dim)

        self.U_f = nn.Linear(factored_dim, hidden_dim)
        self.S_ff = nn.Linear(factored_dim, factored_dim)
        self.V_f = nn.Linear(emb_dim, factored_dim)
        self.W_f = nn.Linear(hidden_dim, hidden_dim)

        self.U_o = nn.Linear(factored_dim, hidden_dim)
        self.S_fo = nn.Linear(factored_dim, factored_dim)
        self.V_o = nn.Linear(emb_dim, factored_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.U_c = nn.Linear(factored_dim, hidden_dim)
        self.S_fc = nn.Linear(factored_dim, factored_dim)
        self.V_c = nn.Linear(emb_dim, factored_dim)
        self.W_c = nn.Linear(hidden_dim, hidden_dim)

        # Feature-to-gate transformations for visual conditioning
        self.F_i = nn.Linear(emb_dim, factored_dim)  # Feature influence on input gate
        self.F_f = nn.Linear(emb_dim, factored_dim)  # Feature influence on forget gate
        self.F_o = nn.Linear(emb_dim, factored_dim)  # Feature influence on output gate
        self.F_c = nn.Linear(emb_dim, factored_dim)  # Feature influence on cell gate

        # Style-specific transformations for romantic
        self.S_ri = nn.Linear(factored_dim, factored_dim)
        self.S_rf = nn.Linear(factored_dim, factored_dim)
        self.S_ro = nn.Linear(factored_dim, factored_dim)
        self.S_rc = nn.Linear(factored_dim, factored_dim)

        # Style-specific transformations for humorous/funny (COMMENTED OUT)
        # self.S_hi = nn.Linear(factored_dim, factored_dim)
        # self.S_hf = nn.Linear(factored_dim, factored_dim)
        # self.S_ho = nn.Linear(factored_dim, factored_dim)
        # self.S_hc = nn.Linear(factored_dim, factored_dim)

        self.C = nn.Linear(hidden_dim, vocab_size)

        # Optional dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward_step(self, embedded, h_0, c_0, mode, features=None):
        """
        Args:
            embedded: [batch_size, emb_dim] - current input embedding
            h_0: [batch_size, hidden_dim] - previous hidden state
            c_0: [batch_size, hidden_dim] - previous cell state
            mode: str - "factual" or "romantic"
            features: [batch_size, emb_dim] - visual features (required for factual mode)
        """
        # Transform input through V matrices
        i = self.V_i(embedded)
        f = self.V_f(embedded)
        o = self.V_o(embedded)
        c = self.V_c(embedded)

        # ALL MODES get visual feature conditioning for image-aware generation
        if features is not None:
            # Base visual conditioning (applied to all modes)
            visual_i = self.F_i(features)
            visual_f = self.F_f(features)
            visual_o = self.F_o(features)
            visual_c = self.F_c(features)
        else:
            # If no features provided, use zeros (for text-only training)
            batch_size = embedded.size(0)
            visual_i = torch.zeros(batch_size, i.size(1), device=embedded.device)
            visual_f = torch.zeros(batch_size, f.size(1), device=embedded.device)
            visual_o = torch.zeros(batch_size, o.size(1), device=embedded.device)
            visual_c = torch.zeros(batch_size, c.size(1), device=embedded.device)

        # Apply style-specific transformations + visual conditioning
        if mode == "factual":
            i = self.S_fi(i) + visual_i  # Factual style + visual info
            f = self.S_ff(f) + visual_f
            o = self.S_fo(o) + visual_o
            c = self.S_fc(c) + visual_c
            
        elif mode == "romantic":
            i = self.S_ri(i) + visual_i  # Romantic style + visual info
            f = self.S_rf(f) + visual_f
            o = self.S_ro(o) + visual_o
            c = self.S_rc(c) + visual_c
            
        # elif mode == "humorous":
        #     i = self.S_hi(i) + visual_i  # Humorous style + visual info
        #     f = self.S_hf(f) + visual_f
        #     o = self.S_ho(o) + visual_o
        #     c = self.S_hc(c) + visual_c
        else:
            sys.stderr.write("mode name wrong!\n")
            raise ValueError(f"Unknown mode: {mode}. Only 'factual' and 'romantic' supported.")

        # Compute LSTM gates
        i_t = torch.sigmoid(self.U_i(i) + self.W_i(h_0))
        f_t = torch.sigmoid(self.U_f(f) + self.W_f(h_0))
        o_t = torch.sigmoid(self.U_o(o) + self.W_o(h_0))
        c_tilda = torch.tanh(self.U_c(c) + self.W_c(h_0))

        # Update cell and hidden states
        c_t = f_t * c_0 + i_t * c_tilda
        h_t = o_t * torch.tanh(c_t)

        # Apply dropout regularization
        h_t = self.dropout(h_t)

        # Generate output logits
        outputs = self.C(h_t)
        return outputs, h_t, c_t

    def forward(self, captions, features=None, mode="factual"):
        """
        
        
        Training Strategy:
        - Factual mode: Use image+caption pairs (features provided)
        - Romantic: Use style text only (features=None for language modeling)
        
        Inference Strategy:
        - ALL modes: Use image features (features provided) for visual conditioning
        """
        batch_size = captions.size(0)
        embedded = self.B(captions)  # [batch, max_len, emb_dim]
        
        # For factual training: use features as first timestep (image+caption pairs)
        # For style training: no features concatenation (text-only language modeling)
        if mode == "factual" and features is not None:
            embedded = torch.cat((features.unsqueeze(1), embedded), 1)

        # Initialize hidden/cell state with uniform distribution (matching original)
        h_t = Variable(torch.Tensor(batch_size, self.hidden_dim))
        c_t = Variable(torch.Tensor(batch_size, self.hidden_dim))
        nn.init.uniform_(h_t)
        nn.init.uniform_(c_t)
        if torch.cuda.is_available():
            h_t = h_t.cuda()
            c_t = c_t.cuda()

        all_outputs = []
        for ix in range(embedded.size(1) - 1):
            emb = embedded[:, ix, :]
            # Pass features for visual conditioning in ALL modes during inference
            # During training: factual gets features, romantic/humorous get None
            outputs, h_t, c_t = self.forward_step(emb, h_t, c_t, mode=mode, features=features)
            all_outputs.append(outputs)
        all_outputs = torch.stack(all_outputs, 1)
        return all_outputs

    def sample(self, feature, tokenizer, beam_size=5, max_len=30, mode="factual"):
        """
        Generate captions from feature vectors with beam search
        Args:
            feature: [1, emb_dim] - visual features for an image
            beam_size: int - beam size for beam search
            max_len: int - max sampling length
            mode: str - caption style ("factual", "romantic", "humorous")
        
        NOTE: ALL modes use visual features during inference for image-conditioned generation
        """
        with torch.no_grad():
            device = feature.device

            # Initialize hidden state (EXACT original)
            h_t = torch.Tensor(1, self.hidden_dim)
            c_t = torch.Tensor(1, self.hidden_dim)
            # EXACTLY match original initialization
            torch.nn.init.uniform_(h_t)
            torch.nn.init.uniform_(c_t)
            h_t = h_t.to(device)
            c_t = c_t.to(device)

            # Forward 1 step with image feature (ALL modes get visual conditioning during inference)
            _, h_t, c_t = self.forward_step(feature, h_t, c_t, mode=mode, features=feature)

            # Use tokenizer's special tokens
            start_id = tokenizer.bos_token_id
            end_id = tokenizer.eos_token_id

            # Initialize beam (EXACT original structure)
            symbol_id = torch.tensor([start_id], device=device).unsqueeze(0)
            candidates = [[0.0, symbol_id, h_t, c_t, [start_id]]]

            # Beam search (EXACT original logic)
            t = 0
            while t < max_len - 1:
                t += 1
                tmp_candidates = []
                end_flag = True

                for score, last_id, h_t, c_t, id_seq in candidates:
                    # Skip finished sequences
                    if id_seq[-1] == end_id:
                        tmp_candidates.append([score, last_id, h_t, c_t, id_seq])
                        continue

                    end_flag = False
                    emb = self.B(last_id)
                    # ALL modes get visual conditioning during inference
                    output, h_t, c_t = self.forward_step(
                        emb, h_t, c_t, mode=mode, features=feature
                    )
                    output = output.squeeze(0).squeeze(0)

                    # Log softmax + sort (EXACT original)
                    output = torch.log_softmax(output, dim=-1)
                    output, indices = torch.sort(output, descending=True)
                    output = output[:beam_size]
                    indices = indices[:beam_size]

                    # Create new candidates (EXACT original)
                    for score_val, wid in zip(output, indices):
                        new_score = score + score_val.item()
                        new_id_seq = id_seq + [int(wid.item())]
                        tmp_candidates.append([
                            new_score,
                            wid.unsqueeze(0),  # Keep as tensor [1,1]
                            h_t,
                            c_t,
                            new_id_seq
                        ])

                # Break if all candidates finished (EXACT original)
                if end_flag:
                    break

                # Sort by normalized log probability (EXACT original)
                candidates = sorted(
                    tmp_candidates,
                    key=lambda x: x[0] / len(x[4]),  # Normalized score
                    reverse=True
                )[:beam_size]

            # Return best sequence (EXACT original)
            return candidates[0][4]














