import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from constant import get_symbol_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderViT(nn.Module):
    def __init__(self, emb_dim):
        super(EncoderViT, self).__init__()
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        vit.heads = nn.Identity()    # <-- This is the fix
        self.vit = vit
        self.fc = nn.Linear(768, emb_dim)

    def forward(self, images):
        features = self.vit(images)   # Shape: [batch_size, 768]
        features = self.fc(features)  # Shape: [batch_size, emb_dim]
        return features

class FactoredLSTM(nn.Module):
    def __init__(self, emb_dim, hidden_dim, factored_dim, vocab_size):
        super(FactoredLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # embedding
        print(f"vocab_size: {vocab_size}")
        self.B = nn.Embedding(vocab_size, emb_dim)

        # factored lstm weights
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

        self.S_hi = nn.Linear(factored_dim, factored_dim)
        self.S_hf = nn.Linear(factored_dim, factored_dim)
        self.S_ho = nn.Linear(factored_dim, factored_dim)
        self.S_hc = nn.Linear(factored_dim, factored_dim)

        
        self.S_ri = nn.Linear(factored_dim, factored_dim)
        self.S_rf = nn.Linear(factored_dim, factored_dim)
        self.S_ro = nn.Linear(factored_dim, factored_dim)
        self.S_rc = nn.Linear(factored_dim, factored_dim)

        # weight for output
        self.C = nn.Linear(hidden_dim, vocab_size)

    def forward_step(self, embedded, h_0, c_0, mode):
        i = self.V_i(embedded)
        f = self.V_f(embedded)
        o = self.V_o(embedded)
        c = self.V_c(embedded)


        if mode == "factual":
            i = self.S_fi(i)
            f = self.S_ff(f)
            o = self.S_fo(o)
            c = self.S_fc(c)
        elif mode == "humorous":
            i = self.S_hi(i)
            f = self.S_hf(f)
            o = self.S_ho(o)
            c = self.S_hc(c)
        elif mode == "romantic":
            i = self.S_ri(i)
            f = self.S_rf(f)
            o = self.S_ro(o)
            c = self.S_rc(c)
        else:
            sys.stderr.write("mode name wrong!\n")

        i_t = torch.sigmoid(self.U_i(i) + self.W_i(h_0))
        f_t = torch.sigmoid(self.U_f(f) + self.W_f(h_0))
        o_t = torch.sigmoid(self.U_o(o) + self.W_o(h_0))
        c_tilda = torch.tanh(self.U_c(c) + self.W_c(h_0))

        c_t = f_t * c_0 + i_t * c_tilda
        h_t = o_t * c_t

        outputs = self.C(h_t)
        return outputs, h_t, c_t

    def forward(self, captions, features=None, mode="factual"):
        batch_size = captions.size(0)
        embedded = self.B(captions)

        if mode == "factual":
            if features is None:
                sys.stderr.write("features is None!\n")
            embedded = torch.cat((features.unsqueeze(1), embedded), 1)

        h_t = Variable(torch.Tensor(batch_size, self.hidden_dim)).to(device)
        c_t = Variable(torch.Tensor(batch_size, self.hidden_dim)).to(device)
        nn.init.uniform_(h_t)
        nn.init.uniform_(c_t)

        all_outputs = []
        for ix in range(embedded.size(1) - 1):
            emb = embedded[:, ix, :]
            outputs, h_t, c_t = self.forward_step(emb, h_t, c_t, mode=mode)
            all_outputs.append(outputs)

        return torch.stack(all_outputs, 1)

    
    def sample(self, feature, beam_size=5, max_len=30, mode="factual"): 
        with torch.no_grad():
        # Initialize hidden states with zeros
            h_t = Variable(torch.zeros(1, self.hidden_dim))
            c_t = Variable(torch.zeros(1, self.hidden_dim))

        # Device handling
            if torch.cuda.is_available():
                h_t = h_t.cuda()
                c_t = c_t.cuda()
                feature = feature.cuda()

        # Process image feature FIRST for ALL modes
            _, h_t, c_t = self.forward_step(feature, h_t, c_t, mode=mode)

        # Initialize beam candidates with <s> token
            start_token = get_symbol_id('<s>')
            symbol = torch.LongTensor([[start_token]])
            if torch.cuda.is_available():
                symbol = symbol.cuda()
            symbol = Variable(symbol)

            candidates = [[0.0, symbol, h_t.clone(), c_t.clone(), [start_token]]]

        # Beam search loop
            t = 0
            while t < max_len:
                t += 1
                tmp_candidates = []
                end_flag = True

                for score, last_id, h_prev, c_prev, id_seq in candidates:
                    if id_seq[-1] == get_symbol_id('</s>'):
                        tmp_candidates.append([score, last_id, h_prev, c_prev, id_seq])
                        continue
                    
                    end_flag = False
                    emb = self.B(last_id)

                # Forward step with cloned states
                    output, h_new, c_new = self.forward_step(
                        emb, 
                        h_prev.clone(), 
                        c_prev.clone(), 
                        mode=mode
                    )

                # Get probabilities
                    log_probs = F.log_softmax(output.squeeze(1), dim=1)
                    top_log_probs, top_indices = torch.topk(log_probs, beam_size, dim=1)

                # Expand candidates
                    for i in range(beam_size):
                        wid = top_indices[0, i]
                        log_prob = top_log_probs[0, i]

                        new_score = score + log_prob.item()
                        new_seq = id_seq + [wid.item()]

                        wid_tensor = torch.LongTensor([[wid.item()]])
                        if torch.cuda.is_available():
                            wid_tensor = wid_tensor.cuda()
                        wid_var = Variable(wid_tensor)

                        tmp_candidates.append([
                            new_score,
                            wid_var,
                            h_new.clone(),
                            c_new.clone(),
                            new_seq
                        ])

                if end_flag:
                    break
                
            # Normalize scores by sequence length
                tmp_candidates.sort(key=lambda x: -x[0]/len(x[-1]))
                candidates = tmp_candidates[:beam_size]

        # Return best sequence
            best_candidate = max(candidates, key=lambda x: x[0]/len(x[-1]))
            id_seq = best_candidate[-1]

        # Trim at </s> token if exists
        eos_token_id = get_symbol_id('</s>')
        if eos_token_id in id_seq:
            id_seq = id_seq[:id_seq.index(eos_token_id) + 1]  # include </s>

        return id_seq
