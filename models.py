import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

# --------- EncoderCNN (ResNet152) ---------
class EncoderCNN(nn.Module):
    def __init__(self, emb_dim):
        '''
        Load the pretrained ResNet152 and replace fc
        '''
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.A = nn.Linear(resnet.fc.in_features, emb_dim)

    def forward(self, images):
        '''Extract the image feature vectors'''
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.A(features)
        return features

# --------- FactoredLSTM ---------
class FactoredLSTM(nn.Module):
    def __init__(self, emb_dim, hidden_dim, factored_dim, vocab_size):
        super(FactoredLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

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

        # Style-specific
        self.S_hi = nn.Linear(factored_dim, factored_dim)
        self.S_hf = nn.Linear(factored_dim, factored_dim)
        self.S_ho = nn.Linear(factored_dim, factored_dim)
        self.S_hc = nn.Linear(factored_dim, factored_dim)
        # If you want romantic style, uncomment these:
        self.S_ri = nn.Linear(factored_dim, factored_dim)
        self.S_rf = nn.Linear(factored_dim, factored_dim)
        self.S_ro = nn.Linear(factored_dim, factored_dim)
        self.S_rc = nn.Linear(factored_dim, factored_dim)

        self.C = nn.Linear(hidden_dim, vocab_size)

        # Optional dropout for regularization (add if you want)
        self.dropout = nn.Dropout(p=0.3)

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
        h_t = o_t * torch.tanh(c_t)

        # dropout regularization
        h_t = self.dropout(h_t)

        outputs = self.C(h_t)
        return outputs, h_t, c_t

    def forward(self, captions, features=None, mode="factual"):
        '''
        Args:
            features: fixed vectors from images, [batch, emb_dim]
            captions: [batch, max_len]
            mode: type of caption to generate
        '''
        batch_size = captions.size(0)
        embedded = self.B(captions)  # [batch, max_len, emb_dim]
        if mode == "factual":
            if features is None:
                sys.stderr.write("features is None!\n")
            embedded = torch.cat((features.unsqueeze(1), embedded), 1)

        # Stylenet: hidden/cell state — random uniform
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
            outputs, h_t, c_t = self.forward_step(emb, h_t, c_t, mode=mode)
            all_outputs.append(outputs)
        all_outputs = torch.stack(all_outputs, 1)
        return all_outputs

    def sample(self, feature, tokenizer, beam_size=5, max_len=30, mode="factual"):
        '''
        generate captions from feature vectors with beam search
        Args:
            feature: fixed vector for an image, [1, emb_dim]
            beam_size: stock size for beam search
            max_len: max sampling length
            mode: type of caption to generate
        '''
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

            # Forward 1 step with image feature
            _, h_t, c_t = self.forward_step(feature, h_t, c_t, mode=mode)

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
                    output, h_t, c_t = self.forward_step(emb, h_t, c_t, mode=mode)
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
