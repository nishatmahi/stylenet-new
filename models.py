"""
models.py — the two models of PPCap, both on gpt2-bengali.

  FactualCaptioner   HuggingFace VisionEncoderDecoderModel. Cached CLIP patch
                     features go STRAIGHT into the decoder's cross-attention.
                     No custom projector, no alignment loss, no staging --
                     this is the ViT-GPT2 recipe, unmodified.

  StyleModel         ClipCap + GeDi, exactly PPCap's design: the prefix is
                     [style token embedding] ++ MLP(CLIP embedding), prepended
                     to the caption's token embeddings inside GPT-2.

Both use the SAME gpt2-bengali tokenizer, so their vocabularies match and the
guided decoding in generate.py is an element-wise operation on logits. PPCap
had to zero out mismatched vocabulary between its discriminator and its
factual model (Sec 4.3); sharing the decoder removes that problem.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import (VisionEncoderDecoderModel, GPT2LMHeadModel,
                          AutoTokenizer)
from transformers.modeling_outputs import BaseModelOutput

ENC = "openai/clip-vit-base-patch32"
DEC = "flax-community/gpt2-bengali"


def build_tokenizer(dec=DEC):
    tok = AutoTokenizer.from_pretrained(dec)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ------------------------------------------------------------------ factual
def build_factual(tok, enc=ENC, dec=DEC):
    """Standard VisionEncoderDecoder. Nothing hand-rolled."""
    m = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(enc, dec)
    m.decoder.resize_token_embeddings(len(tok))
    m.config.decoder_start_token_id = tok.bos_token_id or tok.eos_token_id
    m.config.pad_token_id = tok.pad_token_id
    m.config.eos_token_id = tok.eos_token_id
    m.config.vocab_size = m.config.decoder.vocab_size
    return m


# ------------------------------------------------------------------ style
class MLP(nn.Module):
    """CLIP embedding -> prefix_len token embeddings. ClipCap's mapper."""

    def __init__(self, in_dim, d_model, prefix_len):
        super().__init__()
        out = d_model * prefix_len
        self.net = nn.Sequential(nn.Linear(in_dim, out // 2), nn.Tanh(),
                                 nn.Linear(out // 2, out))
        self.d_model, self.prefix_len = d_model, prefix_len

    def forward(self, x):
        return self.net(x).view(-1, self.prefix_len, self.d_model)


def noise_injection(x, variance, normalize=True):
    """Paper Sec 3.3: CLIP text and image embeddings are close but not
    interchangeable. Noise during training stops the model keying on the exact
    text-side geometry. variance = 0.016 in the paper."""
    if normalize:
        x = x / x.norm(2, dim=-1, keepdim=True).clamp(min=1e-6)
    return x if variance <= 0 else x + torch.randn_like(x) * variance


class StyleModel(nn.Module):
    """Class-conditional captioner trained ONLY on unpaired style text."""

    def __init__(self, clip_dim, tok, dec=DEC, prefix_len=10):
        super().__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained(dec)
        self.gpt.resize_token_embeddings(len(tok))
        self.d_model = self.gpt.config.n_embd
        self.prefix_len = prefix_len
        self.clip_project = MLP(clip_dim, self.d_model, prefix_len)

    def embeds(self, clip_emb, style_ids, input_ids):
        """[style] ++ MLP(clip) ++ caption tokens, as embeddings."""
        wte = self.gpt.transformer.wte
        style = wte(style_ids).unsqueeze(1)                     # [B,1,d]
        prefix = self.clip_project(
            clip_emb.to(self.clip_project.net[0].weight.dtype))  # [B,k,d]
        toks = wte(input_ids)                                   # [B,T,d]
        return torch.cat([style, prefix, toks], dim=1)

    def forward(self, clip_emb, style_ids, input_ids, attn_mask):
        e = self.embeds(clip_emb, style_ids, input_ids)
        pre = torch.ones(e.size(0), 1 + self.prefix_len,
                         dtype=attn_mask.dtype, device=attn_mask.device)
        return self.gpt(inputs_embeds=e,
                        attention_mask=torch.cat([pre, attn_mask], 1)).logits

    def step_logits(self, clip_emb, style_ids, ys):
        """Next-token logits for the running sequence ys. Used by generate.py."""
        e = self.embeds(clip_emb, style_ids, ys)
        return self.gpt(inputs_embeds=e).logits[:, -1, :]


def ppcap_loss(model, clip_emb, style_ids, flip_ids, input_ids, attn_mask,
               labels, is_style, lam=0.8):
    """L = lam * L_g + (1 - lam) * L_d      (paper Eq. 6-8, lam = 0.8)

    L_g : ordinary LM loss under the sentence's TRUE control code.
    L_d : the discriminative half. The same sentence is scored under the true
          code and the flipped one; cross-entropy over those two sequence
          log-likelihoods forces the model to tell the styles apart rather
          than merely to model one corpus.
    """
    B, T = input_ids.shape
    P = 1 + model.prefix_len

    def seq_logprob(sid):
        logits = model(clip_emb, sid, input_ids, attn_mask)[:, P - 1:-1, :]
        lp = F.log_softmax(logits.float(), -1)
        tgt = input_ids.unsqueeze(-1)
        tok_lp = lp.gather(-1, tgt).squeeze(-1) * attn_mask
        return tok_lp.sum(1) / attn_mask.sum(1).clamp(min=1)

    lp_true = seq_logprob(style_ids)
    lp_flip = seq_logprob(flip_ids)

    l_g = -lp_true.mean()
    class_logits = torch.stack([lp_true, lp_flip], 1)
    l_d = F.cross_entropy(class_logits,
                          torch.zeros(B, dtype=torch.long, device=input_ids.device))
    return lam * l_g + (1 - lam) * l_d, l_g.detach(), l_d.detach()
