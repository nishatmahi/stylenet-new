"""
models.py — the two models of PPCap, both on gpt2-bengali.

  build_factual   VisionEncoderDecoderModel. Cached CLIP patch features go
                  STRAIGHT into the decoder's cross-attention. No custom
                  projector, no alignment loss, no staging.

  StyleModel      ClipCap + GeDi, PPCap's design: the prefix is
                  [style token] ++ MLP(CLIP embedding), prepended to the
                  caption's token embeddings inside GPT-2.

Both use the SAME gpt2-bengali tokenizer, so guided decoding in generate.py is
element-wise on logits. PPCap had to zero out mismatched vocabulary between its
discriminator and factual model (Sec 4.3); sharing the decoder removes that.

Three traps this file avoids, all found by running it:
  1. "openai/clip-vit-base-patch32" is the FULL CLIP (vision+text) and its
     CLIPConfig has no .hidden_size -> AttributeError in
     VisionEncoderDecoderModel.__init__. A vision-only config is required.
     Since cached features are always passed as encoder_outputs the encoder is
     never executed, so it is built from config -- no download, no wasted VRAM.
  2. generate() reads model.generation_config, NOT model.config. Setting only
     m.config leaves eos/pad None and generate() dies with IndexError.
  3. num_attention_heads must divide hidden_size.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import (AutoTokenizer, GPT2LMHeadModel,
                          CLIPVisionConfig, CLIPVisionModel,
                          VisionEncoderDecoderConfig, VisionEncoderDecoderModel)

DEC = "flax-community/gpt2-bengali"


def build_tokenizer(dec=DEC):
    tok = AutoTokenizer.from_pretrained(dec)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    if tok.eos_token_id is None:
        raise RuntimeError("tokenizer has no eos token; generation cannot stop")
    return tok


def _vocab_needed(tok):
    """Never shrink embeddings -- special ids can sit past len(tok)."""
    ids = [i for i in (tok.eos_token_id, tok.bos_token_id, tok.pad_token_id)
           if i is not None]
    return max(len(tok), *(i + 1 for i in ids))


# ------------------------------------------------------------------ factual
def build_factual(tok, feat_dim, dec=DEC):
    vcfg = CLIPVisionConfig(hidden_size=feat_dim, intermediate_size=feat_dim,
                            num_hidden_layers=1, num_attention_heads=1)
    enc = CLIPVisionModel(vcfg)                       # placeholder, never run

    d = GPT2LMHeadModel.from_pretrained(dec, add_cross_attention=True,
                                        is_decoder=True)
    need = _vocab_needed(tok)
    if need > d.config.vocab_size:
        d.resize_token_embeddings(need)

    cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc.config,
                                                                 d.config)
    m = VisionEncoderDecoderModel(encoder=enc, decoder=d, config=cfg)

    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    for c in (m.config, m.generation_config):         # BOTH
        c.decoder_start_token_id = start
        c.bos_token_id = start
        c.eos_token_id = tok.eos_token_id
        c.pad_token_id = tok.pad_token_id
    m.config.vocab_size = m.config.decoder.vocab_size
    return m


def factual_params(m):
    """Decoder + the enc_to_dec_proj layer if the dims did not match."""
    proj = [p for n, p in m.named_parameters() if n.startswith("enc_to_dec_proj")]
    return list(m.decoder.parameters()) + proj, bool(proj)


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
    """Paper Sec 3.3. CLIP text and image embeddings are close but not
    interchangeable; noise stops the model keying on the text-side geometry."""
    if normalize:
        x = x / x.norm(2, dim=-1, keepdim=True).clamp(min=1e-6)
    return x if variance <= 0 else x + torch.randn_like(x) * variance


class StyleModel(nn.Module):
    """Class-conditional captioner trained ONLY on unpaired style text."""

    def __init__(self, clip_dim, tok, dec=DEC, prefix_len=10):
        super().__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained(dec)
        need = _vocab_needed(tok)
        if need > self.gpt.config.vocab_size:
            self.gpt.resize_token_embeddings(need)
        self.d_model = self.gpt.config.n_embd
        self.prefix_len = prefix_len
        self.clip_project = MLP(clip_dim, self.d_model, prefix_len)

    def embeds(self, clip_emb, style_ids, input_ids):
        wte = self.gpt.transformer.wte
        style = wte(style_ids).unsqueeze(1)                      # [B,1,d]
        prefix = self.clip_project(
            clip_emb.to(self.clip_project.net[0].weight.dtype))  # [B,k,d]
        toks = wte(input_ids)                                    # [B,T,d]
        return torch.cat([style, prefix, toks], dim=1)

    def forward(self, clip_emb, style_ids, input_ids, attn_mask):
        e = self.embeds(clip_emb, style_ids, input_ids)
        pre = torch.ones(e.size(0), 1 + self.prefix_len,
                         dtype=attn_mask.dtype, device=attn_mask.device)
        return self.gpt(inputs_embeds=e,
                        attention_mask=torch.cat([pre, attn_mask], 1)).logits

    def step_logits(self, clip_emb, style_ids, ys):
        e = self.embeds(clip_emb, style_ids, ys)
        return self.gpt(inputs_embeds=e).logits[:, -1, :]


def ppcap_loss(model, clip_emb, style_ids, flip_ids, input_ids, attn_mask,
               lam=0.8):
    """L = lam * L_g + (1 - lam) * L_d      (paper Eq. 6-8, lam = 0.8)

    L_g : LM loss under the sentence's TRUE control code.
    L_d : the same sentence scored under the true code and the flipped one;
          cross-entropy over those two sequence log-likelihoods forces the
          model to tell the styles apart, not just to model one corpus.
    """
    B = input_ids.size(0)
    P = 1 + model.prefix_len

    def seq_logprob(sid):
        logits = model(clip_emb, sid, input_ids, attn_mask)[:, P - 1:-1, :]
        lp = F.log_softmax(logits.float(), -1)
        tok_lp = lp.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1) * attn_mask
        return tok_lp.sum(1) / attn_mask.sum(1).clamp(min=1)

    lp_true, lp_flip = seq_logprob(style_ids), seq_logprob(flip_ids)
    l_g = -lp_true.mean()
    l_d = F.cross_entropy(torch.stack([lp_true, lp_flip], 1),
                          torch.zeros(B, dtype=torch.long, device=input_ids.device))
    return lam * l_g + (1 - lam) * l_d, l_g.detach(), l_d.detach()
