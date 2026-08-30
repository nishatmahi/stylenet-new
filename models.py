"""
models.py — factual captioner (unchanged) + a TEXT-ONLY GeDi style discriminator.

  build_factual   VisionEncoderDecoderModel, gpt2-bengali decoder, CLIP ViT-B/32
                  patch features into cross-attention. UNCHANGED.

  StyleModel      gpt2-bengali class-conditional LM. Input is [control token] ++
                  caption tokens. NO CLIP embedding, no prefix, no image. Trained
                  on the style corpus vs the factual text corpus; at decoding it is
                  run twice (desired vs undesired control code) and contrasted.

Both models share the gpt2-bengali vocabulary, so guided decoding is element-wise
on logits. The factual head is base width V; the style head is V+3 (the control
tokens), so its logits are sliced to V before mixing.
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
    ids = [i for i in (tok.eos_token_id, tok.bos_token_id, tok.pad_token_id)
           if i is not None]
    return max(len(tok), *(i + 1 for i in ids))


# ------------------------------------------------------------------ factual (unchanged)
def build_factual(tok, feat_dim, dec=DEC):
    vcfg = CLIPVisionConfig(hidden_size=feat_dim, intermediate_size=feat_dim,
                            num_hidden_layers=1, num_attention_heads=1)
    enc = CLIPVisionModel(vcfg)

    d = GPT2LMHeadModel.from_pretrained(dec, add_cross_attention=True,
                                        is_decoder=True)
    need = _vocab_needed(tok)
    if need > d.config.vocab_size:
        d.resize_token_embeddings(need)

    cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc.config,
                                                                 d.config)
    m = VisionEncoderDecoderModel(encoder=enc, decoder=d, config=cfg)

    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    for c in (m.config, m.generation_config):
        c.decoder_start_token_id = start
        c.bos_token_id = start
        c.eos_token_id = tok.eos_token_id
        c.pad_token_id = tok.pad_token_id
    m.config.vocab_size = m.config.decoder.vocab_size
    return m


def factual_params(m):
    proj = [p for n, p in m.named_parameters() if n.startswith("enc_to_dec_proj")]
    return list(m.decoder.parameters()) + proj, bool(proj)


# ------------------------------------------------------------------ style (text-only GeDi)
class StyleModel(nn.Module):
    """Class-conditional gpt2-bengali LM. A control token is prepended to the
    caption tokens; nothing else conditions it. Trained on unpaired style text."""

    def __init__(self, tok, dec=DEC):
        super().__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained(dec)
        need = _vocab_needed(tok)
        if need > self.gpt.config.vocab_size:
            self.gpt.resize_token_embeddings(need)
        self.d_model = self.gpt.config.n_embd

    def embeds(self, style_ids, input_ids):
        wte = self.gpt.transformer.wte
        style = wte(style_ids).unsqueeze(1)   # [B,1,d]  the control token
        toks = wte(input_ids)                 # [B,T,d]
        return torch.cat([style, toks], dim=1)

    def forward(self, style_ids, input_ids, attn_mask):
        e = self.embeds(style_ids, input_ids)
        pre = torch.ones(e.size(0), 1, dtype=attn_mask.dtype, device=attn_mask.device)
        return self.gpt(inputs_embeds=e,
                        attention_mask=torch.cat([pre, attn_mask], 1)).logits

    def step_logits(self, style_ids, ys):
        e = self.embeds(style_ids, ys)
        return self.gpt(inputs_embeds=e).logits[:, -1, :]


def seq_logprob(model, style_ids, input_ids, attn_mask):
    """Length-normalised log p(caption | control code). Prefix is 1 (the code)."""
    P = 1
    logits = model(style_ids, input_ids, attn_mask)[:, P - 1:-1, :]
    tok_lp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                              input_ids.reshape(-1),
                              reduction='none').view(input_ids.shape)
    tok_lp = tok_lp * attn_mask
    return tok_lp.sum(1) / attn_mask.sum(1).clamp(min=1)


def ppcap_loss(model, style_ids, flip_ids, input_ids, attn_mask, lam=0.8):
    """L = lam * L_g + (1 - lam) * L_d   (PPCap Eq. 6-8).
    L_g : LM loss under the TRUE control code.
    L_d : true code must give higher sequence log-likelihood than the flipped one."""
    B = input_ids.size(0)
    lp_true = seq_logprob(model, style_ids, input_ids, attn_mask)
    lp_flip = seq_logprob(model, flip_ids, input_ids, attn_mask)
    l_g = -lp_true.mean()
    l_d = F.cross_entropy(torch.stack([lp_true, lp_flip], 1),
                          torch.zeros(B, dtype=torch.long, device=input_ids.device))
    return lam * l_g + (1 - lam) * l_d, l_g.detach(), l_d.detach()
