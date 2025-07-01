import os
import pickle
import argparse
import torch
from torch.autograd import Variable
from data_loader import get_data_loader, get_styled_data_loader
from models import EncoderViT, FactoredLSTM
from loss import masked_cross_entropy
from transformers import AutoTokenizer
import numpy as np
from tqdm.auto import tqdm
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_var(x, device="cuda", requires_grad=False):
    return x.to(device).requires_grad_(requires_grad)

def eval_outputs(outputs, tokenizer):
    indices = torch.topk(outputs, 1, dim=2)[1]
    indices = indices.squeeze(2).cpu().numpy()
    for i in range(len(indices)):
        caption_list = tokenizer.convert_ids_to_tokens(indices[i])
        caption = " ".join(caption_list)
        print(caption)
        print()

# Paths
permanent_save_folder = "stylenet_models/"
os.makedirs(config.model_path, exist_ok=True)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("hishab/titulm-mpt-1b-v2.0", trust_remote_code=True)
print("Pad token:", tokenizer.pad_token)
print("Pad token ID:", tokenizer.pad_token_id)

pad_token_id = tokenizer.pad_token_id
vocab = tokenizer.get_vocab()

# Data Loaders
data_loader = get_data_loader(config.img_path, config.factual_caption_path, tokenizer, config.caption_batch_size, shuffle=True)
styled_data_loader = get_styled_data_loader(config.humorous_caption_path, tokenizer, config.language_batch_size, shuffle=True)
romantic_styled_data_loader = get_styled_data_loader(config.romantic_caption_path, tokenizer, config.language_batch_size, shuffle=True)

# Models
encoder = EncoderViT(config.emb_dim).to(device)
decoder = FactoredLSTM(config.emb_dim, config.hidden_dim, config.factored_dim, tokenizer.vocab_size).to(device)

# Optimizers
criterion = masked_cross_entropy
cap_params = list(decoder.parameters()) + list(encoder.parameters())
optimizer_cap = torch.optim.Adam(cap_params, lr=config.lr_caption)
optimizer_lang = torch.optim.Adam(decoder.parameters(), lr=config.lr_language)

# ======= Checkpoint Loading (NEW SECTION) =======
start_epoch = 0
checkpoint_path = os.path.join(permanent_save_folder, 'checkpoint-latest.pth')
encoder_last_path = os.path.join(permanent_save_folder, "encoder-last.pkl")
decoder_last_path = os.path.join(permanent_save_folder, "decoder-last.pkl")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer_cap.load_state_dict(checkpoint['optimizer_cap_state_dict'])
    optimizer_lang.load_state_dict(checkpoint['optimizer_lang_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
else:
    loaded_any = False
    if os.path.exists(decoder_last_path):
        decoder.load_state_dict(torch.load(decoder_last_path, map_location=device))
        print("Decoder loaded from saved weight")
        loaded_any = True
    if os.path.exists(encoder_last_path):
        encoder.load_state_dict(torch.load(encoder_last_path, map_location=device))
        print("Encoder loaded from saved weight")
        loaded_any = True
    if not loaded_any:
        print("No checkpoint or pretrained weights found. Training from scratch (random weights).")
    else:
        print("No checkpoint found. Loaded latest pretrained weights only.")

# ======= END Checkpoint Loading =======

# ======= Training Loop (start from start_epoch) =======
for epoch in range(start_epoch, config.epoch_num):
    # Caption - Factual
    encoder.train()
    decoder.train()
    for i, (images, captions, lengths) in tqdm(enumerate(data_loader)):
        images = to_var(images.float(), device=device)
        captions = to_var(captions.long(), device=device)

        # Forward, backward and optimize
        optimizer_cap.zero_grad()
        features = encoder(images)
        outputs = decoder(captions, features, mode="factual")
        loss = criterion(outputs[:, 1:, :].contiguous(),
                        captions[:, 1:].contiguous(), lengths - 1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(cap_params, 1.0)
        optimizer_cap.step()

        if i % config.log_step_caption == 0:
            print(f"Epoch [{epoch+1}/{config.epoch_num}], CAP, Loss: {loss.item():.4f}")

    # Style training loops (humorous and romantic remain exactly the same)
    for i, (captions, lengths) in tqdm(enumerate(styled_data_loader)):
        captions = to_var(captions.long(), device=device)
        optimizer_lang.zero_grad()
        outputs = decoder(captions, mode='humorous')
        loss = criterion(outputs, captions[:, 1:].contiguous(), lengths - 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer_lang.step()

    for i, (captions, lengths) in tqdm(enumerate(romantic_styled_data_loader)):
        captions = to_var(captions.long(), device=device)
        optimizer_lang.zero_grad()
        outputs = decoder(captions, mode='romantic')
        loss = criterion(outputs, captions[:, 1:].contiguous(), lengths - 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer_lang.step()

    # ======= Only Save the Latest Weights & Checkpoint =======
    os.makedirs(permanent_save_folder, exist_ok=True)
    torch.save(decoder.state_dict(), os.path.join(permanent_save_folder, 'decoder-last.pkl'))
    torch.save(encoder.state_dict(), os.path.join(permanent_save_folder, 'encoder-last.pkl'))
    torch.save(decoder.state_dict(), os.path.join(config.model_path, 'decoder-last.pkl'))
    torch.save(encoder.state_dict(), os.path.join(config.model_path, 'encoder-last.pkl'))
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_cap_state_dict': optimizer_cap.state_dict(),
        'optimizer_lang_state_dict': optimizer_lang.state_dict(),
        'loss': loss.item(),
    }, os.path.join(permanent_save_folder, 'checkpoint-latest.pth'))
    print(f"[Checkpoint] Saved at end of epoch {epoch+1}")

# ======= END Training Loop =======
