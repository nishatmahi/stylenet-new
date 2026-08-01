import os
import gc
import re
import argparse
import random
import torch
import torch.nn as nn

from data_loader import get_data_loader, get_styled_data_loader, tokenizer
from models import EncoderViT, TransformerFactoredDecoder

# --- Kaggle disk hygiene: keep HF caches off the persistent /kaggle/working quota ---
os.environ.setdefault("HF_DATASETS_CACHE", "/kaggle/tmp/hf_datasets_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/kaggle/tmp/hf_model_cache")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def split_caption_file(input_file, train_file, val_file, train_ratio=0.8, seed=42):
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found, skipping split")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.seed(seed)
    random.shuffle(lines)

    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(val_file), exist_ok=True)

    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)

    print(f"Split {input_file}: {len(train_lines)} train, {len(val_lines)} val")


def create_data_splits(args):
    train_dir = '/kaggle/working/train_split'
    val_dir = '/kaggle/working/val_split'

    factual_train = os.path.join(train_dir, 'factual_train.txt')
    factual_val = os.path.join(val_dir, 'factual_val.txt')
    split_caption_file(args.factual_caption_path, factual_train, factual_val)

    romantic_train = os.path.join(train_dir, 'romantic_train.txt')
    romantic_val = os.path.join(val_dir, 'romantic_val.txt')
    if args.romantic_caption_path and os.path.exists(args.romantic_caption_path):
        split_caption_file(args.romantic_caption_path, romantic_train, romantic_val)

    return {
        'factual_train': factual_train,
        'factual_val': factual_val,
        'romantic_train': romantic_train if os.path.exists(romantic_train) else None,
        'romantic_val': romantic_val if os.path.exists(romantic_val) else None,
    }


def compute_loss(logits, labels, pad_token_id):
    """
    Replaces masked_cross_entropy + the external `lengths` argument.
    Padding is masked automatically via ignore_index — no separate length
    bookkeeping needed since GPT2's logits are already shift-aligned inside
    TransformerFactoredDecoder.forward().
    """
    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    return loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def validate_epoch(encoder, decoder, val_loader, val_styled_loader, device, pad_token_id):
    encoder.eval()
    decoder.eval()

    factual_loss, factual_samples = 0.0, 0

    with torch.no_grad():
        if val_loader:
            for images, captions, _ in val_loader:
                images = images.to(device)
                captions = captions.long().to(device)

                features = encoder(images)
                logits = decoder(captions, features, mode="factual")
                loss = compute_loss(logits, captions[:, 1:], pad_token_id)

                factual_loss += loss.item() * captions.size(0)
                factual_samples += captions.size(0)

            if factual_samples > 0:
                factual_loss /= factual_samples
                print(f"Validation Factual Loss: {factual_loss:.4f}")

        if val_styled_loader:
            romantic_loss, romantic_samples = 0.0, 0
            for captions, _ in val_styled_loader:
                captions = captions.long().to(device)
                logits = decoder(captions, mode='romantic')  # features=None, text-only — matches training
                loss = compute_loss(logits, captions[:, 1:], pad_token_id)

                romantic_loss += loss.item() * captions.size(0)
                romantic_samples += captions.size(0)

            if romantic_samples > 0:
                romantic_loss /= romantic_samples
                print(f"Validation Romantic Loss (monitor only): {romantic_loss:.4f}")

    return factual_loss if factual_samples > 0 else float('inf')


def eval_outputs(logits, tokenizer):
    indices = torch.argmax(logits, dim=-1)  # (B, T-1)
    indices = indices.data.cpu().numpy()
    for i in range(min(3, len(indices))):
        text = tokenizer.decode(indices[i], skip_special_tokens=True)
        print(f"Generated {i+1}: {text}")


def save_checkpoint_safely(path, encoder, decoder, optimizer_cap, optimizer_lang, epoch, best_val_loss,
                            avg_factual_loss, avg_romantic_loss, val_loss, patience_counter):
    """
    Atomic overwrite: write to .tmp then os.replace(). Never more than one
    checkpoint file exists on disk at a time. Frozen ViT weights are
    excluded — they're re-downloadable and add nothing but disk weight.
    """
    tmp_path = path + ".tmp"
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': {k: v for k, v in encoder.state_dict().items() if not k.startswith('vit.')},
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_cap_state_dict': optimizer_cap.state_dict(),
        'optimizer_lang_state_dict': optimizer_lang.state_dict(),
        'best_val_loss': best_val_loss,
        'factual_train_loss': avg_factual_loss,
        'romantic_train_loss': avg_romantic_loss,
        'val_loss': val_loss,
        'patience_counter': patience_counter,
    }, tmp_path)
    os.replace(tmp_path, path)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    permanent_save_folder = "stylenet_transformer_models/"
    os.makedirs(permanent_save_folder, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    print("Creating train/validation splits...")
    split_paths = create_data_splits(args)

    train_loader = get_data_loader(
        args.img_path, split_paths['factual_train'],
        batch_size=args.caption_batch_size, shuffle=True)

    train_styled_loader = get_styled_data_loader(
        split_paths['romantic_train'], batch_size=args.language_batch_size,
        shuffle=True) if split_paths['romantic_train'] else None

    val_loader = get_data_loader(
        args.img_path, split_paths['factual_val'],
        batch_size=args.caption_batch_size, shuffle=False) if split_paths['factual_val'] else None

    val_styled_loader = get_styled_data_loader(
        split_paths['romantic_val'], batch_size=args.language_batch_size,
        shuffle=False) if split_paths['romantic_val'] else None

    print(f"Train batches: Factual={len(train_loader)}, "
          f"Romantic={len(train_styled_loader) if train_styled_loader else 0}")
    print(f"Val batches: Factual={len(val_loader) if val_loader else 0}, "
          f"Romantic={len(val_styled_loader) if val_styled_loader else 0}")

    # --- Models ---
    encoder = EncoderViT(decoder_hidden_size=768).to(device)
    decoder = TransformerFactoredDecoder(
        tokenizer=tokenizer,
        gpt2_name="flax-community/gpt2-bengali",
        factored_dim=args.factored_dim,
    ).to(device)

    pad_token_id = decoder.pad_token_id

    # Same optimizer split as your original: cap_params drives factual training
    # (decoder + encoder's trainable projection), lang_params drives romantic
    # training (decoder only — encoder.A never sees romantic gradients, same
    # as before since romantic batches never call encoder()).
    cap_params = list(decoder.parameters()) + list(encoder.A.parameters())
    lang_params = list(decoder.parameters())
    optimizer_cap = torch.optim.Adam(cap_params, lr=args.lr_caption)
    optimizer_lang = torch.optim.Adam(lang_params, lr=args.lr_language)

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    checkpoint_path = os.path.join(permanent_save_folder, 'checkpoint-latest.pth')
    best_model_path = os.path.join(permanent_save_folder, 'best_model.pth')

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)  # strict=False: vit.* excluded
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer_cap.load_state_dict(checkpoint['optimizer_cap_state_dict'])
        optimizer_lang.load_state_dict(checkpoint['optimizer_lang_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"[DEBUG] Resumed from epoch {checkpoint['epoch']+1}, best_val_loss={best_val_loss:.4f}")
    else:
        print("[DEBUG] No checkpoint found. Training from scratch (GPT2 backbone still pretrained).")

    for epoch in range(start_epoch, args.epoch_num):
        print(f"\n[DEBUG] Training epoch {epoch+1} of {args.epoch_num}")
        encoder.train()
        decoder.train()

        factual_train_loss, factual_train_samples = 0.0, 0
        romantic_train_loss, romantic_train_samples = 0.0, 0

        logits = None  # for eval_outputs after the loop
        for i, (images, captions, _) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.long().to(device)

            optimizer_cap.zero_grad(set_to_none=True)
            features = encoder(images)
            logits = decoder(captions, features, mode="factual")
            loss = compute_loss(logits, captions[:, 1:], pad_token_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cap_params, 1.0)
            optimizer_cap.step()

            factual_train_loss += loss.item() * captions.size(0)
            factual_train_samples += captions.size(0)

            if i % args.log_step_caption == 0 or i == len(train_loader) - 1:
                print(f"Epoch [{epoch+1}/{args.epoch_num}], CAP, Step [{i}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

            del images, captions, features, logits, loss
            if i % 200 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        if logits is not None:
            eval_outputs(logits.detach(), tokenizer)

        if train_styled_loader:
            for i, (captions, _) in enumerate(train_styled_loader):
                captions = captions.long().to(device)

                optimizer_lang.zero_grad(set_to_none=True)
                logits = decoder(captions, mode='romantic')  # features=None: text-only, matches original
                loss = compute_loss(logits, captions[:, 1:], pad_token_id)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lang_params, 1.0)
                optimizer_lang.step()

                romantic_train_loss += loss.item() * captions.size(0)
                romantic_train_samples += captions.size(0)

                if i % args.log_step_language == 0 or i == len(train_styled_loader) - 1:
                    print(f"Epoch [{epoch+1}/{args.epoch_num}], ROM, Step [{i}/{len(train_styled_loader)}], "
                          f"Loss: {loss.item():.4f}")

                del captions, logits, loss
                if i % 200 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        avg_factual_loss = factual_train_loss / factual_train_samples if factual_train_samples > 0 else 0.0
        avg_romantic_loss = romantic_train_loss / romantic_train_samples if romantic_train_samples > 0 else 0.0
        print(f"\n[EPOCH {epoch+1}] Factual Training Loss:  {avg_factual_loss:.4f}")
        print(f"[EPOCH {epoch+1}] Romantic Training Loss: {avg_romantic_loss:.4f}")

        print(f"[EPOCH {epoch+1}] Running validation...")
        val_loss = validate_epoch(encoder, decoder, val_loader, val_styled_loader, device, pad_token_id)
        print(f"[EPOCH {epoch+1}] Factual Validation Loss (early stopping): {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint_safely(best_model_path, encoder, decoder, optimizer_cap, optimizer_lang,
                                    epoch, best_val_loss, avg_factual_loss, avg_romantic_loss,
                                    val_loss, patience_counter)
            print(f"[EPOCH {epoch+1}] New best model saved! Factual val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"[EPOCH {epoch+1}] No improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"[EARLY STOPPING] No improvement for {args.patience} epochs.")
                print(f"Best factual validation loss was: {best_val_loss:.4f}")
                break

        # Single overwritten "latest" checkpoint — no per-epoch accumulation on disk
        save_checkpoint_safely(checkpoint_path, encoder, decoder, optimizer_cap, optimizer_lang,
                                epoch, best_val_loss, avg_factual_loss, avg_romantic_loss,
                                val_loss, patience_counter)

        gc.collect()
        torch.cuda.empty_cache()
        print(f"[EPOCH {epoch+1}] Checkpoint saved. Files in folder: {os.listdir(permanent_save_folder)}")

    if os.path.exists(best_model_path):
        print(f"\nTraining completed. Loading best model (factual val loss: {best_val_loss:.4f})...")
        best_checkpoint = torch.load(best_model_path, map_location=device)
        encoder.load_state_dict(best_checkpoint['encoder_state_dict'], strict=False)
        decoder.load_state_dict(best_checkpoint['decoder_state_dict'])
        print("Best model loaded successfully!")
    else:
        print("No best model found, using current model weights.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StyleNet Bangla (Transformer decoder)')
    parser.add_argument('--model_path', type=str, default='pretrained_models')
    parser.add_argument('--img_path', type=str, default='/kaggle/input/datasets/kaggleperfect/dataset/data/Images')
    parser.add_argument('--factual_caption_path', type=str,
                         default='/kaggle/input/datasets/kaggleperfect/dataset/data/factual_caption.txt')
    parser.add_argument('--romantic_caption_path', type=str,
                         default='/kaggle/input/datasets/kaggleperfect/dataset/data/romantic_data.txt')
    parser.add_argument('--caption_batch_size', type=int, default=32)   # lowered from 64: GPT2 backbone is heavier per-step than the old LSTM
    parser.add_argument('--language_batch_size', type=int, default=48)  # lowered from 96, same reason
    parser.add_argument('--factored_dim', type=int, default=512)
    parser.add_argument('--lr_caption', type=float, default=0.00002)
    parser.add_argument('--lr_language', type=float, default=0.00005)
    parser.add_argument('--epoch_num', type=int, default=80)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--log_step_caption', type=int, default=200)
    parser.add_argument('--log_step_language', type=int, default=100)
    args = parser.parse_args()
    main(args)
